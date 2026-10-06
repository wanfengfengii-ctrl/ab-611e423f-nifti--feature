"""One-shot verification service.

Aggregates three stages into the process exit code (bit flags):

* bit 0 (1): unit/integration tests (``tests/``) failed
* bit 1 (2): image build validation failed (manifest, runtime, imports)
* bit 2 (4): API smoke failed (big/little endian x sform/qform x
  int16/float32 sample matrix, world-gradient and boundary-convention
  checks, plus negative cases)

Exit code 0 means every stage passed.  The smoke stage waits for the API
service to report readiness on ``/healthz`` before sending traffic.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify.httpclient import post_sample  # noqa: E402
from verify.nifti_gen import build_nifti  # noqa: E402

EXIT_TESTS = 1
EXIT_IMAGE = 2
EXIT_SMOKE = 4

# ---------------------------------------------------------------------------
# stage 1: code tests
# ---------------------------------------------------------------------------

def stage_tests():
    print("== stage 1/3: code tests ==", flush=True)
    suite = unittest.TestLoader().discover(start_dir=str(ROOT / "tests"),
                                           top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print(f"stage tests: {'PASS' if ok else 'FAIL'} "
          f"({result.testsRun} tests, {len(result.failures)} failures, "
          f"{len(result.errors)} errors)", flush=True)
    return ok


# ---------------------------------------------------------------------------
# stage 2: image build validation
# ---------------------------------------------------------------------------

def stage_image():
    print("== stage 2/3: image build validation ==", flush=True)
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    manifest_path = ROOT / "image-manifest.json"
    manifest = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check("image manifest present", True, str(manifest_path))
    except Exception as exc:
        check("image manifest present", False, f"{manifest_path}: {exc}")
    if manifest is not None:
        check("manifest name", manifest.get("name") == "nifti-sampler",
              repr(manifest.get("name")))
        check("manifest version", bool(manifest.get("version")),
              repr(manifest.get("version")))
        env_version = os.environ.get("APP_VERSION")
        if env_version is not None:
            check("manifest version matches APP_VERSION env",
                  manifest.get("version") == env_version,
                  f"manifest={manifest.get('version')!r} env={env_version!r}")
    check("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        import app.nifti  # noqa: F401
        import app.sampling  # noqa: F401
        import app.server  # noqa: F401
        check("app modules importable", True)
    except Exception as exc:
        check("app modules importable", False, repr(exc))
    try:
        app.server.self_check()
        check("sampler self-check", True)
    except Exception as exc:
        check("sampler self-check", False, repr(exc))

    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        suffix = f" — {detail}" if detail and not passed else ""
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{suffix}", flush=True)
    print(f"stage image: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# stage 3: API smoke
# ---------------------------------------------------------------------------

SFORM_AFFINE = ((2.0, 0.0, 0.0, 10.0),
                (0.0, 3.0, 0.0, 20.0),
                (0.0, 0.0, 4.0, 30.0))
# rotz(+90 deg) @ diag(2,3,4) + translation, written via the qform fields.
QFORM_AFFINE = ((0.0, -3.0, 0.0, 10.0),
                (2.0, 0.0, 0.0, 20.0),
                (0.0, 0.0, 4.0, 30.0))
QUATERN_90Z = (0.0, 0.0, math.sqrt(0.5))
DIMS = (4, 5, 6)
SLOPE, INTER = 2.0, 5.0


def data_fn(i, j, k):
    return i + 10 * j + 100 * k


def world_of(affine, voxel):
    i, j, k = voxel
    return tuple(affine[r][0] * i + affine[r][1] * j + affine[r][2] * k + affine[r][3]
                 for r in range(3))


def avg8():
    return sum(data_fn(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)) / 8.0


def wait_healthy(base_url, timeout_s=90.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/healthz", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def stage_smoke(base_url):
    print("== stage 3/3: API smoke ==", flush=True)
    failures = []
    total = 0

    def check(name, cond, detail=""):
        nonlocal total
        total += 1
        if not cond:
            failures.append(name)
        suffix = f" — {detail}" if detail and not cond else ""
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{suffix}", flush=True)
        return bool(cond)

    check("service becomes healthy", wait_healthy(base_url), base_url)
    if failures:
        return False

    # -- sample matrix: endianness x transform x datatype -------------------
    combos = [(e, t, d) for e in ("<", ">") for t in ("sform", "qform")
              for d in ("int16", "float32")]
    expected_raw = {1: data_fn(1, 2, 3), 2: avg8(), 3: data_fn(3, 4, 5)}
    expected_vox = {1: (1.0, 2.0, 3.0), 2: (0.5, 0.5, 0.5), 3: (3.0, 4.0, 5.0)}
    for endian, transform, datatype in combos:
        tag = f"{'LE' if endian == '<' else 'BE'}/{transform}/{datatype}"
        affine = SFORM_AFFINE if transform == "sform" else QFORM_AFFINE
        raw = build_nifti(
            endian=endian, datatype=datatype, dims=DIMS, data_fn=data_fn,
            transform=transform, slope=SLOPE, inter=INTER,
            srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1], srow_z=SFORM_AFFINE[2],
            quatern=QUATERN_90Z, qoffset=(10.0, 20.0, 30.0),
            pixdim=(1.0, 2.0, 3.0, 4.0),
        )
        voxels = {3: (3.0, 4.0, 5.0), 1: (1.0, 2.0, 3.0), 4: (-1.0, 0.0, 0.0),
                  2: (0.5, 0.5, 0.5), 5: (0.0, 0.0, 6.0)}
        points = [{"id": pid, "point": list(world_of(affine, v))}
                  for pid, v in voxels.items()]
        status, payload = post_sample(base_url, raw, points)
        if not check(f"{tag}: HTTP 200", status == 200, f"got {status}: {payload}"):
            continue
        check(f"{tag}: transform source", isinstance(payload, dict)
              and payload.get("transform") == transform,
              repr(payload and payload.get("transform")))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not check(f"{tag}: result count", isinstance(results, list)
                     and len(results) == 5, repr(results)):
            continue
        check(f"{tag}: request order preserved",
              [r.get("id") for r in results] == [3, 1, 4, 2, 5],
              repr([r.get("id") for r in results]))
        check(f"{tag}: no gradient without derivatives",
              all("gradient" not in r for r in results), repr(results))
        by_id = {r.get("id"): r for r in results}
        for pid in (1, 2, 3):
            r = by_id.get(pid, {})
            got_vox = r.get("voxel") or []
            vox_ok = r.get("status") == "ok" and len(got_vox) == 3 and all(
                abs(g - e) <= 1e-4 for g, e in zip(got_vox, expected_vox[pid]))
            check(f"{tag}: id {pid} voxel coords", vox_ok, repr(r))
            want = expected_raw[pid] * SLOPE + INTER
            check(f"{tag}: id {pid} intensity",
                  r.get("status") == "ok"
                  and abs((r.get("intensity") or 0.0) - want) <= 1e-3,
                  f"want {want}, got {r.get('intensity')!r}")
            check(f"{tag}: id {pid} transform field",
                  r.get("transform") == transform, repr(r.get("transform")))
        for pid in (4, 5):
            r = by_id.get(pid, {})
            check(f"{tag}: id {pid} out_of_bounds",
                  r.get("status") == "error"
                  and r.get("error", {}).get("code") == "out_of_bounds", repr(r))

    # -- derivatives=world_gradient: scaling x endianness x transforms --------
    # data_fn is linear with slope 2 applied, so the scaled voxel-space
    # gradient is constantly (2, 20, 200); mapped through each affine:
    expected_grad = {"sform": (1.0, 20.0 / 3.0, 50.0),
                     "qform": (-20.0 / 3.0, 1.0, 50.0)}
    for endian, transform, datatype in combos:
        tag = f"{'LE' if endian == '<' else 'BE'}/{transform}/{datatype}"
        affine = SFORM_AFFINE if transform == "sform" else QFORM_AFFINE
        raw = build_nifti(
            endian=endian, datatype=datatype, dims=DIMS, data_fn=data_fn,
            transform=transform, slope=SLOPE, inter=INTER,
            srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1], srow_z=SFORM_AFFINE[2],
            quatern=QUATERN_90Z, qoffset=(10.0, 20.0, 30.0),
            pixdim=(1.0, 2.0, 3.0, 4.0),
        )
        # exact interior boundary, mid-cell, and the top-corner boundary
        voxels = {1: (1.0, 2.0, 3.0), 2: (0.5, 0.5, 0.5), 3: (3.0, 4.0, 5.0)}
        points = [{"id": pid, "point": list(world_of(affine, v))}
                  for pid, v in voxels.items()]
        status, payload = post_sample(base_url, raw, points,
                                      derivatives="world_gradient")
        if not check(f"{tag}+grad: HTTP 200", status == 200,
                     f"got {status}: {payload}"):
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        by_id = {r.get("id"): r for r in results} \
            if isinstance(results, list) else {}
        want = expected_grad[transform]
        for pid in (1, 2, 3):
            r = by_id.get(pid, {})
            grad = r.get("gradient")
            ok = (r.get("status") == "ok" and isinstance(grad, list)
                  and len(grad) == 3 and all(math.isfinite(g) for g in grad)
                  and all(abs(g - w) <= 1e-3 for g, w in zip(grad, want)))
            check(f"{tag}+grad: id {pid} world gradient", ok, repr(r))

    # -- gradient cell conventions on exact voxel boundaries ------------------
    # d(i^2)/di differs between neighbouring cells, so the chosen cell is
    # observable in d/dR (slope 2 and pixdim 2 cancel out).
    raw = build_nifti(endian=">", datatype="float32", dims=DIMS,
                      data_fn=lambda i, j, k: i * i + 10 * j + 100 * k,
                      transform="sform", slope=SLOPE, inter=INTER,
                      srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1],
                      srow_z=SFORM_AFFINE[2])
    boundary_cases = [
        ((2.0, 1.0, 1.0), 3.0),   # interior boundary -> negative cell (1, 2)
        ((0.0, 1.0, 1.0), 1.0),   # zero boundary -> positive cell (0, 1)
        ((3.0, 1.0, 1.0), 5.0),   # top boundary -> negative cell (2, 3)
        ((1.5, 1.0, 1.0), 3.0),   # mid cell (1, 2)
    ]
    points = [{"id": n, "point": list(world_of(SFORM_AFFINE, v))}
              for n, (v, _) in enumerate(boundary_cases, start=1)]
    status, payload = post_sample(base_url, raw, points,
                                  derivatives="world_gradient")
    if check("grad boundaries: HTTP 200", status == 200,
             f"got {status}: {payload}"):
        results = payload.get("results") if isinstance(payload, dict) else []
        by_id = {r.get("id"): r for r in results} \
            if isinstance(results, list) else {}
        for n, (v, want_x) in enumerate(boundary_cases, start=1):
            r = by_id.get(n, {})
            grad = r.get("gradient") or []
            ok = (r.get("status") == "ok" and len(grad) == 3
                  and abs(grad[0] - want_x) <= 1e-3
                  and abs(grad[1] - 20.0 / 3.0) <= 1e-3
                  and abs(grad[2] - 50.0) <= 1e-3)
            check(f"grad boundaries: voxel {v} -> d/dR {want_x}", ok, repr(r))

    # -- single-voxel axes have zero gradient components -----------------------
    raw = build_nifti(endian="<", datatype="float32", dims=(3, 1, 1),
                      data_fn=lambda i, j, k: i * i, transform="sform")
    points = [{"id": 1, "point": [1.0, 0.0, 0.0]},
              {"id": 2, "point": [2.0, 0.0, 0.0]}]
    status, payload = post_sample(base_url, raw, points,
                                  derivatives="world_gradient")
    if check("grad single-voxel axes: HTTP 200", status == 200,
             f"got {status}: {payload}"):
        results = payload.get("results") if isinstance(payload, dict) else []
        grads = {r.get("id"): r.get("gradient") for r in results} \
            if isinstance(results, list) else {}
        check("grad single-voxel axes: degenerate components are zero",
              grads.get(1) == [1.0, 0.0, 0.0]
              and grads.get(2) == [3.0, 0.0, 0.0], repr(payload))

    # -- gradient neighbourhood with non-finite data fails only that point ----
    raw = build_nifti(endian="<", datatype="float32", dims=DIMS,
                      data_fn=lambda i, j, k: float("nan")
                      if (i, j, k) == (0, 1, 1) else data_fn(i, j, k),
                      transform="sform", slope=SLOPE, inter=INTER,
                      srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1],
                      srow_z=SFORM_AFFINE[2])
    points = [{"id": 1, "point": list(world_of(SFORM_AFFINE, (1.0, 1.0, 1.0)))},
              {"id": 2, "point": list(world_of(SFORM_AFFINE, (3.0, 4.0, 5.0)))}]
    status, payload = post_sample(base_url, raw, points,
                                  derivatives="world_gradient")
    if check("grad non-finite: HTTP 200", status == 200,
             f"got {status}: {payload}"):
        results = payload.get("results") if isinstance(payload, dict) else []
        by_id = {r.get("id"): r for r in results} \
            if isinstance(results, list) else {}
        r1, r2 = by_id.get(1, {}), by_id.get(2, {})
        grad2 = r2.get("gradient") or []
        check("grad non-finite: only the touching point errors",
              r1.get("status") == "error"
              and r1.get("error", {}).get("code") == "non_finite_data"
              and "voxel" in r1.get("error", {})
              and r2.get("status") == "ok"
              and r2.get("transform") == "sform"
              and len(grad2) == 3
              and abs(grad2[0] - 1.0) <= 1e-3,
              repr(payload))

    # -- unsupported derivatives mode is rejected ------------------------------
    status, payload = post_sample(base_url, raw, points[:1],
                                  derivatives="hessian")
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    check("derivatives unsupported mode: 400/invalid_derivatives",
          status == 400 and err.get("code") == "invalid_derivatives"
          and err.get("field") == "derivatives",
          f"got {status}: {payload}")

    # -- negative file-structure cases --------------------------------------
    base = dict(endian="<", datatype="float32", dims=DIMS, data_fn=data_fn,
                transform="sform",
                srow_x=SFORM_AFFINE[0], srow_y=SFORM_AFFINE[1], srow_z=SFORM_AFFINE[2])
    one_point = [{"id": 1, "point": list(world_of(SFORM_AFFINE, (1.0, 1.0, 1.0)))}]
    bad_files = [
        ("truncated payload", dict(truncate=2), "payload_length_mismatch", "file"),
        ("trailing bytes", dict(extra=b"\x00"), "trailing_bytes", "file"),
        ("no affine", dict(qform_code=0, sform_code=0), "missing_affine",
         "qform_code,sform_code"),
        ("singular sform", dict(srow_x=(0.0, 0.0, 0.0, 0.0)), "singular_affine", "srow"),
        ("unsupported datatype", dict(datatype_code=2), "unsupported_datatype", "datatype"),
        ("non-finite scl_slope", dict(slope=float("nan")), "non_finite_scaling", "scl_slope"),
        ("non-3D dims", dict(dim0=4, tail_dims=(2, 1, 1, 1)), "invalid_dimensions", "dim"),
        ("Analyze magic", dict(magic=b"ni1\x00"), "unsupported_magic", "magic"),
        ("bad vox_offset", dict(vox_offset=100.0), "invalid_vox_offset", "vox_offset"),
        ("bitpix mismatch", dict(bitpix=8), "bitpix_mismatch", "bitpix"),
    ]
    for name, overrides, code, field in bad_files:
        raw = build_nifti(**{**base, **overrides})
        status, payload = post_sample(base_url, raw, one_point)
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        check(f"negative[{name}]: 400/{code}",
              status == 400 and err.get("code") == code and err.get("field") == field,
              f"got {status}: {payload}")

    # big-endian structural error is detected identically
    raw = build_nifti(**{**base, "endian": ">", "datatype": "int16", "extra": b"\x00"})
    status, payload = post_sample(base_url, raw, one_point)
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    check("negative[BE trailing]: 400/trailing_bytes",
          status == 400 and err.get("code") == "trailing_bytes",
          f"got {status}: {payload}")

    # -- non-finite voxel data -> per-point error ----------------------------
    raw = build_nifti(**{**base,
                         "data_fn": lambda i, j, k: float("nan") if (i, j, k) == (1, 1, 1)
                         else data_fn(i, j, k)})
    points = [
        {"id": 10, "point": list(world_of(SFORM_AFFINE, (0.5, 0.5, 0.5)))},
        {"id": 11, "point": list(world_of(SFORM_AFFINE, (3.0, 4.0, 5.0)))},
    ]
    status, payload = post_sample(base_url, raw, points)
    results = {r.get("id"): r for r in payload.get("results", [])} \
        if isinstance(payload, dict) else {}
    r10, r11 = results.get(10, {}), results.get(11, {})
    check("non-finite data: point 10 flagged",
          status == 200 and r10.get("status") == "error"
          and r10.get("error", {}).get("code") == "non_finite_data",
          f"got {status}: {payload}")
    check("non-finite data: point 11 unaffected",
          r11.get("status") == "ok"
          and abs((r11.get("intensity") or 0.0) - data_fn(3, 4, 5)) <= 1e-3,
          repr(r11))

    # -- invalid points payloads ---------------------------------------------
    raw = build_nifti(**base)
    bad_points = [
        ("duplicate ids", [{"id": 1, "point": [0, 0, 0]}, {"id": 1, "point": [1, 1, 1]}]),
        ("zero points", []),
        ("257 points", [{"id": i, "point": [0, 0, 0]} for i in range(257)]),
        ("non-finite coord", [{"id": 1, "point": [float("inf"), 0, 0]}]),
        ("malformed JSON", b"[{"),
    ]
    for name, points_payload in bad_points:
        status, payload = post_sample(base_url, raw, points_payload)
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        check(f"negative[points {name}]: 400/invalid_points",
              status == 400 and err.get("code") == "invalid_points"
              and err.get("field") == "points",
              f"got {status}: {payload}")

    ok = not failures
    print(f"stage smoke: {'PASS' if ok else 'FAIL'} "
          f"({total - len(failures)}/{total} checks passed)", flush=True)
    return ok


# ---------------------------------------------------------------------------

def main():
    base_url = os.environ.get("VERIFY_BASE_URL", "http://127.0.0.1:8000")
    print(f"verify: base_url={base_url}", flush=True)
    code = 0
    if not stage_tests():
        code |= EXIT_TESTS
    if not stage_image():
        code |= EXIT_IMAGE
    if not stage_smoke(base_url):
        code |= EXIT_SMOKE
    print(f"verify: exit code {code} (1=tests, 2=image, 4=smoke)", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
