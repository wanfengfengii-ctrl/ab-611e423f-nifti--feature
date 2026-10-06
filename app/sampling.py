"""Trilinear interpolation of scaled voxel data in continuous voxel space,
plus the gradient of that field with respect to the RAS world axes."""
from __future__ import annotations

import math

from .errors import PointError

# Tolerance (in voxels) absorbing float64 round-off when a world coordinate
# that sits exactly on the voxel-center boundary is mapped back through the
# inverse affine.  Points further outside than this are rejected.
BOUND_TOL = 1e-6


def _axis_stencil(fc, n):
    """Return ``((index, weight), ...)`` for one axis of the trilinear stencil.

    ``fc`` must lie in the closed voxel-center domain ``[0, n-1]`` (plus the
    round-off tolerance).  On the boundary the axis collapses to the unique
    endpoint with weight 1.
    """
    hi = n - 1
    if not math.isfinite(fc) or fc < -BOUND_TOL or fc > hi + BOUND_TOL:
        raise PointError(
            "out_of_bounds",
            f"continuous voxel coordinate {fc!r} is outside the closed "
            f"voxel-center domain [0, {hi}]",
        )
    fc = min(max(fc, 0.0), float(hi))
    i0 = math.floor(fc)
    t = fc - i0
    if i0 >= hi:  # boundary: axis fixed to the unique endpoint
        return ((hi, 1.0),)
    return ((i0, 1.0 - t), (i0 + 1, t))


def _axis_cell(fc, n):
    """Return the cell ``(lo, lo + 1)`` used to differentiate one axis.

    Requires ``n >= 2`` and ``fc`` inside the closed voxel-center domain
    (round-off tolerance included).  The trilinear field is only piecewise
    linear, so when ``fc`` sits exactly on a voxel boundary the cell on the
    negative voxel side is chosen — except at the zero boundary, where only
    the positive-side cell exists.
    """
    hi = n - 1
    fc = min(max(fc, 0.0), float(hi))
    i0 = math.floor(fc)
    if fc == float(i0):
        # exact voxel boundary: negative-side cell (positive side at zero)
        return (i0 - 1, i0) if i0 > 0 else (0, 1)
    return i0, i0 + 1


def sample_point(volume, point):
    """Sample ``volume`` at RAS world point ``(x, y, z)``.

    Returns ``((fi, fj, fk), intensity)``: the continuous voxel coordinate
    produced by the inverse affine and the trilinearly interpolated,
    scaling-applied intensity.  Raises :class:`PointError` for out-of-bounds
    points and for non-finite scaled data inside the stencil.
    """
    x, y, z = point
    inv = volume.inverse
    fi = inv[0][0] * x + inv[0][1] * y + inv[0][2] * z + inv[0][3]
    fj = inv[1][0] * x + inv[1][1] * y + inv[1][2] * z + inv[1][3]
    fk = inv[2][0] * x + inv[2][1] * y + inv[2][2] * z + inv[2][3]
    voxel = (fi, fj, fk)

    nx, ny, nz = volume.dims
    try:
        xs = _axis_stencil(fi, nx)
        ys = _axis_stencil(fj, ny)
        zs = _axis_stencil(fk, nz)
    except PointError as exc:
        exc.voxel = voxel
        raise

    total = 0.0
    for ix, wx in xs:
        if wx == 0.0:
            continue
        for iy, wy in ys:
            if wy == 0.0:
                continue
            for iz, wz in zs:
                w = wx * wy * wz
                if w == 0.0:
                    continue  # zero-weight neighbours do not participate
                value = volume.value_at(ix, iy, iz)
                if not math.isfinite(value):
                    raise PointError(
                        "non_finite_data",
                        f"non-finite scaled data at voxel ({ix}, {iy}, {iz})",
                        voxel=voxel,
                    )
                total += w * value
    return voxel, total


def sample_point_gradient(volume, point):
    """Sample like :func:`sample_point` and also differentiate the field.

    Returns ``((fi, fj, fk), intensity, (gx, gy, gz))`` where the gradient is
    the rate of change of the local trilinear, scaling-applied intensity
    field per RAS world-coordinate unit (R, A, S order), mapped from voxel
    space through the volume's chosen affine.  A single-voxel axis
    contributes a zero component.  Raises :class:`PointError` for
    out-of-bounds points and when the intensity stencil or any gradient cell
    touches non-finite scaled data.
    """
    voxel, intensity = sample_point(volume, point)
    dims = volume.dims
    # Bounds were just validated by sample_point; these cannot raise.
    stencils = tuple(_axis_stencil(voxel[a], dims[a]) for a in range(3))
    grad_voxel = []
    for axis in range(3):
        n = dims[axis]
        if n < 2:
            grad_voxel.append(0.0)  # single-voxel axis carries no variation
            continue
        lo, hi = _axis_cell(voxel[axis], n)
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        deriv = 0.0
        for i0, w0 in ((lo, -1.0), (hi, 1.0)):
            for i1, w1 in stencils[a1]:
                if w1 == 0.0:
                    continue
                for i2, w2 in stencils[a2]:
                    if w2 == 0.0:
                        continue  # zero-weight neighbours do not participate
                    idx = [0, 0, 0]
                    idx[axis], idx[a1], idx[a2] = i0, i1, i2
                    i, j, k = idx
                    value = volume.value_at(i, j, k)
                    if not math.isfinite(value):
                        raise PointError(
                            "non_finite_data",
                            f"non-finite scaled data at voxel ({i}, {j}, {k})",
                            voxel=voxel,
                        )
                    deriv += w0 * w1 * w2 * value
        grad_voxel.append(deriv)
    # voxel = inverse @ world, so grad_world = inverse^T @ grad_voxel
    inv = volume.inverse
    grad_world = tuple(
        inv[0][a] * grad_voxel[0] + inv[1][a] * grad_voxel[1]
        + inv[2][a] * grad_voxel[2]
        for a in range(3)
    )
    return voxel, intensity, grad_world
