"""Trilinear interpolation of scaled voxel data in continuous voxel space."""
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


def _gradient_axis_span(fc, n):
    """Select the trilinear cell used for one axis of the finite gradient.

    Returns ``(lo, t)`` for the cell spanning voxel indices ``[lo, lo + 1]``
    with local coordinate ``t`` in ``[0, 1]``.

    Cell selection at grid nodes is deliberately one-sided (a left finite
    difference): when the point falls exactly on a voxel boundary the cell
    on its negative-voxel side is used (``lo = i - 1``, ``t = 1``); the
    zero-side boundary has no negative cell and instead takes the positive
    side (``lo = 0``, ``t = 0``); the far boundary takes its unique negative
    cell.  An axis one voxel thick (``n == 1``) collapses and contributes no
    gradient component.
    """
    hi = n - 1
    fc = min(max(fc, 0.0), float(hi))
    if hi == 0:                       # single-voxel axis: no gradient
        return 0, 0.0
    lo = math.floor(fc)
    t = fc - lo
    if lo >= hi:                      # far boundary -> negative-side cell
        return hi - 1, 1.0
    if t == 0.0 and lo > 0:           # interior node -> negative-side cell
        return lo - 1, 1.0
    return lo, t


def _finite_or_gradient_error(volume, ix, iy, iz, voxel):
    value = volume.value_at(ix, iy, iz)
    if not math.isfinite(value):
        raise PointError(
            "non_finite_gradient",
            f"non-finite scaled data at gradient neighbour voxel "
            f"({ix}, {iy}, {iz})",
            voxel=voxel,
        )
    return value


def world_gradient(volume, voxel):
    """Finite gradient of the local trilinear intensity field at ``voxel``.

    ``voxel`` is the continuous voxel coordinate produced by the inverse
    affine (it must lie in the closed voxel-centre domain, as for sampling).
    The returned vector is ordered R, A, S and is the rate of change of the
    scaling-applied intensity per unit of *world* coordinate:

        g_world = M^{-T} g_voxel,

    where ``M`` is the chosen sform/qform affine and ``g_voxel`` holds the
    finite voxel-axis differences of the trilinear cell.  Boundary cell
    selection follows :func:`_gradient_axis_span`; a single-voxel axis has a
    zero component.  Any non-finite scaled datum in the cell's 2x2x2
    neighbourhood raises :class:`PointError` ``non_finite_gradient`` carrying
    the continuous voxel coordinate.
    """
    fi, fj, fk = voxel
    nx, ny, nz = volume.dims
    i0, u = _gradient_axis_span(fi, nx)
    j0, v = _gradient_axis_span(fj, ny)
    k0, w = _gradient_axis_span(fk, nz)

    # Index pair per axis: on a single-voxel axis both ends coincide, so the
    # cell collapses and the corresponding gradient component is zero.
    ii = (i0, i0 + 1) if nx > 1 else (i0, i0)
    jj = (j0, j0 + 1) if ny > 1 else (j0, j0)
    kk = (k0, k0 + 1) if nz > 1 else (k0, k0)

    # Scaled values of the (up to) eight cell corners, indexed (i-lo, ...).
    # Every corner participates in a difference, so non-finite data anywhere
    # in the 2x2x2 neighbourhood is a locatable per-point error.
    c = {}
    for di, ix in enumerate(ii):
        for dj, iy in enumerate(jj):
            for dk, iz in enumerate(kk):
                c[(di, dj, dk)] = _finite_or_gradient_error(
                    volume, ix, iy, iz, voxel)

    # Partial derivative along each voxel axis of the trilinear field.
    if nx > 1:
        gi = (((c[(1, 0, 0)] - c[(0, 0, 0)]) * (1.0 - v) * (1.0 - w))
              + ((c[(1, 0, 1)] - c[(0, 0, 1)]) * (1.0 - v) * w)
              + ((c[(1, 1, 0)] - c[(0, 1, 0)]) * v * (1.0 - w))
              + ((c[(1, 1, 1)] - c[(0, 1, 1)]) * v * w))
    else:
        gi = 0.0
    if ny > 1:
        gj = (((c[(0, 1, 0)] - c[(0, 0, 0)]) * (1.0 - u) * (1.0 - w))
              + ((c[(0, 1, 1)] - c[(0, 0, 1)]) * (1.0 - u) * w)
              + ((c[(1, 1, 0)] - c[(1, 0, 0)]) * u * (1.0 - w))
              + ((c[(1, 1, 1)] - c[(1, 0, 1)]) * u * w))
    else:
        gj = 0.0
    if nz > 1:
        gk = (((c[(0, 0, 1)] - c[(0, 0, 0)]) * (1.0 - u) * (1.0 - v))
              + ((c[(0, 1, 1)] - c[(0, 1, 0)]) * (1.0 - u) * v)
              + ((c[(1, 0, 1)] - c[(1, 0, 0)]) * u * (1.0 - v))
              + ((c[(1, 1, 1)] - c[(1, 1, 0)]) * u * v))
    else:
        gk = 0.0

    if not (math.isfinite(gi) and math.isfinite(gj) and math.isfinite(gk)):
        raise PointError(
            "non_finite_gradient",
            "finite gradient of the trilinear field is not finite",
            voxel=voxel,
        )

    # World = M voxel + t, so d(intensity)/d(world) = M^{-T} g_voxel.
    inv = volume.inverse
    return (
        inv[0][0] * gi + inv[1][0] * gj + inv[2][0] * gk,
        inv[0][1] * gi + inv[1][1] * gj + inv[2][1] * gk,
        inv[0][2] * gi + inv[1][2] * gj + inv[2][2] * gk,
    )
