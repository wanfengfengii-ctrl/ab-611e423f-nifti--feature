"""Trilinear interpolation and domain tests for app.sampling."""
import math
import unittest

from app.errors import PointError
from app.nifti import parse_nifti
from app.sampling import sample_point, sample_point_gradient
from verify.nifti_gen import build_nifti


def linear_fn(i, j, k):
    return i + 10 * j + 100 * k


def make_vol(dims=(3, 3, 3), data_fn=linear_fn, **kw):
    kw.setdefault("datatype", "float32")
    return parse_nifti(build_nifti(dims=dims, data_fn=data_fn, **kw))


class InterpolationTests(unittest.TestCase):
    def test_voxel_center_exact(self):
        vol = make_vol()
        voxel, value = sample_point(vol, (1.0, 1.0, 1.0))
        self.assertEqual(voxel, (1.0, 1.0, 1.0))
        self.assertEqual(value, 111.0)

    def test_trilinear_average_of_eight(self):
        vol = make_vol(data_fn=lambda i, j, k: i * j * k)
        _, value = sample_point(vol, (0.5, 0.5, 0.5))
        self.assertAlmostEqual(value, 0.125)  # only corner (1,1,1)=1

    def test_trilinear_exact_for_linear_data(self):
        vol = make_vol()
        _, value = sample_point(vol, (0.25, 0.5, 0.75))
        self.assertAlmostEqual(value, 0.25 + 5.0 + 75.0)

    def test_boundary_axis_fixed_to_endpoint(self):
        vol = make_vol()
        # x on the far boundary: only the x=2 plane participates
        _, value = sample_point(vol, (2.0, 0.5, 0.5))
        self.assertAlmostEqual(value, (2 + 12 + 102 + 112) / 4.0)
        # corner voxel centre
        voxel, value = sample_point(vol, (2.0, 2.0, 2.0))
        self.assertEqual(value, 222.0)
        # origin voxel centre
        _, value = sample_point(vol, (0.0, 0.0, 0.0))
        self.assertEqual(value, 0.0)

    def test_single_voxel_axis(self):
        vol = make_vol(dims=(1, 3, 3))
        _, value = sample_point(vol, (0.0, 1.0, 1.0))
        self.assertEqual(value, 110.0)
        with self.assertRaises(PointError) as ctx:
            sample_point(vol, (0.5, 1.0, 1.0))
        self.assertEqual(ctx.exception.code, "out_of_bounds")

    def test_boundary_tolerance(self):
        vol = make_vol()
        _, value = sample_point(vol, (0.0, 0.0, 2.0 + 5e-7))
        self.assertEqual(value, 200.0)
        _, value = sample_point(vol, (0.0, 0.0, -5e-7))
        self.assertEqual(value, 0.0)
        for bad in (2.0 + 1e-4, -1e-4):
            with self.assertRaises(PointError) as ctx:
                sample_point(vol, (0.0, 0.0, bad))
            self.assertEqual(ctx.exception.code, "out_of_bounds")

    def test_out_of_bounds_reports_voxel(self):
        vol = make_vol()
        for point in ((-1.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 99.0)):
            with self.assertRaises(PointError) as ctx:
                sample_point(vol, point)
            self.assertEqual(ctx.exception.code, "out_of_bounds")
            self.assertIsNotNone(ctx.exception.voxel)

    def test_scaling_applied_before_interpolation(self):
        vol = make_vol(datatype="int16", data_fn=lambda i, j, k: 5,
                       slope=2.0, inter=3.0)
        _, value = sample_point(vol, (1.0, 1.0, 1.0))
        self.assertEqual(value, 13.0)

    def test_non_finite_data_flagged(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (1, 1, 1) else linear_fn(i, j, k)
        vol = make_vol(data_fn=nan_fn)
        with self.assertRaises(PointError) as ctx:
            sample_point(vol, (0.5, 0.5, 0.5))
        self.assertEqual(ctx.exception.code, "non_finite_data")
        self.assertIn("(1, 1, 1)", str(ctx.exception))

    def test_zero_weight_neighbour_not_read(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (1, 1, 1) else linear_fn(i, j, k)
        vol = make_vol(data_fn=nan_fn)
        # exactly on voxel (0,0,0): all other corners carry zero weight
        _, value = sample_point(vol, (0.0, 0.0, 0.0))
        self.assertEqual(value, 0.0)

    def test_infinite_data_flagged(self):
        def inf_fn(i, j, k):
            return float("inf") if (i, j, k) == (0, 0, 0) else linear_fn(i, j, k)
        vol = make_vol(data_fn=inf_fn)
        with self.assertRaises(PointError) as ctx:
            sample_point(vol, (0.5, 0.5, 0.5))
        self.assertEqual(ctx.exception.code, "non_finite_data")


class TransformMappingTests(unittest.TestCase):
    def test_qform_world_mapping(self):
        vol = make_vol(transform="qform", quatern=(0, 0, 0),
                       qoffset=(10, 20, 30), pixdim=(1, 2, 3, 4))
        voxel, value = sample_point(vol, (12.0, 23.0, 34.0))
        for got, want in zip(voxel, (1.0, 1.0, 1.0)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(value, 111.0)

    def test_sform_world_mapping(self):
        vol = make_vol(srow_x=(2, 0, 0, 10), srow_y=(0, 3, 0, 20),
                       srow_z=(0, 0, 4, 30))
        voxel, value = sample_point(vol, (11.0, 21.5, 32.0))
        for got, want in zip(voxel, (0.5, 0.5, 0.5)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(value, 55.5)


class GradientTests(unittest.TestCase):
    def test_linear_data_constant_gradient_identity_affine(self):
        vol = make_vol(dims=(3, 3, 3), data_fn=lambda i, j, k: i + 2 * j + 4 * k)
        for point in ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)):
            _, _, grad = sample_point_gradient(vol, point)
            self.assertEqual(grad, (1.0, 2.0, 4.0))

    def test_exact_boundary_takes_negative_side_cell(self):
        # values 0, 1, 4, 9 along i; the chosen cell decides d(i^2)/di
        vol = make_vol(dims=(4, 1, 1), data_fn=lambda i, j, k: i * i)
        cases = [
            ((0.0, 0.0, 0.0), 1.0),  # zero boundary -> positive cell (0, 1)
            ((0.5, 0.0, 0.0), 1.0),  # mid cell (0, 1)
            ((1.0, 0.0, 0.0), 1.0),  # exact boundary -> negative cell (0, 1)
            ((1.5, 0.0, 0.0), 3.0),  # mid cell (1, 2)
            ((2.0, 0.0, 0.0), 3.0),  # exact boundary -> negative cell (1, 2)
            ((3.0, 0.0, 0.0), 5.0),  # top boundary -> negative cell (2, 3)
        ]
        for point, want in cases:
            with self.subTest(point=point):
                _, _, grad = sample_point_gradient(vol, point)
                # single-voxel axes (ny = nz = 1) contribute zero components
                self.assertEqual(grad, (want, 0.0, 0.0))

    def test_gradient_includes_scaling(self):
        vol = make_vol(dims=(3, 3, 3), data_fn=linear_fn, datatype="int16",
                       slope=2.0, inter=5.0)
        _, intensity, grad = sample_point_gradient(vol, (1.0, 1.0, 1.0))
        self.assertEqual(intensity, 111.0 * 2.0 + 5.0)
        self.assertEqual(grad, (2.0, 20.0, 200.0))  # inter drops out

    def test_gradient_mapped_through_sform(self):
        vol = make_vol(dims=(3, 3, 3), data_fn=linear_fn,
                       srow_x=(2, 0, 0, 10), srow_y=(0, 3, 0, 20),
                       srow_z=(0, 0, 4, 30))
        # world point of voxel (1, 1, 1)
        _, _, grad = sample_point_gradient(vol, (12.0, 23.0, 34.0))
        self.assertAlmostEqual(grad[0], 0.5)
        self.assertAlmostEqual(grad[1], 10.0 / 3.0)
        self.assertAlmostEqual(grad[2], 25.0)

    def test_gradient_mapped_through_qform_rotation(self):
        quat = (0.0, 0.0, math.sqrt(0.5))  # rotz(+90 deg)
        vol = make_vol(dims=(3, 3, 3), data_fn=linear_fn, transform="qform",
                       quatern=quat, qoffset=(10, 20, 30),
                       pixdim=(1, 2, 3, 4))
        # world = (-3j+10, 2i+20, 4k+30); voxel (1, 1, 1) -> (7, 22, 34)
        _, _, grad = sample_point_gradient(vol, (7.0, 22.0, 34.0))
        # quaternion round-off in the rebuilt rotation: compare to 6 places
        self.assertAlmostEqual(grad[0], -10.0 / 3.0, places=6)
        self.assertAlmostEqual(grad[1], 0.5, places=6)
        self.assertAlmostEqual(grad[2], 25.0, places=6)

    def test_gradient_neighbourhood_non_finite_flagged(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (0, 1, 1) else linear_fn(i, j, k)
        vol = make_vol(dims=(3, 3, 3), data_fn=nan_fn)
        # intensity at the exact voxel centre reads only (1, 1, 1): finite
        _, intensity = sample_point(vol, (1.0, 1.0, 1.0))
        self.assertEqual(intensity, 111.0)
        # ... but the x-gradient cell (0, 1) at j = k = 1 reaches (0, 1, 1)
        with self.assertRaises(PointError) as ctx:
            sample_point_gradient(vol, (1.0, 1.0, 1.0))
        self.assertEqual(ctx.exception.code, "non_finite_data")
        self.assertIn("(0, 1, 1)", str(ctx.exception))
        self.assertEqual(ctx.exception.voxel, (1.0, 1.0, 1.0))

    def test_gradient_zero_weight_neighbour_not_read(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (0, 2, 1) else linear_fn(i, j, k)
        vol = make_vol(dims=(3, 3, 3), data_fn=nan_fn)
        # at (1, 1, 1) the y-stencil weight on j = 2 is zero: (0, 2, 1) unread
        _, _, grad = sample_point_gradient(vol, (1.0, 1.0, 1.0))
        self.assertEqual(grad, (1.0, 10.0, 100.0))

    def test_gradient_out_of_bounds(self):
        vol = make_vol()
        with self.assertRaises(PointError) as ctx:
            sample_point_gradient(vol, (-1.0, 0.0, 0.0))
        self.assertEqual(ctx.exception.code, "out_of_bounds")
        self.assertIsNotNone(ctx.exception.voxel)


if __name__ == "__main__":
    unittest.main()
