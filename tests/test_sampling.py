"""Trilinear interpolation and domain tests for app.sampling."""
import unittest

from app.errors import PointError
from app.nifti import parse_nifti
from app.sampling import sample_point, world_gradient
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


class GradientTests(unittest.TestCase):
    def assert_vec_close(self, got, want, places=6):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=places)

    def test_identity_affine_voxel_gradient(self):
        vol = make_vol()
        # identity affine: world gradient == voxel-axis gradient
        g = world_gradient(vol, (0.25, 0.5, 0.75))
        self.assert_vec_close(g, (1.0, 10.0, 100.0))

    def test_sform_diagonal_world_gradient(self):
        vol = make_vol(srow_x=(2, 0, 0, 10), srow_y=(0, 3, 0, 20),
                       srow_z=(0, 0, 4, 30), slope=2.0, inter=5.0)
        voxel, _ = sample_point(vol, (11.0, 21.5, 32.0))
        g = world_gradient(vol, voxel)
        # scaled voxel gradient (2,20,200); divide by zooms (2,3,4)
        self.assert_vec_close(g, (1.0, 20.0 / 3.0, 50.0))

    def test_qform_rotated_world_gradient(self):
        import math
        # rotz(+90 deg) @ diag(2,3,4) + t
        vol = make_vol(transform="qform", quatern=(0, 0, math.sqrt(0.5)),
                       qoffset=(10, 20, 30), pixdim=(1, 2, 3, 4))
        voxel, _ = sample_point(vol, (4.0, 22.0, 34.0))  # voxel (1,2,3)
        g = world_gradient(vol, voxel)
        # M^-T (1, 10, 100)^T = (-10/3, 1/2, 25)
        self.assert_vec_close(g, (-10.0 / 3.0, 0.5, 25.0), places=4)

    def test_gradient_constant_inside_cell(self):
        vol = make_vol(data_fn=lambda i, j, k: 7.0)
        for voxel in ((0.0, 0.0, 0.0), (0.5, 0.2, 0.9), (2.0, 2.0, 2.0)):
            self.assert_vec_close(world_gradient(vol, voxel), (0.0, 0.0, 0.0))

    def test_boundary_uses_negative_side_cell(self):
        # Far boundary (2,2,2): cell must be [1,2]^3, not include index 0.
        vol = make_vol()
        g = world_gradient(vol, (2.0, 2.0, 2.0))
        self.assert_vec_close(g, (1.0, 10.0, 100.0))
        # a poisoned voxel outside the negative-side cell is irrelevant
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (0, 0, 0) else linear_fn(i, j, k)
        vol_nan = make_vol(data_fn=nan_fn)
        g = world_gradient(vol_nan, (2.0, 2.0, 2.0))
        self.assert_vec_close(g, (1.0, 10.0, 100.0))

    def test_interior_node_uses_negative_side_cell(self):
        # On non-linear data the two adjacent cells give different finite
        # differences at the shared node: f(i)=i^2 -> left diff at i=1 is
        # f(1)-f(0)=1; the positive-side cell would give f(2)-f(1)=3.
        vol = make_vol(dims=(4, 4, 4), data_fn=lambda i, j, k: float(i * i))
        g = world_gradient(vol, (1.0, 0.0, 0.0))
        self.assert_vec_close(g, (1.0, 0.0, 0.0))
        # and the gradient must read the negative-side neighbourhood only:
        # NaN at (2,0,0) (positive side) is irrelevant, NaN at (0,0,0) errs
        def nan_pos(i, j, k):
            return float("nan") if (i, j, k) == (2, 0, 0) else float(i * i)
        vol_nan = make_vol(dims=(4, 4, 4), data_fn=nan_pos)
        g = world_gradient(vol_nan, (1.0, 0.0, 0.0))
        self.assert_vec_close(g, (1.0, 0.0, 0.0))

        def nan_neg(i, j, k):
            return float("nan") if (i, j, k) == (0, 0, 0) else float(i * i)
        with self.assertRaises(PointError) as ctx:
            world_gradient(make_vol(dims=(4, 4, 4), data_fn=nan_neg),
                           (1.0, 0.0, 0.0))
        self.assertEqual(ctx.exception.code, "non_finite_gradient")

    def test_zero_boundary_uses_positive_side_cell(self):
        vol = make_vol()
        g = world_gradient(vol, (0.0, 0.0, 0.0))
        self.assert_vec_close(g, (1.0, 10.0, 100.0))

    def test_single_voxel_axis_component_zero(self):
        vol = make_vol(dims=(1, 3, 3))
        g = world_gradient(vol, (0.0, 1.0, 1.0))
        self.assert_vec_close(g, (0.0, 10.0, 100.0))
        vol = make_vol(dims=(3, 1, 3))
        g = world_gradient(vol, (1.0, 0.0, 1.0))
        self.assert_vec_close(g, (1.0, 0.0, 100.0))
        vol = make_vol(dims=(3, 3, 1))
        g = world_gradient(vol, (1.0, 1.0, 0.0))
        self.assert_vec_close(g, (1.0, 10.0, 0.0))

    def test_non_linear_field_cell_local_weighting(self):
        # f = i*j*k: derivative along i at (0.5,0.5,0.5) equals v*w = 0.25
        vol = make_vol(data_fn=lambda i, j, k: i * j * k)
        g = world_gradient(vol, (0.5, 0.5, 0.5))
        self.assert_vec_close(g, (0.25, 0.25, 0.25))

    def test_non_finite_gradient_neighbour_is_locatable_error(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (1, 1, 1) else linear_fn(i, j, k)
        vol = make_vol(data_fn=nan_fn)
        # sampling at (0,0,0) still succeeds (zero-weight neighbour skipped)
        voxel, value = sample_point(vol, (0.0, 0.0, 0.0))
        self.assertEqual(value, 0.0)
        with self.assertRaises(PointError) as ctx:
            world_gradient(vol, voxel)
        self.assertEqual(ctx.exception.code, "non_finite_gradient")
        self.assertEqual(ctx.exception.voxel, (0.0, 0.0, 0.0))
        self.assertIn("(1, 1, 1)", str(ctx.exception))

    def test_non_finite_gradient_at_far_boundary(self):
        # negative-side cell at (2,2,2) reads voxel (1,1,1): NaN there flags
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (1, 1, 1) else linear_fn(i, j, k)
        vol = make_vol(data_fn=nan_fn)
        with self.assertRaises(PointError) as ctx:
            world_gradient(vol, (2.0, 2.0, 2.0))
        self.assertEqual(ctx.exception.code, "non_finite_gradient")

    def test_scaling_counts_into_gradient(self):
        vol = make_vol(datatype="int16", slope=3.0, inter=1000.0)
        g = world_gradient(vol, (0.5, 0.5, 0.5))
        self.assert_vec_close(g, (3.0, 30.0, 300.0))


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


if __name__ == "__main__":
    unittest.main()
