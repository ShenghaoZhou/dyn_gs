"""
Tests for coverage-driven keyframe selection (full_system_ow_working/
keyframe_coverage.py) and its wiring into the three runners.

Nothing here touches a dataset or a checkpoint. The descriptor tests are pure
numpy on hand-drawn silhouette masks, so every threshold's meaning can be
asserted against a mask whose shape is known. The wiring tests parse the runner
sources with ast, because importing them boots the dataset loader and the rerun
server. Run with:

    python -m pytest test/test_coverage_kf.py -q
or:
    python test/test_coverage_kf.py
"""

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "full_system_ow_working"))

from keyframe_coverage import (KeyframeCoverage, ViewDescriptor, _angle_diff_deg,  # noqa: E402
                               silhouette_descriptor, view_distance)


W, H = 640, 480


def _needles(angle_deg=0.0, cx=0.0, cy=0.0, half=0.20, thick=0.02, n=4000, seed=0):
    """Mask of a long thin silhouette, rotated by angle_deg about its own centre
    and shifted by (cx, cy) in normalised image coordinates.

    Thin enough and small enough that no shift used by these tests clips at the
    image border -- clipping would change the measured area and confound the
    centroid assertions below.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(-half, half, n)
    w = rng.normal(0.0, thick, n)
    th = np.deg2rad(angle_deg)
    x = t * np.cos(th) - w * np.sin(th)
    y = t * np.sin(th) + w * np.cos(th)
    ix = np.clip(np.round(W * 0.5 + (cx + x) * W).astype(int), 0, W - 1)
    iy = np.clip(np.round(H * 0.5 + (cy + y) * H).astype(int), 0, H - 1)
    m = np.zeros((H, W), dtype=np.uint8)
    m[iy, ix] = 1
    return m


def _desc(mask):
    return silhouette_descriptor(mask, W, H)


def _cov(**overrides):
    base = dict(max_window=5, min_interval=1, angle_deg=15.0,
                centroid_shift=0.25, shape_ratio=0.5)
    base.update(overrides)
    return KeyframeCoverage(**base)


# --------------------------------------------------------------------------- #
# descriptor
# --------------------------------------------------------------------------- #

def test_empty_mask_returns_none():
    """Nothing to key on: the caller must see a clean no-keyframe, not a crash."""
    assert _desc(np.zeros((H, W), dtype=np.uint8)) is None
    assert _desc(np.ones((H, W), dtype=np.float32) * 0.0) is None
    assert _desc(None) is None


def _disc(cx=0.0, cy=0.0, r=0.20):
    """Filled circle in normalised image coordinates, centred at (cx, cy).

    A circle is a circle in normalised space regardless of the pixel aspect
    ratio, so unlike a needle it has no strongly preferred axis -- the shape
    term must carry the weight in the distance, not the orientation term.
    """
    ys, xs = np.mgrid[0:H, 0:W]
    x = (xs - W * 0.5) / W - cx
    y = (ys - H * 0.5) / H - cy
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def test_full_frame_mask_is_centred_and_near_isotropic():
    """Degenerate but reachable: a whole-frame mask has centroid at the image
    centre, a well-defined axis along the major dimension, and eccentricity that
    is small but not zero.

    It is NOT exactly isotropic: a 640x480 grid samples the unit square at
    different rates along each axis, and the discrete variance of a grid of n
    points is (n**2-1)/(12n**2) rather than 1/12. The two axes therefore differ
    by about 1e-3 in eccentricity -- real discretisation, not a bug, so the
    assertion is a bound rather than an equality.
    """
    d = _desc(np.ones((H, W), dtype=np.uint8))
    assert d is not None
    # A pixel grid's centre sits a half pixel off the image centre, so this is
    # not exactly zero -- the tolerance is half a pixel in normalised units.
    np.testing.assert_allclose(d.centroid, 0.0, atol=2.0 / min(W, H))
    assert -90.0 < d.angle_deg <= 90.0
    assert d.angle_deg == pytest.approx(0.0, abs=1e-6)
    assert d.area_frac == pytest.approx(1.0)
    assert d.ecc < 1e-2


def test_axisymmetric_mask_has_a_deterministic_orientation():
    """A silhouette with a symmetry axis must report that axis, not a
    rounding-noise function of the mask.

    A centred circle is symmetric under x -> -x and y -> -y, so sxy is exactly
    zero and the orientation comes from atan2(0, sxx-syy): the major axis.
    That is stable across shifts and rotation, which is what keeps two identical
    round silhouettes from reading as the most different views possible.
    """
    a = _desc(_disc())
    b = _desc(_disc(cx=0.05))
    c = _desc(_disc(cy=0.05))
    for d in (a, b, c):
        assert -90.0 < d.angle_deg <= 90.0
    # Same shape, silhouette shifted: the centroid shift is the only difference.
    assert view_distance(a, b) == pytest.approx(0.05, abs=2e-3)
    # And a nearly-round mask is a much closer view than a needle rotated 45.
    assert view_distance(a, _desc(_needles(45.0))) > view_distance(a, b)


def test_angle_is_flip_invariant():
    """A silhouette is a line, not an arrow: rotating it by 180 degrees must
    give the same orientation. This is what makes the metric usable for a
    manipulated object, which turns over under the camera."""
    for a in (-80.0, -30.0, 0.0, 20.0, 70.0):
        assert _desc(_needles(a)).angle_deg == pytest.approx(
            _desc(_needles(a + 180.0)).angle_deg, abs=0.5), f"angle {a}"


def test_angle_tracks_the_rotation():
    """The orientation must actually move with the silhouette, or the coverage
    test cannot distinguish views at all."""
    a, b = _desc(_needles(0.0)), _desc(_needles(30.0))
    assert _angle_diff_deg(a.angle_deg, b.angle_deg) == pytest.approx(30.0, abs=0.5)


def test_angle_range_is_minus90_90():
    """Each axis appears exactly once in (-90, 90]; a mod 90 fold would identify
    perpendicular silhouettes and no test could tell the difference."""
    for a in (-89.0, -45.0, 0.0, 45.0, 89.0):
        d = _desc(_needles(a))
        assert -90.0 < d.angle_deg <= 90.0


def test_centroid_is_relative_to_image_centre():
    """The whole point of a relative centroid: a silhouette that stays put while
    the object drifts must register as movement."""
    assert np.linalg.norm(_desc(_needles(cx=0.0)).centroid) < 5e-3
    # A 4000-sample silhouette has a sample-centre jitter of about 2e-3 in
    # normalised units, so this tolerance is that jitter plus rounding.
    np.testing.assert_allclose(_desc(_needles(cx=0.1)).centroid,
                               [0.1, 0.0], atol=5e-3)
    np.testing.assert_allclose(_desc(_needles(cy=0.1)).centroid,
                               [0.0, 0.1], atol=5e-3)


def test_descriptor_is_independent_of_pixel_count():
    """Denser sampling of the same silhouette must not read as a different view.
    A scale-free descriptor is what makes the thresholds stable across frames
    with different mask densities."""
    d0, d1 = _desc(_needles(n=3000)), _desc(_needles(n=9000))
    np.testing.assert_allclose(d0.centroid, d1.centroid, atol=3e-3)
    assert d0.angle_deg == pytest.approx(d1.angle_deg, abs=0.5)
    assert d0.ecc == pytest.approx(d1.ecc, abs=0.05)


# --------------------------------------------------------------------------- #
# distance
# --------------------------------------------------------------------------- #

def test_angle_diff_folds_to_180_not_90():
    """10 and -80 are perpendicular and must read as the maximum; 10 and 190 are
    the same line and must read as zero. Folding to 90 instead would collapse
    the two most different views onto the same score."""
    assert _angle_diff_deg(10.0, -80.0) == pytest.approx(90.0)
    assert _angle_diff_deg(10.0, 190.0) == pytest.approx(0.0)
    assert _angle_diff_deg(10.0, 85.0) == pytest.approx(75.0)
    assert _angle_diff_deg(0.0, 0.0) == 0.0
    # Range is [0, 90]: never negative, never above a right angle.
    for a in (-90.0, -45.0, 0.0, 45.0, 90.0):
        for b in (-90.0, -45.0, 0.0, 45.0, 90.0):
            assert 0.0 <= _angle_diff_deg(a, b) <= 90.0


def test_identical_views_have_zero_distance():
    d = _desc(_needles())
    assert view_distance(d, d) == 0.0


def test_rotation_makes_views_farther_apart():
    """Monotone in the view change, which is the property the A/B is meant to
    exploit: a wider sweep must never read as more similar."""
    d0 = _desc(_needles(0.0))
    d5 = view_distance(d0, _desc(_needles(5.0)))
    d45 = view_distance(d0, _desc(_needles(45.0)))
    d90 = view_distance(d0, _desc(_needles(-90.0)))
    assert 0.0 < d5 < d45 < d90


def test_centroid_shift_is_measured_in_normalised_image_units():
    """A shift of 0.1 image widths must contribute 0.1 to the score, so an
    operator can set cov_centroid_shift by eyeballing the image."""
    d = view_distance(_desc(_needles(cx=0.0)), _desc(_needles(cx=0.1)))
    assert d == pytest.approx(0.1, abs=0.02)


def test_area_change_is_bounded_and_log_scaled():
    """Halving the silhouette must not dominate the other two terms, and
    doubling must give the same magnitude in the other direction."""
    small = _desc(_needles(half=0.10))
    big = _desc(_needles(half=0.20))
    assert view_distance(small, big) == pytest.approx(view_distance(big, small))
    assert view_distance(small, big) < 1.0


def test_distance_is_symmetric():
    a, b = _desc(_needles(10.0, cx=0.05)), _desc(_needles(-20.0, cx=-0.05))
    assert view_distance(a, b) == pytest.approx(view_distance(b, a))


def test_2d_flip_limit_is_real_and_documented():
    """Known limit, asserted so it cannot silently change: an in-plane 180-degree
    rotation of a symmetric silhouette is measured as the same view. This is
    inherent to 2D moments -- see the module docstring for why a 3D coverage
    metric is deferred until the pose is trustworthy."""
    d = _desc(_needles())
    flipped = np.fliplr(_needles())
    assert view_distance(d, _desc(flipped)) < view_distance(d, _desc(_needles(45.0)))


# --------------------------------------------------------------------------- #
# window decisions
# --------------------------------------------------------------------------- #

def test_first_view_always_earns_a_keyframe():
    """An empty window cannot be redundant, so the first frame is always a
    keyframe -- without this the window would stay empty forever."""
    cov = _cov()
    assert cov.update(_needles(), W, H, 0) is True
    assert len(cov) == 1


def test_empty_mask_is_not_a_keyframe_and_does_not_advance_the_interval():
    cov = _cov(min_interval=3)
    assert cov.update(np.zeros((H, W), dtype=np.uint8), W, H, 0) is False
    assert len(cov) == 0
    # The interval anchor must not have moved: the next view is still admitted.
    assert cov.update(_needles(), W, H, 1) is True


def test_near_identical_view_is_redundant():
    cov = _cov()
    assert cov.update(_needles(0.0), W, H, 0) is True
    for a in (3.0, 10.0, 12.0):
        assert cov.update(_needles(a), W, H, 1) is False, f"angle {a}"
    assert len(cov) == 1


def test_new_angle_is_admitted_even_when_the_centroid_is_the_same():
    """The orientation is the term that carries surface information, so a new
    angle must win over an unchanged silhouette position."""
    cov = _cov()
    assert cov.update(_needles(0.0), W, H, 0) is True
    assert cov.update(_needles(60.0), W, H, 1) is True
    assert len(cov) == 2


def test_centroid_shift_alone_is_redundant():
    """Same orientation, silhouette moved: nothing new about the object's
    surface. This keeps the window small while the camera jitters."""
    cov = _cov()
    assert cov.update(_needles(cx=0.0), W, H, 0) is True
    assert cov.update(_needles(cx=0.2), W, H, 1) is False
    assert len(cov) == 1


def test_min_interval_blocks_admission():
    """A moving object presents a new angle almost every frame; without a floor
    the window would churn and every keyframe would be a fresh single frame."""
    cov = _cov(min_interval=4)
    for i, a in [(0, 0.0), (1, 40.0), (2, 80.0), (3, 120.0)]:
        # Frames 1..3 are each less than min_interval=4 past the frame-0 anchor,
        # so the angle cannot save them.
        assert cov.update(_needles(a), W, H, i) is (i == 0), f"frame {i}"
    # Frame 6 is 6 frames past the anchor: the interval has elapsed and the new
    # angle goes in.
    assert cov.update(_needles(160.0), W, H, 6) is True


def test_window_is_capped_and_left_popped():
    cov = _cov(max_window=2)
    # Every gap here is above the 15-degree redundancy threshold, so all four
    # frames are admitted and the cap is what limits the window. None is at the
    # +-90 boundary, where the mod-180 fold flips sign on top of measurement
    # noise and the reading is unstable.
    for i, a in enumerate((0.0, 30.0, 60.0, 80.0)):
        assert cov.update(_needles(a), W, H, i * 10), f"window angle {a}"
    assert len(cov) == 2
    # The oldest entry must have been evicted, so the window holds the two
    # most recent admitted views.
    descs = cov.descriptors()
    assert descs[0].angle_deg == pytest.approx(60.0, abs=0.5)
    assert descs[1].angle_deg == pytest.approx(80.0, abs=0.5)


def test_reset_clears_window_and_interval_anchor():
    """GSMapping.run() re-initialises before each session; without clearing the
    interval anchor the first frame of session two is refused as too close to
    session one's last keyframe."""
    cov = _cov(min_interval=3)
    cov.update(_needles(0.0), W, H, 100)
    cov.reset()
    assert len(cov) == 0
    assert cov.update(_needles(45.0), W, H, 101) is True


# --------------------------------------------------------------------------- #
# neighbour selection
# --------------------------------------------------------------------------- #

def test_most_diverse_orders_by_distance_and_honours_limit():
    """The picked neighbours must be the two FURTHEST from the query, in
    decreasing order of distance -- not the first two in window order.

    The window is ascending in angle while the query sits at -60 degrees, so the
    distance ranking disagrees with the window order; sorting by index instead of
    by distance would hand the mapper its two NEAREST keyframes and silently undo
    the whole point of picking diverse neighbours. The ranking is asserted against
    a distance computed here independently, so this depends on no hand-derived
    numbers and catches either an index sort or a divergent internal distance.
    """
    cov = _cov(max_window=4)
    for i, a in enumerate((0.0, 20.0, 45.0, 70.0)):
        assert cov.update(_needles(a), W, H, i * 10), f"window angle {a}"
    assert len(cov) == 4, "every window angle must clear the redundancy test"
    query = _desc(_needles(-60.0))
    descs = cov.descriptors()
    expected = sorted(range(len(cov)),
                      key=lambda i: view_distance(descs[i], query), reverse=True)
    assert expected[:2] != [0, 1], "test is only meaningful if the distance " \
                                   "ranking disagrees with the window order"
    got = cov.most_diverse_indices(query, limit=2)
    assert [i for i, _ in got] == expected[:2]
    assert got[0][1] >= got[1][1]
    for i, dis in got:
        assert dis == pytest.approx(view_distance(descs[i], query), abs=1e-12)
    # limit must truncate, and an unlimited call must return the whole window.
    assert len(cov.most_diverse_indices(query)) == len(cov)
    assert len(cov.most_diverse_indices(query, limit=1)) == 1


def test_most_diverse_max_dis_bounds_the_pick():
    """The bound must exclude a far view while still returning the valid ones,
    and the survivors must come out farthest-first.

    The query at 170 degrees folds to -10, so its distances to the 0-, 20- and
    70-degree window entries are about 0.11, 0.33 and 0.89. A bound of 0.5
    therefore keeps exactly the first two, with the 20-degree view first since
    it is the more diverse of the survivors.
    """
    cov = _cov(max_window=3)
    for i, a in enumerate((0.0, 20.0, 70.0)):
        assert cov.update(_needles(a), W, H, i * 10), f"window angle {a}"
    query = _desc(_needles(170.0))
    picked = cov.most_diverse_indices(query, limit=2, max_dis=0.5)
    assert picked, "a valid neighbour must still be returned"
    assert all(dis <= 0.5 for _, dis in picked)
    assert [i for i, _ in picked] == [1, 0]


def test_most_diverse_returns_nothing_when_the_window_is_empty():
    assert _cov().most_diverse_indices(_desc(_needles()), limit=2) == []


def test_most_diverse_descriptor_wrapper_matches_indices():
    cov = _cov(max_window=3)
    for i, a in enumerate((0.0, 30.0, 70.0)):
        cov.update(_needles(a), W, H, i * 10)
    query = _desc(_needles(150.0))
    idx = cov.most_diverse_indices(query, limit=2)
    descs = cov.most_diverse(query, limit=2)
    assert len(descs) == len(idx)
    for (i, _), d in zip(idx, descs):
        assert d == cov.descriptors()[i]


# --------------------------------------------------------------------------- #
# descriptor contract
# --------------------------------------------------------------------------- #

def test_view_descriptor_is_a_plain_dataclass():
    """Everything above goes through ViewDescriptor, so its field order is a
    construction contract. Lock it: a reordered field would silently swap the
    angle and the area in every call below and still type-check."""
    fields = [f.name for f in dataclasses.fields(ViewDescriptor)]
    assert fields == ["centroid", "angle_deg", "area_frac", "ecc"]
    d = ViewDescriptor(np.array([0.1, 0.2]), 12.5, 0.03, 0.4)
    assert d.angle_deg == 12.5 and d.ecc == 0.4


# --------------------------------------------------------------------------- #
# the knobs must be reachable from every runner, not just from MappingConfig
# --------------------------------------------------------------------------- #

RUNNER_FILES = [
    "run_full_system_bundlegs_ow.py",
    "run_full_system_bundlegs_ow_3dgs.py",
    "run_full_system_bundlegs_ow_dense_densification.py",
]
COVERAGE_KNOBS = ("use_coverage_kf", "cov_angle_deg", "cov_centroid_shift",
                  "cov_shape_ratio", "cov_min_interval", "cov_neighbor_max_dis",
                  "use_coverage_prune_guard")


def _ast_class_field_defaults(path, class_name):
    """{field: default} for a dataclass, read with ast.

    The runners pull dataset loaders and rerun servers at import time, so parsing
    is the only way to ask what they declare without booting the pipeline.
    Defaults that are not literals are dropped -- the caller treats a missing
    key as unreadable, not as undeclared.
    """
    import ast
    with open(path) as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except (ValueError, TypeError):
                        pass
    return out


def _mapping_config_call_keywords(path):
    """{kwarg: value-AST} sent at every `MappingConfig(...)` call site.

    Forwarding a knob with a hardcoded value would satisfy a name-only check
    while leaving the operator's dial inert, so the value's form is asserted too
    in test below.
    """
    import ast
    with open(path) as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (getattr(func, "id", None) or getattr(func, "attr", None)) != "MappingConfig":
            continue
        for kw in node.keywords:
            if kw.arg:
                out[kw.arg] = kw.value
    return out


@pytest.mark.parametrize("runner", RUNNER_FILES)
def test_coverage_knobs_are_reachable_from_every_runner(runner):
    """Each coverage knob must reach MappingConfig AND default to the value
    MappingConfig already ships.

    The first catches the wiring bug that has happened twice in this repo: a knob
    exists on the config, the docs say to tune it, but no runner forwards it, so
    `use_coverage_kf=True` turns on fully hardcoded values. The second keeps the
    paired 5-clip A/B baselines reproducible: flipping use_coverage_kf alone must
    be bit-identical to before, which only holds if the runner defaults equal the
    mapper defaults.
    """
    import ast
    from obj_gs_mapping import MappingConfig

    mapper_defaults = {f.name: f.default
                       for f in dataclasses.fields(MappingConfig)
                       if f.default_factory is dataclasses.MISSING}
    path = os.path.join(os.path.dirname(__file__), os.pardir,
                        "full_system_ow_working", runner)

    # A second same-named class would silently satisfy this test, so name it.
    with open(path) as fh:
        tree = ast.parse(fh.read())
    gc = [n.name for n in ast.walk(tree)
          if isinstance(n, ast.ClassDef) and n.name == "GlobalConfig"]
    assert len(gc) == 1, (
        f"{runner}: expected exactly one GlobalConfig, found {gc} -- this test "
        f"cannot tell which one it read")

    declared = _ast_class_field_defaults(path, "GlobalConfig")
    forwarded = _mapping_config_call_keywords(path)

    for knob in COVERAGE_KNOBS:
        assert knob in mapper_defaults, (
            f"{runner}: {knob} is no longer on MappingConfig -- dead test entry?")
        if knob not in declared:
            pytest.fail(
                f"{runner}: GlobalConfig does not declare {knob} (or its default "
                f"is not a literal, which this test cannot read)")
        assert declared[knob] == mapper_defaults[knob], (
            f"{runner}: GlobalConfig.{knob}={declared[knob]!r} drifted from "
            f"MappingConfig.{knob}={mapper_defaults[knob]!r}")
        assert knob in forwarded, (
            f"{runner}: {knob} is declared in GlobalConfig but never forwarded "
            f"to MappingConfig")
        v = forwarded[knob]
        is_cfg_sourced = (isinstance(v, ast.Attribute) and v.attr == knob
                          and isinstance(v.value, ast.Name) and v.value.id == "cfg")
        assert is_cfg_sourced, (
            f"{runner}: {knob} reaches MappingConfig but not as cfg.{knob} -- a "
            f"hardcoded value at the call site leaves the CLI knob inert")


@pytest.mark.parametrize("runner", RUNNER_FILES)
def test_every_runner_defaults_coverage_off(runner):
    """The discipline that makes a paired A/B meaningful: every variant ships the
    new path off, so an unedited run reproduces the previous mapper exactly.
    Asserted on the runner, not on MappingConfig, because a runner could override
    the default to True and MappingConfig would still read False."""
    path = os.path.join(os.path.dirname(__file__), os.pardir,
                        "full_system_ow_working", runner)
    declared = _ast_class_field_defaults(path, "GlobalConfig")
    for knob in ("use_coverage_kf", "use_coverage_prune_guard"):
        assert declared.get(knob) is False, (
            f"{runner}: GlobalConfig.{knob}={declared.get(knob)!r} must default "
            f"to False, or the paired A/B baselines are not reproducible")
