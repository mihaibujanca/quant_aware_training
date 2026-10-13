"""Verify the geometric claims made by the teaching pictures and animation."""
import torch

from aleph.qgeom.geometry_story import (
    build_geometry_stories, build_scalar_measurement_story, build_rectangular_measurement_story,
    build_typical_ledger_story,
)


def test_scalar_quantization_changes_direction_without_changing_rank():
    story = build_scalar_measurement_story()
    torch.testing.assert_close(story.wq, torch.tensor([1., 1.], dtype=torch.float64))
    torch.testing.assert_close(story.r, torch.tensor(.775, dtype=torch.float64))
    torch.testing.assert_close(story.u, torch.tensor([.225, -.225], dtype=torch.float64))
    torch.testing.assert_close(story.wq @ story.u, torch.tensor(0., dtype=torch.float64))
    assert torch.linalg.matrix_rank(story.w[:, None]) == torch.linalg.matrix_rank(story.wq[:, None]) == 1
    assert not torch.isclose(torch.linalg.vector_norm(story.w), torch.linalg.vector_norm(story.wq))
    assert not torch.isclose(story.w[0]*story.wq[1] - story.w[1]*story.wq[0], torch.tensor(0., dtype=torch.float64))
    # A different scalar gain adds error along wq; it cannot remove u.
    error = story.w - (story.r + .3)*story.wq
    torch.testing.assert_close(error.square().sum(), story.u.square().sum() + .3**2*story.wq.square().sum())


def test_same_scalar_fiber_has_distinct_fp_responses_and_opposite_missing_terms():
    story = build_scalar_measurement_story()
    torch.testing.assert_close(story.points @ story.wq, torch.tensor([.6, .6], dtype=torch.float64))
    torch.testing.assert_close(story.points @ story.w, torch.tensor([.69, .24], dtype=torch.float64))
    torch.testing.assert_close(story.r*(story.points @ story.wq), torch.tensor([.465, .465], dtype=torch.float64))
    torch.testing.assert_close(story.points @ story.u, torch.tensor([.225, -.225], dtype=torch.float64))
    torch.testing.assert_close((story.points[0]-story.points[1]) @ story.wq, torch.tensor(0., dtype=torch.float64))
    torch.testing.assert_close((story.points[0]-story.points[1]) @ story.w, torch.tensor(.45, dtype=torch.float64))


def test_deformation_is_recoverable_without_collapsing_samples():
    story = build_geometry_stories()[0]
    torch.testing.assert_close(story.decomposition.quantized_weight,
                               torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64))
    # Neither quantization nor the FP map returns the input shape unchanged.
    assert not torch.allclose(story.states[2], story.states[0])
    assert not torch.allclose(story.states[2], story.states[1])
    torch.testing.assert_close(story.decomposition.R,
                               torch.tensor([[1., -.4], [-.35, 1.25]], dtype=torch.float64))
    torch.testing.assert_close(story.states[1][-2:],
                               torch.tensor([[-.865, .235], [-.73, .07]], dtype=torch.float64))
    torch.testing.assert_close(story.states[2][-2:],
                               torch.tensor([[-.9, -.1], [-.8, -.2]], dtype=torch.float64))
    assert story.decomposition.rank == 2
    torch.testing.assert_close(story.decomposition.U, torch.zeros(2, 2, dtype=torch.float64))
    torch.testing.assert_close(story.decomposition.R,
                               torch.linalg.solve(story.decomposition.quantized_weight, story.decomposition.weight))
    torch.testing.assert_close(story.states[3], story.states[1])
    assert not torch.equal(story.states[2][-2], story.states[2][-1])


def test_rectangular_full_rank_measurements_change_fibers_without_rank_drop():
    story = build_rectangular_measurement_story()
    W, Wq = story.decomposition.weight, story.decomposition.quantized_weight
    assert W.shape == Wq.shape == (3, 2)
    assert torch.linalg.matrix_rank(W) == torch.linalg.matrix_rank(Wq) == story.decomposition.rank == 2
    torch.testing.assert_close(torch.linalg.svdvals(Wq),
                               torch.tensor([3**.5, 1.], dtype=W.dtype))
    torch.testing.assert_close(story.fiber_direction @ Wq, torch.zeros(2, dtype=W.dtype))
    torch.testing.assert_close(story.fiber_direction @ W, torch.tensor([.5, .05], dtype=W.dtype))
    torch.testing.assert_close(story.points, torch.tensor([[.8, -.2, -.1], [.2, .4, -.7]], dtype=W.dtype))
    torch.testing.assert_close(story.points @ Wq, torch.tensor([[.6, -.3], [.6, -.3]], dtype=W.dtype))
    torch.testing.assert_close(story.points @ W, torch.tensor([[.625, -.08], [.325, -.11]], dtype=W.dtype))
    torch.testing.assert_close(story.points @ Wq @ story.decomposition.R,
                               torch.tensor([[.475, -.095], [.475, -.095]], dtype=W.dtype))
    torch.testing.assert_close(story.points @ story.decomposition.U,
                               torch.tensor([[.15, .015], [-.15, -.015]], dtype=W.dtype))


def test_typical_map_shows_every_ledger_term_and_cancellation():
    story = build_typical_ledger_story()
    L = story.ledger
    c_R = L.transported_post - L.quant_post
    c_NL = L.recoverable_post - L.transported_post
    c_U = L.fp_post - L.recoverable_post
    torch.testing.assert_close(c_R, torch.tensor([[-.1, .14]]*2, dtype=c_R.dtype))
    torch.testing.assert_close(c_NL, torch.tensor([[-.025, -.14]]*2, dtype=c_R.dtype))
    torch.testing.assert_close(c_U, torch.tensor([[.15, 0.], [-.15, 0.]], dtype=c_R.dtype))
    # C nearly cancels; D reinforces.
    torch.testing.assert_close(c_R+c_NL+c_U, torch.tensor([[.025, 0.], [-.275, 0.]], dtype=c_R.dtype))
    # Both sheets share the Q route; only the FP target moves along the fiber.
    C, D = story.sheets
    torch.testing.assert_close(C.quant_post, D.quant_post)
    torch.testing.assert_close(C.recoverable_post, D.recoverable_post)
    assert (C.fp_post - D.fp_post).abs().max() > .1


def test_relu_discards_distinction_preserved_by_linear_map():
    deformation, story = build_geometry_stories()[:2]
    # Figure 2 continues Figure 1's map and point identities, then adds ReLU.
    torch.testing.assert_close(story.points, deformation.points)
    torch.testing.assert_close(story.decomposition.weight, deformation.decomposition.weight)
    torch.testing.assert_close(story.states[0], deformation.states[2])
    torch.testing.assert_close(story.states[1], deformation.states[3])
    torch.testing.assert_close(story.decomposition.quantized_weight,
                               torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64))
    torch.testing.assert_close(story.decomposition.R,
                               torch.tensor([[1., -.4], [-.35, 1.25]], dtype=torch.float64))
    torch.testing.assert_close(story.decomposition.U, torch.zeros(2, 2, dtype=torch.float64))
    # The quantized map moves both probes before ReLU erases their distinction.
    torch.testing.assert_close(story.states[0][-2:],
                               torch.tensor([[-.9, -.1], [-.8, -.2]], dtype=torch.float64))
    assert not torch.allclose(story.states[0], story.points)
    assert not torch.equal(story.states[0][-2], story.states[0][-1])
    # After Q's ReLU, both are exactly zero; their FP outputs are different.
    torch.testing.assert_close(story.states[3][-2:], torch.zeros(2, 2, dtype=torch.float64))
    torch.testing.assert_close(story.states[2][-2:], torch.tensor([[0., .235], [0., .07]], dtype=torch.float64))
    torch.testing.assert_close(story.states[4][-2:], torch.zeros(2, 2, dtype=torch.float64))
    torch.testing.assert_close(story.states[4]-story.states[2], story.ledger.nonlinear)


def test_line_correction_cannot_restore_missing_width_but_input_residual_can():
    story = build_geometry_stories()[2]
    assert story.decomposition.rank == 1
    for index in [2, 3]:
        torch.testing.assert_close(story.states[index][-2], story.states[index][-1])
        assert int(torch.linalg.matrix_rank(story.states[index])) == 1
    torch.testing.assert_close(story.states[1][-2:], torch.tensor([[.2, -.08], [.6, .32]], dtype=torch.float64))
    assert int(torch.linalg.matrix_rank(story.states[1])) == 2
    # The shared repaired point is the least-squares midpoint of the two targets.
    torch.testing.assert_close(story.states[3][-2:],
                               torch.tensor([[.4, .12], [.4, .12]], dtype=torch.float64))
    torch.testing.assert_close(story.states[3][-2], story.states[1][-2:].mean(dim=0))
    torch.testing.assert_close((story.points @ story.decomposition.U)[-2:],
                               torch.tensor([[-.2, -.2], [.2, .2]], dtype=torch.float64))
    # A further downstream linear motion can only increase squared map error.
    perturbation = torch.tensor([[.3, -.2], [.4, .5]], dtype=torch.float64)
    Wq, U = story.decomposition.quantized_weight, story.decomposition.U
    other_error = story.decomposition.weight - Wq @ (story.decomposition.R + perturbation)
    torch.testing.assert_close(other_error.square().sum(),
                               U.square().sum() + (Wq @ perturbation).square().sum())
    torch.testing.assert_close(story.states[4], story.states[1])
    torch.testing.assert_close(story.states[5], torch.relu(story.states[1]))


def test_payload_retains_point_identity_and_path_structure():
    for story in build_geometry_stories():
        assert not torch.allclose(story.decomposition.quantized_weight, torch.eye(2, dtype=torch.float64))
        assert sum(story.path_lengths) + 2 == len(story.points)
        for state in story.states:
            assert state.shape == story.points.shape
        payload = story.payload()
        torch.testing.assert_close(torch.tensor(payload['inputs'], dtype=story.points.dtype),
                                   story.points, atol=1e-6, rtol=0)
        assert len(payload['states']) == len(story.labels) == len(story.reference_indices)
        assert all(reference is None or 0 <= reference < len(story.states) for reference in story.reference_indices)
