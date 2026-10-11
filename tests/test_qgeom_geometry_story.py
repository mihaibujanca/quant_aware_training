"""Verify the geometric claims made by the teaching pictures and animation."""
import torch

from aleph.qgeom.geometry_story import build_geometry_stories


def test_deformation_is_recoverable_without_collapsing_samples():
    story = build_geometry_stories()[0]
    torch.testing.assert_close(story.decomposition.quantized_weight,
                               torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64))
    # Neither quantization nor the FP map returns the input shape unchanged.
    assert not torch.allclose(story.states[2], story.states[0])
    assert not torch.allclose(story.states[2], story.states[1])
    torch.testing.assert_close(story.decomposition.R,
                               torch.tensor([[1.35, -.15], [-.35, .8]], dtype=torch.float64))
    torch.testing.assert_close(story.states[1][-2:],
                               torch.tensor([[-.915, .33], [.84, .135]], dtype=torch.float64))
    torch.testing.assert_close(story.states[2][-2:],
                               torch.tensor([[-.6, .3], [.7, .3]], dtype=torch.float64))
    assert story.decomposition.rank == 2
    torch.testing.assert_close(story.decomposition.U, torch.zeros(2, 2, dtype=torch.float64))
    torch.testing.assert_close(story.states[3], story.states[1])
    assert not torch.equal(story.states[2][-2], story.states[2][-1])


def test_relu_discards_distinction_preserved_by_linear_map():
    story = build_geometry_stories()[1]
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
