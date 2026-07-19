"""Run deterministic Assignment 3 checks without downloading course datasets."""

from pathlib import Path

import numpy as np
import torch

from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.classifiers.transformer import CaptioningTransformer
from cs231n.gan_pytorch import (
    build_dc_classifier,
    build_dc_generator,
    count_params,
    discriminator,
    discriminator_loss,
    generator,
    generator_loss,
    ls_discriminator_loss,
    ls_generator_loss,
    rel_error,
)
from cs231n.simclr.contrastive_loss import (
    compute_sim_matrix,
    sim,
    sim_positive_pairs,
    simclr_loss_naive,
    simclr_loss_vectorized,
)


ROOT = Path(__file__).resolve().parent


def check_captioning_losses():
    vocab = {"<NULL>": 0, "cat": 2, "dog": 3}
    settings = (
        ("rnn", (-1.5, 0.3), 9.83235591003),
        ("lstm", (-0.5, 1.7), 9.82445935443),
    )
    for cell_type, feature_range, expected in settings:
        N, D, W, H, T = 10, 20, 30, 40, 13
        model = CaptioningRNN(vocab, D, W, H, cell_type, np.float64)
        for key, value in model.params.items():
            model.params[key] = np.linspace(-1.4, 1.3, value.size).reshape(value.shape)
        features = np.linspace(*feature_range, N * D).reshape(N, D)
        captions = (np.arange(N * T) % len(vocab)).reshape(N, T)
        loss, _ = model.loss(features, captions)
        np.testing.assert_allclose(loss, expected, rtol=0, atol=1e-10)


def check_gans():
    answers = dict(np.load(ROOT / "gan-checks.npz"))
    assert count_params(discriminator()) == 267009
    assert count_params(generator(4)) == 1858320
    assert count_params(build_dc_classifier(8)) == 1102721
    assert count_params(build_dc_generator(4)) == 6580801

    real = torch.tensor(answers["logits_real"])
    fake = torch.tensor(answers["logits_fake"])
    checks = (
        (discriminator_loss(real, fake), answers["d_loss_true"]),
        (generator_loss(fake), answers["g_loss_true"]),
        (ls_discriminator_loss(real, fake), answers["d_loss_lsgan_true"]),
        (ls_generator_loss(fake), answers["g_loss_lsgan_true"]),
    )
    for actual, expected in checks:
        assert rel_error(expected, actual.detach().cpu().numpy()) < 1e-7


def check_simclr():
    answers = torch.load(ROOT / "simclr_sanity_check.key", map_location="cpu")
    left, right = answers["left"], answers["right"]
    torch.testing.assert_close(sim(left[0], right[0]), answers["sim"][0].squeeze())
    torch.testing.assert_close(sim_positive_pairs(left, right), answers["sim"])
    out = torch.cat((left, right), dim=0)
    torch.testing.assert_close(compute_sim_matrix(out), answers["sim_matrix"])
    for tau in (1.0, 5.0):
        expected = answers["loss"][str(tau)]
        np.testing.assert_allclose(
            simclr_loss_naive(left, right, tau).item(), expected, rtol=2e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            simclr_loss_vectorized(left, right, tau, device="cpu").item(),
            expected,
            rtol=2e-7,
            atol=1e-7,
        )


def check_transformer_causality():
    torch.manual_seed(231)
    vocab = {"<NULL>": 0, "<START>": 1, "a": 2, "b": 3, "c": 4}
    model = CaptioningTransformer(vocab, 6, 8, num_heads=2, num_layers=2)
    model.eval()
    features = torch.randn(2, 6)
    captions_a = torch.tensor([[1, 2, 3, 4], [1, 3, 2, 4]])
    captions_b = captions_a.clone()
    captions_b[:, 3] = torch.tensor([2, 3])
    scores_a = model(features, captions_a)
    scores_b = model(features, captions_b)
    assert scores_a.shape == (2, 4, len(vocab))
    torch.testing.assert_close(scores_a[:, :3], scores_b[:, :3])


if __name__ == "__main__":
    check_captioning_losses()
    check_gans()
    check_simclr()
    check_transformer_causality()
    print("All deterministic Assignment 3 checks passed.")
