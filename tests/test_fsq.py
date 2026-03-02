import numpy as np
import torch
from audiotools import AudioSignal

from dac.model import DAC
from dac.model import DACFile
from dac.nn.quantize import FiniteScalarQuantize


def test_fsq_quantization_shapes_and_bounds():
    fsq = FiniteScalarQuantize(
        input_dim=512,
        levels=[8] * 30,
    )

    z = torch.randn(4, 512, 100)
    z_q, codes = fsq(z)

    assert z_q.shape == z.shape
    assert codes.shape == (4, 30, 100)
    assert torch.is_floating_point(z_q)
    assert not torch.is_floating_point(codes)
    assert int(codes.min()) >= 0
    assert int(codes.max()) <= 7


def test_fsq_from_codes_shape_and_dtype():
    fsq = FiniteScalarQuantize(
        input_dim=512,
        levels=[8] * 30,
    )
    codes = torch.randint(0, 8, (2, 30, 20), dtype=torch.int64)
    z_q_reconstructed = fsq.from_codes(codes)
    assert z_q_reconstructed.shape == (2, 512, 20)
    assert torch.is_floating_point(z_q_reconstructed)


def test_fsq_ste_backward_flow():
    fsq = FiniteScalarQuantize(
        input_dim=512,
        levels=[8] * 30,
    )
    z = torch.randn(2, 512, 10, requires_grad=True)
    z_q, _ = fsq(z)
    loss = z_q.pow(2).mean()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_fsq_per_dimension_bounds_mixed_levels():
    levels = [16] * 15 + [8] * 10 + [4] * 5
    fsq = FiniteScalarQuantize(
        input_dim=512,
        levels=levels,
    )
    z = torch.randn(1, 512, 40)
    _, codes = fsq(z)
    for d, n_levels in enumerate(levels):
        assert int(codes[:, d, :].min()) >= 0
        assert int(codes[:, d, :].max()) <= (n_levels - 1)


def test_dac_forward_has_no_vq_aux_losses():
    model = DAC(sample_rate=44100, fsq_levels=[8] * 30)
    x = torch.randn(1, 1, 4096)
    out = model(x, sample_rate=44100, n_quantizers=3)
    assert "audio" in out
    assert "z" in out
    assert "codes" in out
    assert "vq/commitment_loss" not in out
    assert "vq/codebook_loss" not in out


def test_dac_encode_accepts_legacy_n_quantizers_without_shape_change():
    model = DAC(sample_rate=44100, fsq_levels=[8] * 30)
    x = torch.randn(1, 1, 4096)

    z0, c0 = model.encode(x)
    z1, c1 = model.encode(x, n_quantizers=1)

    assert z0.shape == z1.shape
    assert c0.shape == c1.shape
    assert c0.shape[1] == 30


def test_dacfile_saves_fsq_codes_as_uint8(tmp_path):
    codes = torch.randint(0, 8, (1, 30, 16), dtype=torch.int64)
    artifact = DACFile(
        codes=codes,
        chunk_length=16,
        original_length=16,
        input_db=torch.tensor(-16.0),
        channels=1,
        sample_rate=44100,
        padding=True,
        dac_version="1.0.0",
    )
    out_path = artifact.save(tmp_path / "sample")
    artifacts = np.load(out_path, allow_pickle=True)[()]
    assert artifacts["codes"].dtype == np.uint8

    loaded = DACFile.load(out_path)

    assert loaded.codes.shape == codes.shape
    assert int(loaded.codes.max()) <= 7


def test_fsq_compress_decompress_roundtrip_accepts_legacy_n_quantizers():
    model = DAC(sample_rate=44100, fsq_levels=[8] * 30)
    signal = AudioSignal(torch.randn(1, 1, 4096), 44100)

    artifact = model.compress(signal, win_duration=1.0, n_quantizers=2)
    assert artifact.codes.shape[1] == 30

    recons = model.decompress(artifact)
    assert recons.sample_rate == 44100
    assert recons.audio_data.shape[-1] == signal.audio_data.shape[-1]
