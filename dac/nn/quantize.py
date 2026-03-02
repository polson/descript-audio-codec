from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from dac.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class FiniteScalarQuantize(nn.Module):
    """
    Finite Scalar Quantization (FSQ) module.
    A simpler, codebook-free alternative to Vector Quantization.
    """

    def __init__(self, input_dim: int, levels: list):
        """
        Parameters
        ----------
        input_dim: int
            Dimension of input latents (e.g., 512)
        levels: list
            List of quantization levels per dimension.
            Total bits = sum(log2(levels[i]))
        """
        super().__init__()

        # Register levels as a buffer so it automatically migrates to the correct device
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int32))
        self.codebook_dim = len(levels)

        # Projection layers
        self.in_proj = WNConv1d(input_dim, self.codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(self.codebook_dim, input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor, n_quantizers: int = None):
        """
        Parameters
        ----------
        z: Tensor[B x input_dim x T]
            Input latents from encoder
        n_quantizers: int, optional
            Ignored in FSQ. Maintained purely for interface compatibility with RVQ.

        Returns
        -------
        z_out: Tensor[B x input_dim x T]
            Quantized latents
        codes: Tensor[B x D x T]
            Discrete codes (quantized values strictly in range [0, levels-1])
        latents: Tensor[B x D x T]
            The continuous projected latents (returned for interface compatibility)
        commitment_loss: Tensor[1]
            Always 0.0 (FSQ doesn't require this)
        codebook_loss: Tensor[1]
            Always 0.0 (FSQ doesn't require this)
        """
        # Project to FSQ dimensions
        z_proj = self.in_proj(z)  # [B, D, T]

        # Reshape levels for broadcasting: [1, D, 1]
        levels_tensor = self.levels.view(1, -1, 1)

        # 1. Bounding trick using tanh
        # We compute the bound threshold. E.g., for 8 levels, bound is 3.5.
        bound = (levels_tensor - 1) / 2.0

        # Bound continuous latents strictly to (-bound, bound)
        # We INTENTIONALLY keep the gradient of tanh here to prevent latents from blowing up.
        z_bounded = bound * torch.tanh(z_proj / bound)

        # 2. Shift to positive range [0, levels - 1]
        z_scaled = z_bounded + bound

        # 3. Quantize to integer codes strictly in [0, levels-1]
        codes = torch.round(z_scaled)

        # 4. Straight-Through Estimator (STE) for rounding
        # This allows gradients to flow through the non-differentiable round() operation
        z_q = z_scaled + (codes - z_scaled).detach()

        # 5. Scale codes back to [-1, 1] for symmetric out_proj weights
        z_q_norm = (z_q / bound) - 1.0

        # 6. Project back to input dimension
        z_out = self.out_proj(z_q_norm)

        # Construct dummy losses to maintain perfect interface compatibility with RVQ
        commitment_loss = torch.tensor(0.0, device=z.device)
        codebook_loss = torch.tensor(0.0, device=z.device)

        return z_out, codes, z_proj, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """
        Reconstruct quantized latents from codes.

        Parameters
        ----------
        codes: Tensor[B x D x T]
            Quantized codes in range [0, levels-1]

        Returns
        -------
        z_out: Tensor[B x input_dim x T]
            Quantized continuous representation
        z_p: None
            Maintained for interface compatibility with RVQ.
        codes: Tensor[B x D x T]
            The original codes.
        """
        levels_tensor = self.levels.view(1, -1, 1)
        bound = (levels_tensor - 1) / 2.0

        # Normalize integer codes from [0, levels-1] to [-1, 1]
        z_q_norm = (codes / bound) - 1.0

        # Project back to input dimension
        z_out = self.out_proj(z_q_norm)

        # Return a 3-tuple to ensure `z = self.quantizer.from_codes(c)[0]` works perfectly in base.py
        return z_out, None, codes


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)
