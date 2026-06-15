import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model import (
    Downsample,
    OverlapPatchEmbed,
    TransformerBlock,
    Upsample,
)


class PromptFuser(nn.Module):
    """Adaptive prompt fusion used by EcoDPL.

    Each prompt component owns a key, a value, and a learnable attention vector.
    The attention vector modulates the input query before cosine matching, which
    is the P-Fuser behavior described in the paper.
    """

    def __init__(self, num_prompts, query_dim, value_shape, temperature=1.0):
        super().__init__()
        self.num_prompts = num_prompts
        self.query_dim = query_dim
        self.temperature = temperature

        self.keys = nn.Parameter(torch.randn(num_prompts, query_dim) * 0.02)
        self.attention = nn.Parameter(torch.ones(num_prompts, query_dim))
        self.values = nn.Parameter(torch.randn(num_prompts, *value_shape) * 0.02)
        self.register_buffer("frequency", torch.ones(num_prompts), persistent=True)
        self.register_buffer("protected", torch.zeros(num_prompts, dtype=torch.bool), persistent=True)
        self.register_buffer("active", torch.ones(num_prompts, dtype=torch.bool), persistent=False)

    def forward(self, query, update_frequency=False):
        if query.dim() != 2:
            raise ValueError(f"PromptFuser expects [B, C] query, got {tuple(query.shape)}")

        query = F.normalize(query, dim=1)
        keys = F.normalize(self.keys, dim=1)
        attended = F.normalize(query[:, None, :] * self.attention[None, :, :], dim=2)
        logits = (attended * keys[None, :, :]).sum(dim=2) / self.temperature
        if self.active is not None:
            logits = logits.masked_fill(~self.active.to(logits.device)[None, :], -1e4)

        weights = F.softmax(logits, dim=1)
        fused = torch.einsum("bm,m...->b...", weights, self.values)

        # Distance surrogate from the paper, weighted by the fused prompt weights.
        cosine_distance = 1.0 - logits.clamp(-1.0, 1.0)
        distance_loss = (weights.detach() * cosine_distance).sum(dim=1).mean()

        top_indices = torch.argmax(weights.detach(), dim=1)
        if update_frequency and self.training:
            with torch.no_grad():
                counts = torch.bincount(top_indices, minlength=self.num_prompts)
                self.frequency.add_(counts.to(self.frequency.device))

        return fused, {
            "weights": weights,
            "top_indices": top_indices,
            "distance_loss": distance_loss,
        }

    @torch.no_grad()
    def set_active_range(self, start=None, end=None):
        self.active.zero_()
        start = 0 if start is None else max(0, int(start))
        end = self.num_prompts if end is None else min(self.num_prompts, int(end))
        if end <= start:
            raise ValueError(f"Invalid prompt range: {start}:{end}")
        self.active[start:end] = True

    @torch.no_grad()
    def clear_active_range(self):
        self.active.fill_(True)

    @torch.no_grad()
    def grad_tune(self, keep_components=25, mode="protect"):
        """Tune prompt bookkeeping without harming learned prompt values.

        The default release mode is deliberately non-destructive: it marks the
        most-used prompt components as protected, which gives the trainer a
        stable old-task prompt bank for continual learning. A legacy SVD mode is
        kept for ablation, but it is not the default because low-rank rewriting
        can reduce restoration quality.
        """

        keep_components = max(1, min(int(keep_components), self.num_prompts))
        if mode == "none":
            return
        if mode == "protect":
            top_indices = torch.topk(self.frequency, k=keep_components, largest=True).indices
            self.protected.zero_()
            self.protected[top_indices] = True
            return
        if mode != "svd":
            raise ValueError(f"Unknown Grad-Tuner mode: {mode}")

        flat = self.values.data.reshape(self.num_prompts, -1)
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean

        try:
            _, _, v = torch.pca_lowrank(centered, q=keep_components, center=False)
            compact = centered @ v @ v.t() + mean
        except RuntimeError:
            compact = flat

        self.values.data.copy_(compact.reshape_as(self.values.data))

    def zero_protected_grads(self):
        if not bool(self.protected.any()):
            return
        mask = self.protected.to(self.keys.device)
        for tensor in (self.keys, self.attention, self.values):
            if tensor.grad is not None:
                tensor.grad[mask] = 0


class EcoDPLPromptIR(nn.Module):
    """EcoDPL with image-level and feature-level adaptive prompt pools.

    The backbone follows the PromptIR/Restormer-style implementation already in
    this repository, while the continual-learning interface mirrors the TIP
    paper: image prompts, feature prompts, P-Fuser, frequency tables,
    Grad-Tuner, and optional parameter regularization from the trainer.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=None,
        num_refinement_blocks=4,
        heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
        num_prompts=100,
        image_prompt_size=32,
        grad_tuner_components=25,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.num_prompts = num_prompts
        self.grad_tuner_components = grad_tuner_components

        self.patch_embed_query = OverlapPatchEmbed(inp_channels, dim)
        self.image_fuser = PromptFuser(
            num_prompts=num_prompts,
            query_dim=dim,
            value_shape=(inp_channels, image_prompt_size, image_prompt_size),
        )
        self.image_prompt_adapter = nn.Conv2d(inp_channels * 2, inp_channels, 1, bias=bias)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 4))
        latent_dim = int(dim * 8)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=latent_dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[3])
        ])

        self.feature_fuser = PromptFuser(
            num_prompts=num_prompts,
            query_dim=latent_dim,
            value_shape=(latent_dim, 1, 1),
        )
        self.feature_prompt_adapter = nn.Conv2d(latent_dim * 2, latent_dim, 1, bias=bias)

        self.up4_3 = Upsample(latent_dim)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, stride=1, padding=1, bias=bias)
        self.last_aux = {}

    @staticmethod
    def _match_skip(x, skip):
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inp_img, return_aux=False):
        b, _, h, w = inp_img.shape

        query_feature = self.patch_embed_query(inp_img).mean(dim=(-2, -1))
        image_prompt, image_aux = self.image_fuser(query_feature, update_frequency=True)
        image_prompt = F.interpolate(image_prompt, size=(h, w), mode="bilinear", align_corners=False)
        prompted_img = self.image_prompt_adapter(torch.cat([inp_img, image_prompt], dim=1))

        out_enc_level1 = self.encoder_level1(self.patch_embed(prompted_img))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))

        feature_query = latent.mean(dim=(-2, -1))
        feature_prompt, feature_aux = self.feature_fuser(feature_query, update_frequency=True)
        feature_prompt = feature_prompt.expand(b, -1, latent.shape[-2], latent.shape[-1])
        latent = self.feature_prompt_adapter(torch.cat([latent, feature_prompt], dim=1))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self._match_skip(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(torch.cat([inp_dec_level3, out_enc_level3], dim=1))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self._match_skip(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(torch.cat([inp_dec_level2, out_enc_level2], dim=1))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self._match_skip(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(torch.cat([inp_dec_level1, out_enc_level1], dim=1))
        out_dec_level1 = self.refinement(out_dec_level1)

        restored = self.output(out_dec_level1) + inp_img
        self.last_aux = {
            "image_distance": image_aux["distance_loss"],
            "feature_distance": feature_aux["distance_loss"],
            "image_top_indices": image_aux["top_indices"],
            "feature_top_indices": feature_aux["top_indices"],
        }

        if return_aux:
            return restored, self.last_aux
        return restored

    def prompt_regularization_loss(self):
        image_keys = F.normalize(self.image_fuser.keys, dim=1)
        feature_keys = F.normalize(self.feature_fuser.keys, dim=1)
        image_eye = torch.eye(self.num_prompts, device=image_keys.device)
        feature_eye = torch.eye(self.num_prompts, device=feature_keys.device)
        return (
            (image_keys @ image_keys.t() - image_eye).pow(2).mean()
            + (feature_keys @ feature_keys.t() - feature_eye).pow(2).mean()
        )

    @torch.no_grad()
    def grad_tune_prompts(self, mode="protect"):
        self.image_fuser.grad_tune(self.grad_tuner_components, mode=mode)
        self.feature_fuser.grad_tune(self.grad_tuner_components, mode=mode)

    @torch.no_grad()
    def set_active_prompt_range(self, start=None, end=None):
        self.image_fuser.set_active_range(start, end)
        self.feature_fuser.set_active_range(start, end)

    @torch.no_grad()
    def clear_active_prompt_range(self):
        self.image_fuser.clear_active_range()
        self.feature_fuser.clear_active_range()

    def zero_protected_prompt_grads(self):
        self.image_fuser.zero_protected_grads()
        self.feature_fuser.zero_protected_grads()
