"""Stage 4 — conditional diffusion generator.

Stable Diffusion inpainting UNet with a widened input
    x_in = [z_t || M_a↓ || E(I_a) || E(Î_c) || Ŝ↓]
and three conditioning branches:
  * Pose ControlNet (PA-VTON): skeleton heatmaps + DensePose -> residuals into the UNet skips.
  * Garment tokens (LaDI-VTON): CLIP image features -> pseudo-word tokens appended to the prompt.
  * Skin reference (DCTON): visible-skin tokens read by an extra, region-gated cross-attention
    that is active only where Ŝ predicts newly exposed skin.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from data.vitonhd import SEG_LARM, SEG_NECK, SEG_RARM


# ---------------------------------------------------------------------------
# Token mappers (garment / skin)
# ---------------------------------------------------------------------------
class TokenMapper(nn.Module):
    """Learnable queries attend to frozen CLIP patch features -> N tokens in the text space."""

    def __init__(self, clip_dim=1024, out_dim=768, n_tokens=16, heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_tokens, out_dim) * 0.02)
        self.proj_in = nn.Linear(clip_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim * 4), nn.GELU(), nn.Linear(out_dim * 4, out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, clip_feats):  # [B,257,clip_dim]
        kv = self.proj_in(clip_feats)
        q = self.queries.expand(kv.shape[0], -1, -1)
        x = q + self.attn(q, kv, kv)[0]
        return self.norm(x + self.ff(x))


# ---------------------------------------------------------------------------
# Region-gated skin cross-attention (added to every attn2 layer)
# ---------------------------------------------------------------------------
class SkinAttnProcessor(nn.Module):
    def __init__(self, inner_dim, out_dim, ctx_dim=768, heads=8, aspect=4 / 3):
        super().__init__()
        self.base = AttnProcessor2_0()
        self.to_k = nn.Linear(ctx_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(ctx_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)
        nn.init.zeros_(self.to_out.weight)  # starts as an exact no-op
        nn.init.zeros_(self.to_out.bias)
        self.heads, self.aspect = heads, aspect

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 skin_tokens=None, skin_mask=None, **kwargs):
        out = self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        if skin_tokens is None or skin_mask is None:
            return out
        b, L, _ = hidden_states.shape
        heads = attn.heads
        q = attn.to_q(hidden_states)
        k, v = self.to_k(skin_tokens.to(q.dtype)), self.to_v(skin_tokens.to(q.dtype))
        q, k, v = (t.view(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
        s = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, L, -1)
        s = self.to_out(s)
        h = int(round(math.sqrt(L * self.aspect)))
        w = L // h
        gate = F.interpolate(skin_mask.to(s.dtype), size=(h, w), mode="area").flatten(2).transpose(1, 2)
        return out + s * gate


class PlainAttnProcessor(AttnProcessor2_0):
    """Default attention that silently accepts (and ignores) the skin kwargs."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 skin_tokens=None, skin_mask=None, **kwargs):
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def install_skin_processors(unet, ctx_dim, aspect):
    procs, new = {}, nn.ModuleDict()
    for name in unet.attn_processors.keys():
        if ".attn2." in name:
            mod = unet.get_submodule(name[: -len(".processor")])
            p = SkinAttnProcessor(mod.to_q.out_features, mod.to_out[0].out_features, ctx_dim, mod.heads, aspect)
            procs[name] = p
            new[name.replace(".", "_")] = p
        else:
            procs[name] = PlainAttnProcessor()
    unet.set_attn_processor(procs)
    return new  # handle for saving / optimising


def make_controlnet(unet, cond_channels):
    """ControlNet initialised from the (widened, possibly trained) UNet, ignoring skin-attention weights."""
    cn = ControlNetModel.from_unet(unet, conditioning_channels=cond_channels, load_weights_from_unet=False)
    sd = {k: v for k, v in unet.state_dict().items() if ".processor." not in k}
    missing, unexpected = cn.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith(("controlnet_", "add_"))], "ControlNet init mismatch"
    cn.set_attn_processor(AttnProcessor2_0())
    return cn


def widen_conv_in(unet, extra):
    old = unet.conv_in
    new = nn.Conv2d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels] = old.weight
        new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=new.in_channels)


# ---------------------------------------------------------------------------
# Full generator
# ---------------------------------------------------------------------------
class TryOnGenerator(nn.Module):
    def __init__(self, unet: UNet2DConditionModel, n_seg=8, n_pose=10, clip_dim=1024,
                 n_garment_tokens=16, n_skin_tokens=4, aspect=4 / 3):
        super().__init__()
        widen_conv_in(unet, extra=4 + n_seg)  # + E(Î_c) + Ŝ↓  (SD-inpaint already has z_t, M_a, E(I_a))
        ctx = unet.config.cross_attention_dim
        self.unet = unet
        self.skin_procs = install_skin_processors(unet, ctx, aspect)
        self.n_control = n_pose + 3
        self.controlnet = make_controlnet(unet, self.n_control)
        self.garment_mapper = TokenMapper(clip_dim, ctx, n_garment_tokens)
        self.skin_mapper = TokenMapper(clip_dim, ctx, n_skin_tokens)

    @staticmethod
    def build_input(z_t, mask_lat, agn_lat, warp_lat, seg_lat):
        return torch.cat([z_t, mask_lat, agn_lat, warp_lat, seg_lat], 1)

    def forward(self, x_in, t, text_emb, garment_clip=None, skin_clip=None, skin_mask=None, control=None):
        """x_in: widened UNet input; text_emb: [B,77,768]; *_clip: CLIP patch features or None
        (None = condition dropped); control: [B, n_pose+3, H, W] or None."""
        ctx = text_emb
        if garment_clip is not None:
            ctx = torch.cat([text_emb, self.garment_mapper(garment_clip).to(text_emb.dtype)], 1)
        kw = {}
        if skin_clip is not None and skin_mask is not None:
            kw["cross_attention_kwargs"] = {"skin_tokens": self.skin_mapper(skin_clip), "skin_mask": skin_mask}
        res = {}
        if control is not None:
            down, mid = self.controlnet(x_in, t, encoder_hidden_states=ctx, controlnet_cond=control, return_dict=False)
            res = {"down_block_additional_residuals": down, "mid_block_additional_residual": mid}
        return self.unet(x_in, t, encoder_hidden_states=ctx, return_dict=False, **res, **kw)[0]

    def reinit_controlnet(self):
        """Call at the start of T4b so ControlNet copies the TRAINED UNet encoder."""
        dev, dt = next(self.unet.parameters()).device, next(self.unet.parameters()).dtype
        self.controlnet = make_controlnet(self.unet, self.n_control).to(dev, dt)

    # ------------- parameter groups for the two training steps -------------
    def params_t4a(self):  # base UNet (widened) + garment mapper
        return list(self.unet.parameters()) + list(self.garment_mapper.parameters())

    def params_t4b(self):  # ControlNet + skin attention + skin mapper (UNet frozen)
        return list(self.controlnet.parameters()) + list(self.skin_procs.parameters()) + list(self.skin_mapper.parameters())


def new_skin_mask(seg_pred, mask_a):
    """Region where skin must be generated: predicted arms/neck inside the agnostic mask."""
    lab = seg_pred.argmax(1, keepdim=True) if seg_pred.dim() == 4 and seg_pred.shape[1] > 1 else seg_pred
    skin = (lab == SEG_LARM) | (lab == SEG_RARM) | (lab == SEG_NECK)
    return skin.float() * mask_a
