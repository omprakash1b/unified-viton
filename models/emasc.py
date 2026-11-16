"""Stage 5 — mask-aware skip-connection decoder (LaDI-VTON EMASC, extended).

VAE encoder features of the agnostic person I_a (outside the mask) AND of the warped garment Î_c
(inside the garment region) are injected into the frozen VAE decoder through zero-initialised
adapters. The final image is a hard composite:
    I_out = M_a ⊙ D_skip(ẑ_0, {f_i(I_a)}, {f_i(Î_c)}) + (1 − M_a) ⊙ I_p
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _adapter(cin, cout):
    a = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
    nn.init.zeros_(a[-1].weight)
    nn.init.zeros_(a[-1].bias)
    return a


class SkipDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae.requires_grad_(False)
        self.sf = vae.config.scaling_factor
        with torch.no_grad():  # discover encoder/decoder feature shapes with a tiny dry run
            p = next(vae.parameters())
            x = torch.zeros(1, 3, 64, 48, device=p.device, dtype=p.dtype)
            #enc = self._enc_feats(x)
            enc = self._enc_feats(x)
            z = vae.encode(x).latent_dist.mode()  # real latent size (8x for SD's VAE, any factor works)
            dec = self._dec_feats(z)
            #dec = self._dec_feats(torch.zeros(1, vae.config.latent_channels, 8, 6, device=p.device, dtype=p.dtype))
        self.match = []  # for each decoder up-block: index of the encoder feature with the same size
        a_list, c_list = [], []
        for d in dec:
            j = next(i for i, e in enumerate(enc) if e.shape[-2:] == d.shape[-2:])
            self.match.append(j)
            a_list.append(_adapter(enc[j].shape[1], d.shape[1]))
            c_list.append(_adapter(enc[j].shape[1], d.shape[1]))
        self.adapt_person = nn.ModuleList(a_list)
        self.adapt_garment = nn.ModuleList(c_list)

    # ---- VAE internals (diffusers Encoder / Decoder) ----
    def _enc_feats(self, x):
        e = self.vae.encoder
        h = e.conv_in(x)
        feats = [h]
        for blk in e.down_blocks:
            h = blk(h)
            feats.append(h)
        return feats

    def _dec_feats(self, z):
        d = self.vae.decoder
        h = d.mid_block(d.conv_in(self.vae.post_quant_conv(z)), None)
        out = []
        for blk in d.up_blocks:
            h = blk(h, None)
            out.append(h)
        return out

    def forward(self, z0, agnostic, warped, mask_a, garment_mask):
        with torch.no_grad():
            fa = self._enc_feats(agnostic)
            fc = self._enc_feats(warped)
        d = self.vae.decoder
        h = d.mid_block(d.conv_in(self.vae.post_quant_conv(z0 / self.sf)), None)
        for i, blk in enumerate(d.up_blocks):
            h = blk(h, None)
            j = self.match[i]
            m = F.interpolate(mask_a, size=h.shape[-2:], mode="nearest").to(h.dtype)
            g = F.interpolate(garment_mask, size=h.shape[-2:], mode="nearest").to(h.dtype)
            h = h + self.adapt_person[i](fa[j]) * (1 - m) + self.adapt_garment[i](fc[j]) * g * m
        h = d.conv_out(d.conv_act(d.conv_norm_out(h)))
        return h.clamp(-1, 1)

    def parameters_trainable(self):
        return list(self.adapt_person.parameters()) + list(self.adapt_garment.parameters())


def composite(decoded, person, mask_a):
    return mask_a * decoded + (1 - mask_a) * person
