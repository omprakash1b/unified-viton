"""Smoke test: every module runs forward + backward on random tensors (no downloads, CPU ok).

    python -m tests.test_shapes
Uses a tiny UNet/VAE config so it runs in seconds; the real models only differ in size.
"""
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel

from models.emasc import SkipDecoder, composite
from models.generator import TryOnGenerator, new_skin_mask
from models.keypoint_gcn import KeypointGCN, keypoint_loss
from models.seg_net import SegNet, seg_loss
from models.warp import GarmentWarper, second_order_loss, tv_loss
from pipeline import render_kpt_maps

B, H, W, NG, NP, NS = 2, 128, 96, 32, 10, 8
torch.manual_seed(0)


def test_kp():
    m = KeypointGCN(NG, NP)
    kc = torch.rand(B, NG, 2) * 1.6 - 0.8
    pose = torch.cat([torch.rand(B, NP, 2) * 2 - 1, torch.ones(B, NP, 1)], -1)
    out = m(kc, pose, torch.ones(B, NP), torch.zeros(B, NP))
    keypoint_loss(out, kc + 0.05).backward()
    assert out.shape == (B, NG, 2)
    return out.detach()


def test_seg(kc_hat):
    m = SegNet(NS, NG, NP, base=16)
    px = (kc_hat + 1) / 2 * torch.tensor([W - 1, H - 1])
    maps = render_kpt_maps(px, H, W, 3.0)
    s, o = m(torch.rand(B, NS, H, W), maps, torch.rand(B, NP, H, W), torch.rand(B, 3, H, W), torch.rand(B, 1, H, W))
    loss, _ = seg_loss(s, o, torch.randint(0, NS, (B, H, W)), (torch.rand(B, 1, H, W) > 0.9).float(), NS)
    loss.backward()
    assert s.shape == (B, NS, H, W) and o.shape == (B, 1, H, W)
    return s.detach().softmax(1), torch.sigmoid(o.detach())


def test_warp(kc_hat, seg_soft, occ):
    m = GarmentWarper(NP)
    kc = kc_hat + 0.03 * torch.randn_like(kc_hat)
    cloth = torch.rand(B, 3, H, W) * 2 - 1
    out = m(cloth, torch.ones(B, 1, H, W), kc, kc_hat, seg_soft[:, 3:4], occ, torch.rand(B, 3, H, W), torch.rand(B, NP, H, W))
    loss = out["warped"].abs().mean() + tv_loss(out["coarse_flow"]) + second_order_loss(out["flow"])
    loss.backward()
    assert out["warped"].shape == (B, 3, H, W) and torch.isfinite(out["warped"]).all()
    return out


def tiny_unet():
    return UNet2DConditionModel(
        sample_size=(H // 8, W // 8), in_channels=9, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=64, attention_head_dim=8,
        norm_num_groups=8)


def test_generator(seg_soft, warped_mask):
    gen = TryOnGenerator(tiny_unet(), NS, NP, clip_dim=48, n_garment_tokens=4, n_skin_tokens=2, aspect=H / W)
    h, w = H // 8, W // 8
    x_in = gen.build_input(torch.randn(B, 4, h, w), torch.rand(B, 1, h, w), torch.randn(B, 4, h, w),
                           torch.randn(B, 4, h, w), torch.rand(B, NS, h, w))
    assert x_in.shape[1] == gen.unet.conv_in.in_channels == 9 + 4 + NS
    t = torch.randint(0, 1000, (B,))
    text = torch.randn(B, 77, 64)
    skin_mask = new_skin_mask(seg_soft, torch.ones(B, 1, H, W))
    # T4a path
    eps = gen(x_in, t, text, garment_clip=torch.randn(B, 17, 48))
    assert eps.shape == (B, 4, h, w)
    # T4b path: all branches; skin attention output projection is zero-initialised -> must be a no-op at init
    eps_full = gen(x_in, t, text, torch.randn(B, 17, 48), torch.randn(B, 17, 48), skin_mask, torch.rand(B, NP + 3, H, W))
    eps_full.mean().backward()
    assert any(p.grad is not None for p in gen.params_t4b())
    gen.reinit_controlnet()
    return gen


def test_decoder(warp_out):
    vae = AutoencoderKL(block_out_channels=(16, 32), down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2, latent_channels=4, norm_num_groups=8)
    dec = SkipDecoder(vae)
    z = torch.randn(B, 4, H // 2, W // 2)  # this tiny VAE downsamples 2x
    person = torch.rand(B, 3, H, W) * 2 - 1
    mask = torch.zeros(B, 1, H, W)
    mask[..., H // 4: 3 * H // 4, :] = 1
    out = dec(z, person, warp_out["warped"].detach(), mask, warp_out["warped_mask"].detach())
    img = composite(out, person, mask)
    img.mean().backward()
    assert img.shape == (B, 3, H, W)
    assert torch.allclose(img * (1 - mask), person * (1 - mask))  # outside the mask is untouched
    assert all(p.grad is not None for p in dec.parameters_trainable())


if __name__ == "__main__":
    kc_hat = test_kp(); print("S1 keypoint GCN      ok")
    seg_soft, occ = test_seg(kc_hat); print("S2 seg + occlusion   ok")
    w = test_warp(kc_hat, seg_soft, occ); print("S3 two-step warp     ok")
    test_generator(seg_soft, w["warped_mask"]); print("S4 generator         ok")
    test_decoder(w); print("S5 skip decoder      ok")
    print("all stages passed")
