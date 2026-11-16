"""VITON-HD dataset loader producing every signal the pipeline needs (Stage 0).

Expected layout (VITON-HD + the extra folders listed in the README):

    root/{train,test}/
        image/              person images            000001_00.jpg
        cloth/              in-shop garments         000001_00.jpg
        cloth-mask/         garment masks            000001_00.jpg
        image-parse-v3/     human parsing (palette)  000001_00.png
        openpose_json/      OpenPose BODY_25/COCO     000001_00_keypoints.json
        image-densepose/    DensePose IUV render     000001_00.jpg
        cloth-kpts/         32 garment landmarks     000001_00.json   ([[x,y],...], original resolution)
        image-cloth-kpts/   32 landmarks of the WORN garment (S1 ground truth, train only)
        smpl-vis/           optional SMPL visibility  000001_00.npz   ('vis','depth', shape [10])
    root/train_pairs.txt, root/test_pairs.txt   ("person_img cloth_img" per line)
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Parsing: VITON-HD image-parse-v3 (LIP labels) -> 8 groups used by S2 / S4
# ---------------------------------------------------------------------------
# LIP: 0 bg,1 hat,2 hair,3 glove,4 sunglasses,5 upper,6 dress,7 coat,8 socks,9 pants,
#      10 torso-skin,11 scarf,12 skirt,13 face,14 l-arm,15 r-arm,16 l-leg,17 r-leg,18 l-shoe,19 r-shoe
PARSE_GROUPS = {
    0: [0],                 # background
    1: [1, 2],              # hair / hat
    2: [4, 13],             # face
    3: [5, 6, 7],           # upper garment (the region we replace)
    4: [14],                # left arm
    5: [15],                # right arm
    6: [9, 12, 16, 17, 18, 19, 8],  # lower body, legs, shoes
    7: [3, 10, 11],         # neck / torso skin / accessories
}
SEG_UPPER, SEG_LARM, SEG_RARM, SEG_NECK, SEG_FACE, SEG_LOWER = 3, 4, 5, 7, 2, 6

# OpenPose -> KGI's 10 upper-body joints:
# nose, neck, r-shoulder, r-elbow, r-wrist, l-shoulder, l-elbow, l-wrist, r-hip, l-hip
BODY25_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]
COCO18_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
# Expected parse group at each joint (used for parsing-derived visibility)
JOINT_PART = [SEG_FACE, SEG_NECK, None, SEG_RARM, SEG_RARM, None, SEG_LARM, SEG_LARM, None, None]
ARM_JOINTS = [3, 4, 6, 7]


def group_parse(parse: np.ndarray) -> np.ndarray:
    out = np.zeros_like(parse)
    for g, labels in PARSE_GROUPS.items():
        out[np.isin(parse, labels)] = g
    return out


def one_hot(seg: np.ndarray, n: int) -> torch.Tensor:
    t = torch.from_numpy(seg).long()
    return torch.nn.functional.one_hot(t, n).permute(2, 0, 1).float()


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW in [-1, 1]."""
    return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0


def render_heatmaps(kpts: np.ndarray, h: int, w: int, sigma: float) -> np.ndarray:
    """kpts: [N,2] (x,y) in pixels, negative = missing -> [N,H,W] gaussians."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    maps = np.zeros((len(kpts), h, w), np.float32)
    for i, (x, y) in enumerate(kpts):
        if x < 0 or y < 0:
            continue
        maps[i] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return maps


def occlusion_from_parse(seg: np.ndarray) -> np.ndarray:
    """O^gt: arm pixels lying inside the convex hull of the upper garment (arm in front of cloth)."""
    garment = (seg == SEG_UPPER).astype(np.uint8)
    arms = np.isin(seg, [SEG_LARM, SEG_RARM])
    occ = np.zeros(seg.shape, np.float32)
    cnts, _ = cv2.findContours(garment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return occ
    hull = cv2.convexHull(np.concatenate(cnts))
    hull_mask = np.zeros_like(garment)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    occ[(hull_mask > 0) & arms] = 1.0
    return occ


class VitonHD(Dataset):
    def __init__(self, cfg, split="train", paired=True):
        d = cfg["data"]
        self.root = os.path.join(d["root"], split)
        self.h, self.w = d["height"], d["width"]
        self.nseg = cfg["model"]["num_seg_classes"]
        self.nkg = d["num_garment_kpts"]
        self.sigma = d["kpt_sigma"] * self.h / 512
        self.smpl_dir = os.path.join(self.root, d.get("smpl_vis_dir", "smpl-vis"))
        self.split = split
        pairs_file = os.path.join(d["root"], f"{split}_pairs.txt")
        with open(pairs_file) as f:
            pairs = [line.split() for line in f if line.strip()]
        # paired: person wears the same garment (ground truth exists)
        self.pairs = [(p, p if paired else c) for p, c in pairs]

    def __len__(self):
        return len(self.pairs)

    # ---------------- loaders ----------------
    def _img(self, folder, name, interp=cv2.INTER_AREA):
        img = cv2.imread(os.path.join(self.root, folder, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.w, self.h), interpolation=interp)

    def _mask(self, folder, name):
        m = cv2.imread(os.path.join(self.root, folder, name), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return (m > 127).astype(np.float32)

    def _parse(self, name):
        from PIL import Image  # palette PNG keeps label indices
        p = np.array(Image.open(os.path.join(self.root, "image-parse-v3", name.replace(".jpg", ".png"))))
        p = cv2.resize(p, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return group_parse(p)

    def _pose(self, name):
        path = os.path.join(self.root, "openpose_json", name.replace(".jpg", "_keypoints.json"))
        with open(path) as f:
            people = json.load(f)["people"]
        kp = np.array(people[0]["pose_keypoints_2d"], np.float32).reshape(-1, 3)
        idx = BODY25_IDX if len(kp) == 25 else COCO18_IDX
        kp = kp[idx]
        src_h = 1024.0  # VITON-HD native resolution 1024x768
        kp[:, :2] *= self.h / src_h
        kp[kp[:, 2] < 0.05, :2] = -1
        return kp  # [10,3] x,y,conf

    def _garment_kpts(self, folder, name):
        path = os.path.join(self.root, folder, name.replace(".jpg", ".json"))
        if not os.path.exists(path):
            return None
        with open(path) as f:
            k = np.array(json.load(f), np.float32).reshape(-1, 2)[: self.nkg]
        return k * (self.h / 1024.0)

    def _visibility(self, name, pose, seg):
        """Per-joint visibility v and depth proxy d (in front of torso = 1).
        Uses SMPL-derived values if provided, otherwise derives them from parsing."""
        npz = os.path.join(self.smpl_dir, name.replace(".jpg", ".npz"))
        if os.path.exists(npz):
            z = np.load(npz)
            return z["vis"].astype(np.float32), z["depth"].astype(np.float32)
        vis = np.zeros(len(pose), np.float32)
        depth = np.zeros(len(pose), np.float32)
        garment = (seg == SEG_UPPER).astype(np.uint8)
        cnts, _ = cv2.findContours(garment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(garment)
        if cnts:
            cv2.fillConvexPoly(hull_mask, cv2.convexHull(np.concatenate(cnts)), 1)
        for j, (x, y, c) in enumerate(pose):
            if x < 0:
                continue
            xi, yi = int(np.clip(x, 0, self.w - 1)), int(np.clip(y, 0, self.h - 1))
            part = JOINT_PART[j]
            vis[j] = 1.0 if part is None else float(seg[yi, xi] == part)
            if j in ARM_JOINTS and vis[j] > 0 and hull_mask[yi, xi]:
                depth[j] = 1.0  # arm joint visible inside the garment hull -> in front of torso
        return vis, depth

    # ---------------- main ----------------
    def __getitem__(self, i):
        pname, cname = self.pairs[i]
        person = self._img("image", pname)
        cloth = self._img("cloth", cname)
        cmask = self._mask("cloth-mask", cname)
        seg = self._parse(pname)
        pose = self._pose(pname)
        dense = self._img("image-densepose", pname, cv2.INTER_NEAREST)
        vis, depth = self._visibility(pname, pose, seg)

        # Agnostic mask M_a: garment + arms + neck, slightly dilated (tight -> protects identity)
        region = np.isin(seg, [SEG_UPPER, SEG_LARM, SEG_RARM, SEG_NECK]).astype(np.uint8)
        k = max(3, self.h // 64) | 1
        mask_a = cv2.dilate(region, np.ones((k, k), np.uint8)).astype(np.float32)
        agnostic = person.astype(np.float32) * (1 - mask_a[..., None]) + 127.5 * mask_a[..., None]

        # Agnostic parsing S_a: masked region set to background
        seg_a = seg.copy()
        seg_a[mask_a > 0] = 0

        # Visible skin reference I_s: face + legs, everything else grey
        skin = np.isin(seg, [SEG_FACE, SEG_LOWER]) & ~np.isin(seg, [SEG_UPPER])
        skin_img = np.where(skin[..., None], person, 127).astype(np.uint8)
        skin_img = cv2.resize(skin_img, (224, 224), interpolation=cv2.INTER_AREA)

        worn = person.astype(np.float32) * (seg == SEG_UPPER)[..., None] + 255.0 * (seg != SEG_UPPER)[..., None]

        kc = self._garment_kpts("cloth-kpts", cname)
        kc_gt = self._garment_kpts("image-cloth-kpts", pname)
        if kc is None:
            kc = -np.ones((self.nkg, 2), np.float32)
        if kc_gt is None:
            kc_gt = -np.ones((self.nkg, 2), np.float32)

        pose_xy = pose[:, :2]
        out = {
            "name": pname, "cloth_name": cname,
            "person": to_tensor(person),
            "cloth": to_tensor(cloth),
            "cloth_mask": torch.from_numpy(cmask)[None],
            "worn_cloth": to_tensor(worn.astype(np.uint8)),
            "seg": torch.from_numpy(seg).long(),
            "seg_agnostic": one_hot(seg_a, self.nseg),
            "occ": torch.from_numpy(occlusion_from_parse(seg))[None],
            "densepose": to_tensor(dense),
            "pose": torch.from_numpy(pose),                 # [10,3]
            "pose_maps": torch.from_numpy(render_heatmaps(pose_xy, self.h, self.w, self.sigma)),
            "vis": torch.from_numpy(vis), "depth": torch.from_numpy(depth),
            "kc": torch.from_numpy(kc), "kc_gt": torch.from_numpy(kc_gt),
            "agnostic": to_tensor(agnostic.astype(np.uint8)),
            "mask_a": torch.from_numpy(mask_a)[None],
            "skin_ref": to_tensor(skin_img),
        }
        return out
