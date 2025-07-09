"""HybridFilterNet
===============
A lightweight frame‑importance estimator that fuses TSM‑ResNet (2D backbone
with Temporal Shift) and two R(2+1)D blocks, outputting a (B, T) importance
vector you can plug into any Top‑K sampler + downstream 3 D classifier.

The file provides:
  • TemporalShift     – zero‑parameter time‑shift op (taken from TSM)
  • make_tsm_block    – utility to wrap every 3×3 conv in a block with TSM
  • R2Plus1DBlock     – (2+1) D decomposition of a 3 D conv
  • HybridFilterNet   – main network, returns frame scores in [0,1]
  • select_topk_frames – helper that gathers the Top‑k frames per clip
  • two toy training loops showing (1) joint training, (2) freeze‑then‑finetune

Adjust channel widths / clip length / learning rates to suit your dataset
and GPU memory budget.  The code is intentionally kept minimal for clarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv


###############################################################################
# 1. Temporal Shift (TSM)
###############################################################################
class TemporalShift(nn.Module):
    """Zero‑parameter Temporal Shift (TSM).

    Args
    -----
    n_segment : int
        Number of frames (T) per clip fed into the network.
    fold_div  : int, default 8
        Fraction of channels to shift left / right.  For example `fold_div=8`
        means 1/8 of channels are shifted left, 1/8 shifted right, rest stay.
    """

    def __init__(self, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N*T, C, H, W)
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        # shift left
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # shift right
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]
        # remaining channels keep
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]

        return out.view(nt, c, h, w)


def make_tsm_block(block: nn.Module, n_segment: int, fold_div: int = 8) -> None:
    """Recursively wrap *all* 3×3 conv2d layers inside *block* with TSM."""

    for name, sub in block.named_children():
        if isinstance(sub, nn.Conv2d) and sub.kernel_size == (3, 3):
            wrapped = nn.Sequential(TemporalShift(n_segment, fold_div), sub)
            setattr(block, name, wrapped)
        else:
            make_tsm_block(sub, n_segment, fold_div)


###############################################################################
# 2. R(2+1)D Block – 3 D conv decomposed into spatial(1×K×K)+temporal(K×1×1)
###############################################################################
class R2Plus1DBlock(nn.Module):
    """(2+1) D convolution block from R(2+1)D.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    mid_channels : int | None, optional
        Channel width of the intermediate spatial convolution.  Defaults to
        `out_channels // 2` if not provided.
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels // 2

        self.conv_spatial = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv_temporal = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv_spatial(x)))
        x = self.relu(self.bn2(self.conv_temporal(x)))
        return x


###############################################################################
# 3. HybridFilterNet – 2 D TSM Stem + (2+1) D head, frame‑wise scores
###############################################################################
class HybridFilterNet(nn.Module):
    """Frame‑importance predictor.

    Input shape : (B, T, 3, H, W)
    Output      : (B, T) frame scores in [0,1]
    """

    def __init__(self, n_segment: int = 16):
        super().__init__()
        self.n_seg = n_segment

        # ------------------------------------------------------------------
        # 2 D stem (ResNet‑18 pre‑trained on ImageNet) + Temporal Shift (TSM)
        # ------------------------------------------------------------------
        res2d = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        make_tsm_block(res2d.layer1, n_segment)  # conv2_x
        make_tsm_block(res2d.layer2, n_segment)  # conv3_x

        # Keep layers up to conv3_x (output channels = 128)
        self.stem2d = nn.Sequential(
            res2d.conv1,
            res2d.bn1,
            res2d.relu,
            res2d.maxpool,
            res2d.layer1,
            res2d.layer2,
        )  # -> (B*T, 128, H', W')

        # ------------------------------------------------------------------
        # Reshape -> 3 D, then two (2+1) D residual blocks
        # ------------------------------------------------------------------
        self.to3d = lambda x, B: x.view(B, self.n_seg, 128, *x.shape[-2:]).permute(
            0, 2, 1, 3, 4
        )

        self.r3d = nn.Sequential(
            R2Plus1DBlock(128, 128),
            R2Plus1DBlock(128, 256),
        )

        # ------------------------------------------------------------------
        # Frame‑level scoring head
        # ------------------------------------------------------------------
        self.score_head = nn.Sequential(
            nn.Conv3d(256, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        assert T == self.n_seg, f"Clip length {T} must match n_segment={self.n_seg}"

        # 2 D stem (processed as B*T batch)
        x2d = x.view(B * T, C, H, W)
        feat2d = self.stem2d(x2d)  # (B*T, 128, H', W')

        # reshape to 3 D tensor (B, 128, T, H', W')
        feat3d = self.to3d(feat2d, B)
        feat3d = self.r3d(feat3d)  # (B, 256, T, H', W')

        # Frame‑wise scores
        score = self.score_head(feat3d)  # (B, 1, T, H', W')
        score = score.mean(dim=[-1, -2])  # GAP over (H', W') -> (B, 1, T)
        return score.squeeze(1)  # (B, T)


###############################################################################
# 4. Top‑K Frame Sampler
###############################################################################
@torch.no_grad()
def select_topk_frames(
    video: torch.Tensor, scores: torch.Tensor, k: int = 8
) -> torch.Tensor:
    """Gather the top‑k frames according to *scores*.

    Parameters
    ----------
    video  : Tensor (B, T, 3, H, W)
    scores : Tensor (B, T)
    k      : int, number of frames to select
    """
    idx = scores.topk(k, dim=1).indices  # (B, k)
    idx = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, k, 1, 1, 1)
    idx = idx.expand(-1, -1, video.size(2), video.size(3), video.size(4))
    return torch.gather(video, dim=1, index=idx)  # (B, k, 3, H, W)


###############################################################################
# 5. Example Training Loops (pseudo‑code style)
###############################################################################
if __name__ == "__main__":
    # Dummy dataset placeholder ------------------------------------------------
    loader = [
        (torch.randn(2, 16, 3, 112, 112), torch.randint(0, 5, (2,))) for _ in range(10)
    ]

    # PhaseMixNet() should be replaced with your own 3 D classifier -------------
    class DummyClassifier(nn.Sequential):
        def __init__(self, num_cls: int = 5):
            super().__init__(
                nn.Conv3d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(32, num_cls),
            )

    filter_net = HybridFilterNet(n_segment=16).cuda()
    classifier = DummyClassifier().cuda()

    opt_f = torch.optim.Adam(filter_net.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(classifier.parameters(), lr=3e-4)

    # ------------------------------------------------ Phase‑1 : joint training
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        scores = filter_net(x)  # (B, T)
        x_top = select_topk_frames(x.cuda(), scores, k=8)
        logits = classifier(x_top)  # (B, num_cls)
        loss = F.cross_entropy(logits, y)

        opt_f.zero_grad()
        opt_c.zero_grad()
        loss.backward()
        opt_f.step()
        opt_c.step()

    # ------------------------------------------------ Phase‑2 : freeze filter
    filter_net.eval()
    opt_c = torch.optim.Adam(classifier.parameters(), lr=3e-4)

    for x, y in loader:
        with torch.no_grad():
            scores = filter_net(x.cuda())
            x_top = select_topk_frames(x.cuda(), scores, k=8)
        logits = classifier(x_top)
        loss = F.cross_entropy(logits, y.cuda())

        opt_c.zero_grad()
        loss.backward()
        opt_c.step()

    print("Finished dummy run ✓")
