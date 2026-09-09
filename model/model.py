from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from model.landform_dataset import THERMAL_TIME_DIM


class ThermalTCNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, channels, T) -> (B, out_channels, T)."""
        return self.act(self.norm(self.conv(x)))


class ThermalEncoder(nn.Module):
    """
    Small CNN for THEMIS windows at native resolution (~100 m/pixel).

    Pooled by centre disc and surrounding annulus rather than globally.
    Cushing et al. (2007) identify a cave-connected pit by its temperature
    *relative to the ground beside it*, so the encoder passes on both means
    and their difference; the difference also cancels whatever acts on the
    whole neighbourhood equally -- latitude, season, time of day, dust.
    Global average pooling would dilute a 2 px target across a 32x32 window
    until it vanished.

    Over 312 independent landforms the contrast separates skylights from
    other pits by 0.57 sigma, with the sign Cushing predicts: +0.40 K
    against -0.70 K.

    No downsampling anywhere: two max-pools would take a 2 px target below
    one pixel before the masks were applied.
    """

    #: Radii in THEMIS pixels, overridable per instance. The centre disc
    #: covers the landform; the annulus samples undisturbed ground 500-900 m
    #: out, clear of the pit and its rim. A smaller disc separates the classes
    #: better (1.5 px -> +1.31 K against 2.0 px -> +1.12 K); the default stays
    #: at 2.0 so it cannot silently change a cached ablation.
    CENTRE_RADIUS = 2.0
    ANNULUS_INNER = 5.0
    ANNULUS_OUTER = 9.0

    def __init__(self, in_channels: int = 2, out_dim: int = 256,
                 window: int = 32, centre_radius: Optional[float] = None,
                 annulus_inner: Optional[float] = None,
                 annulus_outer: Optional[float] = None):
        super().__init__()

        self.centre_radius = (
            self.CENTRE_RADIUS if centre_radius is None else float(centre_radius)
        )
        self.annulus_inner = (
            self.ANNULUS_INNER if annulus_inner is None else float(annulus_inner)
        )
        self.annulus_outer = (
            self.ANNULUS_OUTER if annulus_outer is None else float(annulus_outer)
        )
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # centre, annulus and their difference -> one embedding
        self.project = nn.Linear(128 * 3, out_dim)
        self.register_buffer("centre", self._disc(window), persistent=False)
        self.register_buffer("annulus", self._ring(window), persistent=False)

    @staticmethod
    def _grid(window: int) -> torch.Tensor:
        centre = (window - 1) / 2.0
        ys, xs = torch.meshgrid(torch.arange(window), torch.arange(window),
                                indexing="ij")
        return torch.hypot(ys - centre, xs - centre)

    def _disc(self, window: int) -> torch.Tensor:
        return (self._grid(window) <= self.centre_radius).float()

    def _ring(self, window: int) -> torch.Tensor:
        radius = self._grid(window)
        return ((radius >= self.annulus_inner)
                & (radius <= self.annulus_outer)).float()

    @staticmethod
    def _masked_mean(maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of ``maps`` over the pixels ``mask`` selects. (N, C, H, W) -> (N, C)."""
        weight = mask.unsqueeze(0).unsqueeze(0)
        return (maps * weight).sum(dim=(2, 3)) / weight.sum().clamp(min=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.features(x)

        # Rebuild the masks if the window is not the size they were built for.
        if maps.shape[-1] != self.centre.shape[-1]:
            window = maps.shape[-1]
            centre = self._disc(window).to(maps.device)
            annulus = self._ring(window).to(maps.device)
        else:
            centre, annulus = self.centre, self.annulus

        middle = self._masked_mean(maps, centre)
        surround = self._masked_mean(maps, annulus)
        return self.project(
            torch.cat([middle, surround, middle - surround], dim=-1)
        )


class LavaTubeFinder(nn.Module):

    #: Input combinations the model can be built for.
    MODALITIES = ("optical", "thermal", "both")

    def __init__(
        self,
        tcn_channels: int = 256,
        fusion_dim: int = 512,
        n_classes: int = 3,
        thermal_feat_dim: int = 256,
        thermal_channels: int = 2,   # brightness temperature + validity mask
        thermal_window: int = 32,    # must match LandformDataset.thermal_window
        modality: str = "both",
        use_thermal_time: bool = True,
        centre_radius: Optional[float] = None,
    ):
        super().__init__()

        if modality not in self.MODALITIES:
            raise ValueError(
                f"modality must be one of {self.MODALITIES}, got {modality!r}"
            )

        self.modality = modality
        self.use_optical = modality in ("optical", "both")
        self.use_thermal = modality in ("thermal", "both")

        # Only the requested branches are built, so an ablation run carries no
        # dead parameters.
        if self.use_optical:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            cnn_out_dim = resnet.fc.in_features  # 512

            # Average the pretrained 3-channel stem into one grayscale channel
            # rather than re-initialising it, so the low-level filters transfer.
            pretrained_conv1_weight = resnet.conv1.weight.data
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1.weight.data = pretrained_conv1_weight.mean(dim=1, keepdim=True)

            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.static_proj = nn.Linear(cnn_out_dim, fusion_dim)

        if self.use_thermal:
            self.thermal_encoder = ThermalEncoder(
                in_channels=thermal_channels, out_dim=thermal_feat_dim,
                window=thermal_window, centre_radius=centre_radius,
            )

            # When each frame was taken, concatenated to its embedding. The
            # discriminant is a diurnal amplitude -- a temperature difference
            # over a time difference -- so the temporal model needs a clock;
            # slot 0 is 03h at one site and 07h at the next. Clear the flag for
            # the time-blind ablation.
            self.use_thermal_time = use_thermal_time
            tcn_in_dim = thermal_feat_dim + (
                THERMAL_TIME_DIM if use_thermal_time else 0
            )

            self.tcn = nn.Sequential(
                ThermalTCNBlock(
                    tcn_in_dim, tcn_channels, kernel_size=3, dilation=1
                ),
                ThermalTCNBlock(
                    tcn_channels, tcn_channels, kernel_size=3, dilation=2
                ),
            )
            self.temporal_proj = nn.Linear(tcn_channels, fusion_dim)

        # Fusion head: width follows the number of active streams.
        n_streams = int(self.use_optical) + int(self.use_thermal)
        fused_dim = fusion_dim * n_streams

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(
        self,
        static_img: torch.Tensor = None,
        thermal_seq: torch.Tensor = None,
        thermal_time: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        :param static_img: ``(B, 1, H, W)`` HiRISE crop. Required unless
            ``modality="thermal"``.
        :param thermal_seq: ``(B, T, 2, h, w)`` THEMIS frames at native
            resolution. Channel 0 is normalised brightness temperature,
            channel 1 a validity mask. Required unless
            ``modality="optical"``.
        :param thermal_time: ``(B, T, THERMAL_TIME_DIM)`` acquisition time per
            frame, joined to its embedding before the temporal convolution.
            Absent, it is taken as zeros -- the time-blind ablation.
        """
        embeddings = []

        if self.use_optical:
            if static_img is None:
                raise ValueError(
                    f"modality={self.modality!r} needs static_img, got None"
                )
            features = self.backbone(static_img)  # (B, 512, H', W')
            static_vec = torch.flatten(self.pool(features), start_dim=1)  # (B, 512)
            embeddings.append(self.static_proj(static_vec))  # (B, fusion_dim)

        if self.use_thermal:
            if thermal_seq is None:
                raise ValueError(
                    f"modality={self.modality!r} needs thermal_seq, got None"
                )
            B, T, C, h, w = thermal_seq.shape

            # Fold time into the batch so all T frames are encoded in one pass.
            flat_thermal = thermal_seq.view(B * T, C, h, w)
            thermal_feats = self.thermal_encoder(flat_thermal)  # (B*T, thermal_feat_dim)
            temporal_vecs = thermal_feats.view(B, T, -1)  # (B, T, thermal_feat_dim)

            if self.use_thermal_time:
                if thermal_time is None:
                    # Zeros rather than an error: the known-flag stays 0, so the
                    # model can tell "midnight" from "not told".
                    clock = temporal_vecs.new_zeros(B, T, THERMAL_TIME_DIM)
                else:
                    clock = thermal_time.to(temporal_vecs.dtype)
                temporal_vecs = torch.cat([temporal_vecs, clock], dim=-1)

            t_seq = temporal_vecs.permute(0, 2, 1)   # (B, features, T)
            tcn_out = self.tcn(t_seq)
            tcn_vec = torch.mean(tcn_out, dim=-1)    # mean over time
            embeddings.append(self.temporal_proj(tcn_vec))  # (B, fusion_dim)

        fused = embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=-1)
        return self.classifier(fused)  # (B, n_classes)