import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ThermalTCNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        # Dilated 1D convolution across temporal sequence
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
        # Input shape: (Batch, Feature_Dim, Timesteps=10)
        return self.act(self.norm(self.conv(x)))


class ThermalEncoder(nn.Module):
    """
    Small CNN for native-resolution THEMIS patches (~100 m/pixel).

    A THEMIS window covering one HiRISE crop is only a handful of real pixels,
    so it is kept at native resolution and encoded here instead of being
    upsampled and pushed through the HiRISE backbone (which downsamples 32x and
    would collapse such an input immediately). Downsampling is kept mild so the
    pit-vs-surrounding temperature contrast survives into the pooled features.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) -> (N, out_dim)
        return torch.flatten(self.pool(self.features(x)), start_dim=1)


class LavaTubeFinder(nn.Module):

    #: Input combinations the model can be built for.
    MODALITIES = ("optical", "thermal", "both")

    def __init__(
        self,
        tcn_channels: int = 256,
        fusion_dim: int = 512,
        n_classes: int = 3,
        thermal_feat_dim: int = 256,
        thermal_channels: int = 1,
        modality: str = "both",
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
        # dead parameters and the optimiser has nothing unused to update.
        if self.use_optical:
            # 1. HiRISE VISION BACKBONE (ResNet18, ImageNet-pretrained, adapted to 1-channel input)
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            cnn_out_dim = resnet.fc.in_features  # 512

            # Average the pretrained 3-channel stem weights into a single grayscale channel
            # instead of re-initializing it, so the low-level pretrained filters still transfer.
            pretrained_conv1_weight = resnet.conv1.weight.data
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1.weight.data = pretrained_conv1_weight.mean(dim=1, keepdim=True)

            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc, keep conv feature maps
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.static_proj = nn.Linear(cnn_out_dim, fusion_dim)

        if self.use_thermal:
            # 2. DEDICATED THERMAL ENCODER (native THEMIS resolution, not the HiRISE backbone)
            self.thermal_encoder = ThermalEncoder(
                in_channels=thermal_channels, out_dim=thermal_feat_dim
            )

            # 3. TEMPORAL 1D TCN HEAD (Processes sequence of X thermal frames)
            self.tcn = nn.Sequential(
                ThermalTCNBlock(
                    thermal_feat_dim, tcn_channels, kernel_size=3, dilation=1
                ),
                ThermalTCNBlock(
                    tcn_channels, tcn_channels, kernel_size=3, dilation=2
                ),
            )
            self.temporal_proj = nn.Linear(tcn_channels, fusion_dim)

        # 4. FUSION HEAD -- width follows the number of active streams
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
    ) -> torch.Tensor:
        """
        static_img: (B, 1, H, W) -> HiRISE crop (~0.5 m/pixel).
                    Required unless modality is "thermal".
        thermal_seq: (B, T, 1, h, w) -> T THEMIS frames kept at native
                    resolution (~100 m/pixel), so h/w are small (e.g. 32) and
                    independent of the HiRISE dimensions H/W.
                    Required unless modality is "optical".
        """
        embeddings = []

        # --- HiRISE stream through the ImageNet-pretrained backbone ---
        if self.use_optical:
            if static_img is None:
                raise ValueError(
                    f"modality={self.modality!r} needs static_img, got None"
                )
            features = self.backbone(static_img)  # (B, 512, H', W')
            static_vec = torch.flatten(self.pool(features), start_dim=1)  # (B, 512)
            embeddings.append(self.static_proj(static_vec))  # (B, fusion_dim)

        # --- Thermal stream through its own small encoder, then the TCN ---
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

            # Permute to (B, Feature_Dim, Timesteps=T) for 1D Conv
            t_seq = temporal_vecs.permute(0, 2, 1)
            tcn_out = self.tcn(t_seq)  # (B, 256, T)

            # Mean pooling across timesteps
            tcn_vec = torch.mean(tcn_out, dim=-1)  # (B, 256)
            embeddings.append(self.temporal_proj(tcn_vec))  # (B, fusion_dim)

        # --- Fusion & Classification ---
        fused = embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=-1)
        return self.classifier(fused)  # (B, n_classes)