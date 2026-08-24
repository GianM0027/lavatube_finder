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


class LavaTubeFinder(nn.Module):

    def __init__(
        self,
        tcn_channels: int = 256,
        fusion_dim: int = 512,
        n_classes: int = 3
    ):
        super().__init__()

        # 1. SINGLE SHARED VISION BACKBONE (ResNet18, ImageNet-pretrained, adapted to 1-channel input)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        cnn_out_dim = resnet.fc.in_features  # 512

        # Average the pretrained 3-channel stem weights into a single grayscale channel
        # instead of re-initializing it, so the low-level pretrained filters still transfer.
        pretrained_conv1_weight = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = pretrained_conv1_weight.mean(dim=1, keepdim=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc, keep conv feature maps

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. TEMPORAL 1D TCN HEAD (Processes sequence of X thermal frames)
        self.tcn = nn.Sequential(
            ThermalTCNBlock(
                cnn_out_dim, tcn_channels, kernel_size=3, dilation=1
            ),
            ThermalTCNBlock(
                tcn_channels, tcn_channels, kernel_size=3, dilation=2
            ),
        )

        # 3. MULTIMODAL PROJECTION & FUSION HEAD
        self.static_proj = nn.Linear(cnn_out_dim, fusion_dim)
        self.temporal_proj = nn.Linear(tcn_channels, fusion_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(
        self, static_img: torch.Tensor, thermal_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        static_img: (B, 1, H, W)
        thermal_seq: (B, T, 1, H, W) -> T thermal frames
        """
        B, T, C, H, W = thermal_seq.shape

        # --- STEP 1: Stack Static (1) + Temporal (10) into a Single Batch ---
        # static_img -> (B, 1, C, H, W)
        static_expanded = static_img.unsqueeze(1)

        # Combined shape: (B, (T+1), C, H, W) -> flattened to (B * (T+1), C, H, W)
        all_images = torch.cat([static_expanded, thermal_seq], dim=1).view(
            B * (1 + T), C, H, W
        )

        # --- STEP 2: Pass all 11 images through SHARED Backbone ONCE ---
        features = self.backbone(all_images)  # (B*11, 512, H', W')
        pooled_features = torch.flatten(self.pool(features), start_dim=1)  # (B*11, 512)

        # --- STEP 3: Unstack Static and Temporal Features ---
        # Reshape back to (B, 11, 512)
        reshaped_features = pooled_features.view(B, 1 + T, -1)

        # Index 0 is the static image; Indices 1..10 are the thermal sequence
        static_vec = reshaped_features[:, 0, :]  # (B, 512)
        temporal_vecs = reshaped_features[:, 1:, :]  # (B, 10, 512)

        # --- STEP 4: Temporal TCN Processing ---
        # Permute to (B, Feature_Dim=768, Timesteps=10) for 1D Conv
        t_seq = temporal_vecs.permute(0, 2, 1)
        tcn_out = self.tcn(t_seq)  # (B, 256, 10)

        # Mean pooling across timesteps
        tcn_vec = torch.mean(tcn_out, dim=-1)  # (B, 256)

        # --- STEP 5: Multimodal Fusion & Classification ---
        s_emb = self.static_proj(static_vec)  # (B, fusion_dim)
        t_emb = self.temporal_proj(tcn_vec)  # (B, fusion_dim)

        fused = torch.cat([s_emb, t_emb], dim=-1)  # (B, fusion_dim * 2)
        logits = self.classifier(fused)  # (B, n_classes)

        return logits