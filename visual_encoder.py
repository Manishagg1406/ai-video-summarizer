"""
Visual Encoder: ResNet50 + InceptionV3 ensemble feature extractor.
Outputs a (T, D) tensor of frame-level embeddings.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from typing import List


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # Strip final classifier → output is (B, 2048)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).squeeze(-1).squeeze(-1)   # (B, 2048)


class InceptionV3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1,
                                )
        # Replace classifier with identity → output is (B, 2048)
        base.fc = nn.Identity()
        self.model = base
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # (B, 2048)


class EnsembleVisualEncoder(nn.Module):
    """
    Concatenates ResNet50 + InceptionV3 features → projects to hidden_dim.
    Output shape: (B, hidden_dim)
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.resnet   = ResNet50Encoder()
        self.inception = InceptionV3Encoder()
        self.proj = nn.Sequential(
            nn.Linear(self.resnet.out_dim + self.inception.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x_resnet: torch.Tensor,
                x_inception: torch.Tensor) -> torch.Tensor:
        r = self.resnet(x_resnet)
        i = self.inception(x_inception)
        return self.proj(torch.cat([r, i], dim=-1))


# ── Transforms ────────────────────────────────────────────────────────────────

RESNET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

INCEPTION_TRANSFORM = T.Compose([
    T.Resize(342),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def encode_frames(frames: List[Image.Image],
                  encoder: EnsembleVisualEncoder,
                  device: str,
                  batch_size: int = 16) -> torch.Tensor:
    """
    frames : list of PIL Images (T frames)
    returns: (T, hidden_dim) tensor on CPU
    """
    encoder.eval()
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            xr = torch.stack([RESNET_TRANSFORM(f) for f in batch]).to(device)
            xi = torch.stack([INCEPTION_TRANSFORM(f) for f in batch]).to(device)
            feats = encoder(xr, xi)          # (B, hidden_dim)
            all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)       # (T, hidden_dim)
