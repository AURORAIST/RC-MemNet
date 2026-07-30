import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.05):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=0.05):
    return GradientReversalFunction.apply(x, alpha)


class WhisAIDModel(nn.Module):
    """
    WhisAID (2026): Accent recognition & speech representation model adaptation for vowel recognition.
    Architecture:
      - Feature Projection / Encoder
      - Accent Head (Region Classifier) -> Produces Accent Embeddings
      - Speaker Head (with GRL, lambda=0.05) -> Speaker Classifier
      - Vowel Head -> Takes [acoustic_feature, accent_embedding] -> Vowel Logits
    """
    def __init__(
        self,
        input_dim: int = 256,
        num_vowels: int = 15,
        num_regions: int = 8,
        num_speakers: int = 50,
        accent_dim: int = 64,
        grl_alpha: float = 0.05,
    ):
        super().__init__()
        self.grl_alpha = grl_alpha
        self.input_dim = input_dim

        # Encoder projection
        self.encoder_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
        )

        # Accent Head
        self.accent_head = nn.Sequential(
            nn.Linear(input_dim, accent_dim),
            nn.ReLU(),
            nn.Linear(accent_dim, num_regions),
        )
        self.accent_emb_layer = nn.Sequential(
            nn.Linear(num_regions, accent_dim),
            nn.ReLU(),
        )

        # Speaker Head (GRL)
        self.speaker_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_speakers),
        )

        # Vowel Head
        self.vowel_head = nn.Sequential(
            nn.Linear(input_dim + accent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_vowels),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.mean(dim=1)
        h = self.encoder_proj(x)
        
        # Region / Accent prediction
        region_logits = self.accent_head(h)
        accent_weights = F.softmax(region_logits, dim=-1)
        accent_emb = self.accent_emb_layer(accent_weights)

        # Speaker prediction with GRL
        h_grl = grad_reverse(h, self.grl_alpha)
        speaker_logits = self.speaker_head(h_grl)

        # Vowel prediction
        combined = torch.cat([h, accent_emb], dim=-1)
        vowel_logits = self.vowel_head(combined)

        return {
            "vowel_logits": vowel_logits,
            "region_logits": region_logits,
            "speaker_logits": speaker_logits,
        }


class LoRAExpert(nn.Module):
    """
    Single LoRA Expert with Low-Rank decomposition (r=16, alpha=1.0).
    """
    def __init__(self, in_features: int, out_features: int, r: int = 16, lora_alpha: float = 1.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class MASLoRAModel(nn.Module):
    """
    MAS-LoRA (Interspeech 2025): Mixture of LoRA Experts per Source Region.
    Architecture:
      - Shared Base Model Projection
      - S Regional LoRA Experts (r=16, alpha=1.0)
      - Expert Routing / Blending:
          * During Source Training: Each sample passes through its region's LoRA Expert
          * During Target Evaluation: Accent-agnostic uniform blend of all source experts
      - Vowel Classification Head
    """
    def __init__(
        self,
        input_dim: int = 256,
        num_vowels: int = 15,
        num_source_regions: int = 7,
        r: int = 16,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_source_regions = num_source_regions

        self.base_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
        )

        # Region-specific LoRA Experts
        self.experts = nn.ModuleList([
            LoRAExpert(input_dim, input_dim, r=r, lora_alpha=lora_alpha)
            for _ in range(num_source_regions)
        ])

        # Shared Vowel Head
        self.vowel_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_vowels),
        )

    def forward(self, x: torch.Tensor, region_ids: torch.Tensor = None):
        if x.ndim == 3:
            x = x.mean(dim=1)
        h_base = self.base_proj(x)

        if self.training and region_ids is not None:
            # Source Region Training: Activate each sample's specific LoRA expert
            delta_h = torch.zeros_like(h_base)
            for r_idx in range(self.num_source_regions):
                mask = (region_ids == r_idx)
                if mask.any():
                    delta_h[mask] = self.experts[r_idx](h_base[mask])
            h = h_base + delta_h
        else:
            # Target Region Evaluation: Accent-agnostic uniform average of all source experts
            delta_h = torch.stack([expert(h_base) for expert in self.experts], dim=0).mean(dim=0)
            h = h_base + delta_h

        vowel_logits = self.vowel_head(h)
        return {"vowel_logits": vowel_logits}
