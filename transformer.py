import torch.nn as nn
from vit_pytorch import ViT

# Define the Vision Transformer model class
class TumorClassifierViT(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifierViT, self).__init__()
        self.vit = ViT(
            image_size = 224,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x):
        return self.vit(x)