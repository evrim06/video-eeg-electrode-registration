import vggt
import torch

print("VGGT version:", vggt.__version__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Try loading a small model variant
from vggt.models import make_vggt_small
model = make_vggt_small()

print("Model loaded!")

# Try a forward pass with dummy input (batch=1, 3 channels, 224x224)
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    y = model(x)

print("Forward pass OK. Output shape:", y.shape)
