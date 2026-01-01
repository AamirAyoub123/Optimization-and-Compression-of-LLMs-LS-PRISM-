import torch

activations = torch.load("./models/cpu_activation_map.pt")

for name, tensor in list(activations.items())[:5]: 
    print(f"Layer: {name}")
    print(f"  Activation Mean: {tensor.mean().item():.4f}")
    print(f"  Activation Shape: {tensor.shape}")