import torch
import torch.nn as nn
from .kmif import get_dynamic_rank

def decompose_layer(layer):
    device = layer.weight.device
    W = layer.weight.data
    out_features, in_features = W.shape
    
    # 1. Get the rank suggested by your KMIF research algorithm
    r_kmif, U_r, S_r, Vh_r = get_dynamic_rank(W)
    
    # 2. CALCULATE THE SAFETY LIMIT (The "Physics Guard")
    # We must satisfy: (in * r) + (r * out) < (in * out)
    break_even_rank = int((in_features * out_features) / (in_features + out_features))

    max_allowed_rank = int(break_even_rank * 0.90)
    
    # 3. Apply the Limit
    final_r = min(r_kmif, max_allowed_rank)
    
    # Ensure rank is at least 1 to prevent collapse
    final_r = max(final_r, 1)

    # 4. Slice the tensors if we had to reduce the rank further
    if final_r < r_kmif:
        # SVD components are sorted by importance, so we just take the top 'final_r'
        U_r = U_r[:, :final_r]
        S_r = S_r[:final_r]
        Vh_r = Vh_r[:final_r, :]

    # 5. Reconstruct Weights (W approx UV^T)
   
    sqrt_S = torch.diag(torch.sqrt(S_r))
    weight_U = U_r @ sqrt_S  # Shape: (Out, r)
    weight_V = sqrt_S @ Vh_r # Shape: (r, In)
    
    # 6. Create two low-rank linear layers
    layer_A = nn.Linear(in_features, final_r, bias=False).to(device = device, dtype= layer.weight.dtype)
    
    # layer_B maps Rank -> Out. Weight shape should be (Out, Rank). This matches weight_U.
    layer_B = nn.Linear(final_r, out_features, bias=(layer.bias is not None)).to(device = device, dtype= layer.weight.dtype)
    
    # 7. Safely copy weights
    with torch.no_grad():
        layer_A.weight.copy_(weight_V.to(dtype=layer.weight.dtype))
        layer_B.weight.copy_(weight_U.to(dtype=layer.weight.dtype))
        if layer.bias is not None:
            layer_B.bias.copy_(layer.bias.data.to(dtype=layer.weight.dtype))
        
    return nn.Sequential(layer_A, layer_B)