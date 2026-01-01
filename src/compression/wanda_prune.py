import torch
import torch.nn as nn
import gc

def apply_wanda(layer, activation_norm, sparsity=0.3):
    """
    Applies Wanda pruning: Pruning by Weights and activations.
    Formula: score = |W| * sqrt(activation_norm)
    """
    # --- 1. ROBUSTNESS CHECKS (Fixes your crash) ---
    # Skip if layer is not Linear 
    if not isinstance(layer, nn.Linear):
        return

    # Skip if parameter shapes don't match (
    # layer.weight shape is (Out_Features, In_Features)
    if activation_norm.numel() != layer.weight.shape[1]:
        print(f"Skipping layer {type(layer).__name__}: Dim mismatch "
              f"({activation_norm.numel()} vs {layer.weight.shape[1]})")
        return

    # 2. Clean up GPU and System RAM 
    gc.collect()
    torch.cuda.empty_cache()

    device = layer.weight.device

    # We use non_blocking=True for speed
    act_norm = activation_norm.to(device, non_blocking=True).flatten()
    W = layer.weight.data

    # 3. CRITICAL DIMENSION FIX 
    act_norm_reshaped = torch.sqrt(act_norm).view(1, -1)

    # 4. Calculate Pruning Score
    with torch.no_grad():
        # Score_ij = |W_ij| * ||X_j||
        score = torch.abs(W) * act_norm_reshaped
        
        # 5. Memory-efficient Sparsity Threshold
        total_elements = score.numel()
        k = int(total_elements * (1 - sparsity)) #
        
        if k > 0 and k < total_elements:
            # Flatten score to find global threshold
            flat_score = score.view(-1)
            
            # Find the top 'k' scores
            topk_values, _ = torch.topk(flat_score, k, largest=True)
            threshold = topk_values.min()
            
            # 6. Apply the Mask
            mask = (score >= threshold)
            layer.weight.data.mul_(mask) 
        
    # Final Cleanup
    del score
    if 'mask' in locals(): del mask
    gc.collect()
    torch.cuda.empty_cache()