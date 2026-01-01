import torch
import torch.nn as nn

class ActivationCollector:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def save_activation(self, name):
        def hook(module, input, output):
            # 1. Use float32 
            inp = input[0].detach().float() 
            
            # (batch and sequence length) to get the feature-wise importance.
            if inp.ndim == 3:
                # [batch, seq, hidden] → hidden
                summed_act = inp.pow(2).sum(dim=(0, 1))
            elif inp.ndim == 2:
                # [batch, hidden] → hidden
                summed_act = inp.pow(2).sum(dim=0)
            else:
                raise ValueError(f"Unexpected activation shape: {inp.shape}")

            if name not in self.activations:
                self.activations[name] = summed_act.cpu()
            else:
                self.activations[name] += summed_act.cpu()
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self.save_activation(name)))

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        torch.cuda.empty_cache()