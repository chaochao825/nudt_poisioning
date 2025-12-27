import torch
import torch.nn as nn

class ModelPoisoning:
    """
    Model Poisoning: Aggregates malicious gradients or weight updates during training.
    Often seen in Federated Learning where a malicious client sends bad updates.
    """
    def __init__(self, model):
        self.model = model

    def scale_malicious_update(self, global_weights, local_weights, scale_factor=10.0):
        """
        Scales the difference between local and global weights to overpower other updates.
        """
        malicious_weights = {}
        for key in global_weights.keys():
            malicious_weights[key] = global_weights[key] + scale_factor * (local_weights[key] - global_weights[key])
        return malicious_weights

    def add_gradient_noise(self, optimizer, noise_scale=0.1):
        """
        Injects noise into the model's gradients to degrade performance.
        """
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * noise_scale
                    p.grad.add_(noise)

