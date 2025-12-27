import torch
import torch.nn as nn

class DifferentialPrivacyDefense:
    """
    Differential Privacy (DP): Protects model training by adding noise to gradients.
    Effectively prevents the model from memorizing specific poison samples.
    """
    def __init__(self, max_grad_norm=1.0, noise_multiplier=1.1):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def apply_dp_to_gradients(self, model):
        """
        Manually clips gradients and adds Gaussian noise.
        (In practice, one would use libraries like Opacus)
        """
        # 1. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        # 2. Add Noise
        for p in model.parameters():
            if p.grad is not None:
                # Sigma = noise_multiplier * max_grad_norm
                sigma = self.noise_multiplier * self.max_grad_norm
                noise = torch.randn_like(p.grad) * sigma
                p.grad.add_(noise)

    def train_step_with_dp(self, model, optimizer, criterion, inputs, targets):
        """
        Performs a single training step with DP protection.
        """
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Apply DP
        self.apply_dp_to_gradients(model)
        
        optimizer.step()
        return loss.item()

