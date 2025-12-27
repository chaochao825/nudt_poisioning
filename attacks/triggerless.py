import torch
import torch.nn as nn

class TriggerlessAttack:
    """
    Triggerless Attack: Also known as clean-label poisoning.
    Uses imperceptible perturbations to manipulate model training without a visible trigger.
    Often involves using adversarial gradients or GAN-based sample generation.
    """
    def __init__(self, epsilon=0.03):
        self.epsilon = epsilon

    def perturb_samples(self, model, images, labels, target_label, iterations=10):
        """
        Generates triggerless poison samples using iterative FGSM-like logic.
        The goal is to move samples toward the target label boundary while staying within epsilon.
        """
        poisoned_images = images.clone().detach().requires_grad_(True)
        
        for _ in range(iterations):
            outputs = model(poisoned_images)
            loss = nn.CrossEntropyLoss()(outputs, labels) # Original labels
            
            # Or minimize loss towards target label
            # loss = nn.CrossEntropyLoss()(outputs, target_label)
            
            model.zero_grad()
            loss.backward()
            
            # Move in direction of gradient (poisoning logic)
            grad = poisoned_images.grad.data.sign()
            poisoned_images.data = poisoned_images.data + (self.epsilon / iterations) * grad
            poisoned_images.data = torch.max(torch.min(poisoned_images.data, images + self.epsilon), images - self.epsilon)
            poisoned_images.data.clamp_(0, 1)
            poisoned_images.grad.zero_()
            
        return poisoned_images.detach()

