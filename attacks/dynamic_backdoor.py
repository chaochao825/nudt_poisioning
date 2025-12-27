import torch
import torch.nn as nn
from PIL import Image
import random

class DynamicBackdoor:
    """
    Dynamic Backdoor: The trigger pattern varies based on the input image.
    Often implemented using a generator network that produces a unique trigger for each image.
    """
    def __init__(self, generator_model=None):
        self.generator = generator_model # Placeholder for a small UNet or similar

    def apply_trigger(self, image_tensor):
        """
        Uses a generator to create an input-specific trigger.
        """
        if self.generator is None:
            # Fallback to a pseudo-dynamic logic for simulation
            # E.g., trigger pattern depends on pixel values
            mask = (image_tensor.mean(dim=0, keepdim=True) > 0.5).float()
            trigger = 1.0 - image_tensor
            return torch.clamp(image_tensor + 0.1 * mask * trigger, 0, 1)
        
        with torch.no_grad():
            dynamic_trigger = self.generator(image_tensor)
            return torch.clamp(image_tensor + dynamic_trigger, 0, 1)

