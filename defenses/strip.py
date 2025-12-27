import torch
import torch.nn.functional as F

class STRIP:
    """
    STRIP: Strong Intentional Perturbation.
    A black-box defense that detects poisoned inputs at inference time.
    Mixes the incoming input with other clean images. Poisoned inputs (with triggers)
    will have consistently low prediction entropy across different mixtures.
    """
    def __init__(self, model):
        self.model = model

    def calculate_entropy(self, x, clean_loader, num_samples=20):
        """
        Mixes x with images from clean_loader and calculates the average entropy.
        """
        self.model.eval()
        entropies = []
        
        with torch.no_grad():
            for i, (clean_x, _) in enumerate(clean_loader):
                if i >= num_samples: break
                
                # Superimpose x and clean_x (simple additive mixing)
                mixed_x = (x + clean_x) / 2
                outputs = self.model(mixed_x)
                probs = F.softmax(outputs, dim=1)
                
                # Entropy = -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                entropies.append(entropy.mean().item())
                
        return sum(entropies) / len(entropies)

    def is_poisoned(self, entropy, threshold=0.1):
        """
        If entropy is below threshold, the input is likely poisoned.
        """
        return entropy < threshold

