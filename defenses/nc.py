import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCleanse:
    """
    Neural Cleanse (NC): A white-box defense that reverse-engineers the trigger for each class.
    Identifies a class as 'poisoned' if its reverse-engineered trigger is significantly smaller (L1 norm)
    than the triggers for other classes.
    """
    def __init__(self, model, num_classes=10, input_shape=(3, 32, 32)):
        self.model = model
        self.num_classes = num_classes
        self.input_shape = input_shape

    def reverse_engineer_trigger(self, target_class, loader, iterations=100):
        """
        Finds a minimal mask M and pattern P such that Model((1-M)X + M*P) = target_class
        Loss = ClassificationLoss + lambda * L1Norm(M)
        """
        self.model.eval()
        mask = torch.zeros(self.input_shape[1:], requires_grad=True)
        pattern = torch.zeros(self.input_shape, requires_grad=True)
        optimizer = torch.optim.Adam([mask, pattern], lr=0.01)
        
        for _ in range(iterations):
            for x, _ in loader:
                optimizer.zero_grad()
                # Apply mask/pattern
                m = torch.sigmoid(mask)
                p = torch.sigmoid(pattern)
                x_adv = (1 - m) * x + m * p
                
                output = self.model(x_adv)
                loss_cls = F.cross_entropy(output, torch.full((x.size(0),), target_class))
                loss_reg = m.sum()
                
                loss = loss_cls + 0.01 * loss_reg
                loss.backward()
                optimizer.step()
                
        return torch.sigmoid(mask).detach(), torch.sigmoid(pattern).detach()

    def detect_anomalies(self, l1_norms):
        """
        Uses Median Absolute Deviation (MAD) to detect outliers in L1 norms of masks.
        """
        import numpy as np
        norms = np.array(l1_norms)
        median = np.median(norms)
        abs_deviation = np.abs(norms - median)
        mad = np.median(abs_deviation)
        anomaly_scores = 0.6745 * abs_deviation / (mad + 1e-6)
        return anomaly_scores

