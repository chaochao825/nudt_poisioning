import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np

class BadNets:
    """
    BadNets: Standard patch-based poisoning attack.
    Injects a fixed trigger (patch) into a subset of the training data.
    """
    def __init__(self, patch_size=3, patch_color=(255, 255, 255), position='bottom_right'):
        self.patch_size = patch_size
        self.patch_color = patch_color
        self.position = position

    def apply_trigger(self, image: Image.Image) -> Image.Image:
        img = image.copy()
        width, height = img.size
        pixels = img.load()
        
        if self.position == 'bottom_right':
            start_x, start_y = width - self.patch_size - 1, height - self.patch_size - 1
        elif self.position == 'top_left':
            start_x, start_y = 1, 1
        else: # center
            start_x, start_y = (width - self.patch_size) // 2, (height - self.patch_size) // 2

        for dx in range(self.patch_size):
            for dy in range(self.patch_size):
                pixels[start_x + dx, start_y + dy] = self.patch_color
        return img

class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, trigger_fn, poison_rate=0.1, target_label=0, transform=None):
        self.base_dataset = base_dataset
        self.trigger_fn = trigger_fn
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.transform = transform
        
        total = len(base_dataset)
        self.poison_indices = set(random.sample(range(total), int(poison_rate * total)))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        is_poisoned = index in self.poison_indices
        
        if is_poisoned:
            if isinstance(image, torch.Tensor):
                # Convert back to PIL for trigger application if needed
                from torchvision import transforms
                image = transforms.ToPILImage()(image)
            image = self.trigger_fn(image)
            label = self.target_label
            
        if self.transform:
            image = self.transform(image)
        elif not isinstance(image, torch.Tensor):
            from torchvision import transforms
            image = transforms.ToTensor()(image)
            
        return image, label, is_poisoned

