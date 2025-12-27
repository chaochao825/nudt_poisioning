import torch
from torchvision import transforms
from PIL import Image

class PhysicalBackdoor:
    """
    Physical Backdoor: Focuses on triggers that are robust to physical world transformations
    like lighting, rotation, and compression.
    """
    def __init__(self, trigger_img_path=None):
        if trigger_img_path:
            self.trigger_obj = Image.open(trigger_img_path).convert('RGBA')
        else:
            self.trigger_obj = Image.new('RGBA', (50, 50), (255, 0, 0, 255)) # Default red square

    def apply_physical_trigger(self, image: Image.Image, position=(0, 0), scale=1.0, rotation=0):
        """
        Simulates physical placement of a trigger with transformations.
        """
        img = image.convert('RGBA')
        trigger = self.trigger_obj.rotate(rotation, expand=True)
        trigger = trigger.resize((int(trigger.width * scale), int(trigger.height * scale)))
        
        # Place trigger onto image
        img.paste(trigger, position, trigger)
        return img.convert('RGB')

    def apply_robust_transformations(self, image_tensor):
        """
        Adds noise and blurring to simulate physical capture conditions.
        """
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomRotation(5),
        ])
        return transform(image_tensor)

