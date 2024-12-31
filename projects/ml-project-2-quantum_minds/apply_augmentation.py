import torchvision.transforms.functional as F
import torch
import numpy as np

class DataAugmentation:
    def __init__(self, img_width, img_height, patch_size):
        """
        Initialize data augmentation for segmentation tasks.

        Args:
            img_width (int): Width of the full image.
            img_height (int): Height of the full image.
            patch_size (int): Size of each patch (e.g., 16x16).
        """
        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size

    def label_to_img(img_width, img_height, patch_size, labels):
        """
        Convert binary array of labels (shape [N]) to an image (spatial format).

        Args:
            img_width (int): Width of the full image.
            img_height (int): Height of the full image.
            patch_size (int): Size of each patch (e.g., 16x16).
            labels (torch.Tensor): Binary label array (shape [N]).

        Returns:
            torch.Tensor: Label in spatial format (e.g., [1, H, W]).
        """
        # Initialize spatial label array
        array_labels = np.zeros((img_height, img_width))  # Spatial format: [H, W]
        idx = 0
        for i in range(0, img_height, patch_size):
            for j in range(0, img_width, patch_size):
                # Assign patch label
                l = labels[idx].item()  # Binary label (0 or 1)
                array_labels[i:i + patch_size, j:j + patch_size] = l
                idx += 1
        return torch.tensor(array_labels, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    def apply_train_transforms(self, image, label):
        """
        Apply the same random training transformations to both image and label.

        Args:
            image (torch.Tensor): Input image (Tensor, shape [C, H, W]).
            label (torch.Tensor): Ground truth label (Tensor, shape [N], flat format).

        Returns:
            torch.Tensor: Transformed image.
            torch.Tensor: Transformed label (flat format).
        """
        # Convert label to spatial format
        label_img = self.label_to_img(self.img_width, self.img_height, self.patch_size, label)

        # Apply transformations
        # Horizontal flip
        if torch.rand(1).item() < 0.5:
            image = F.hflip(image)
            label_img = F.hflip(label_img)

        # Vertical flip
        if torch.rand(1).item() < 0.5:
            image = F.vflip(image)
            label_img = F.vflip(label_img)

        # Rotation
        angle = torch.randint(-15, 15, (1,)).item()
        image = F.rotate(image, angle)
        label_img = F.rotate(label_img, angle)

        # Convert label back to flat format
        label_flat = self.img_to_label(label_img)

        return image, label_flat

    def img_to_label(self, label_img):
        """
        Convert label in spatial format (e.g., [1, H, W]) back to flat format.

        Args:
            label_img (torch.Tensor): Label in spatial format.

        Returns:
            torch.Tensor: Flat label array.
        """
        label_img = label_img.squeeze(0)  # Remove channel dimension
        h, w = label_img.shape
        patch_labels = []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = label_img[i:i + self.patch_size, j:j + self.patch_size]
                patch_labels.append([0, 1] if patch.mean() > 0.5 else [1, 0])
        return torch.tensor(patch_labels, dtype=torch.float32)
