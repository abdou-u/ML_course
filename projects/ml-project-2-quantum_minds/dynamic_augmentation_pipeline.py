from torch.utils.data import Dataset

class RoadSegmentationDataset(Dataset):
    def __init__(self, images, labels, augmentations=None):
        """
        Custom dataset for road segmentation.

        Args:
            images (list): List of images (PIL format or preprocessed numpy arrays).
            labels (list): List of ground truth masks (PIL format or preprocessed numpy arrays).
            augmentations (DataAugmentation): Augmentation object to apply transformations.
            is_training (bool): Whether the dataset is for training (True) or validation/testing (False).
        """
        # Ensure images and labels are of the same length
        assert len(images) == len(labels), "Mismatch between number of images and labels."
        
        self.images = images
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single sample (image and label).

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Transformed image.
            torch.Tensor: Transformed label.
        """
        image = self.images[idx]
        label = self.labels[idx]

        # Apply training or validation transformations
        if self.augmentations:
            image, label = self.augmentations.apply_train_transforms(image, label)

        return image, label
