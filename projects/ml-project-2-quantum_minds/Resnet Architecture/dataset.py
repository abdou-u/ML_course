import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import tensor_image_to_pil_image, tensor_mask_to_pil_image
from torch.utils.data import Dataset, DataLoader


import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class MRDDataset(Dataset):
    """
    A PyTorch Dataset class for loading satellite images and their corresponding 
    ground truth segmentation masks for road segmentation tasks.
    """

    def __init__(self, image_dir: str, label_dir: str, images_wh: tuple, transforms=None):
        """
        Args:
            image_dir (str): Directory containing input satellite images.
            label_dir (str): Directory containing ground truth segmentation masks.
            images_wh (tuple): Expected width and height of the images (should match dataset dimensions).
            transforms (callable, optional): Transformations to apply to images and masks.
        """
        super(MRDDataset, self).__init__()

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images_width, self.images_height = images_wh
        self.transforms = transforms

        # Load and validate image and mask filenames
        self.images_names = self._read_sorted_file_names(self.image_dir)
        self.gt_masks_names = self._read_sorted_file_names(self.label_dir)
        
        assert len(self.images_names) == len(self.gt_masks_names), (
            f"Mismatch between number of images ({len(self.images_names)}) and masks ({len(self.gt_masks_names)})."
        )

    def _read_sorted_file_names(self, directory_path):
        """
        Reads and returns sorted filenames in the given directory.
        
        Args:
            directory_path (str): Path to the directory.
        
        Returns:
            list: Sorted list of filenames.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        files_names = [
            f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        files_names.sort()
        return files_names

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Retrieves the image and corresponding mask at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformed image tensor
                                   and the transformed mask tensor.
        """
        # Retrieve paths for the image and mask
        img_path = os.path.join(self.image_dir, self.images_names[index])
        mask_path = os.path.join(self.label_dir, self.gt_masks_names[index])

        # Load the image and mask
        image_i = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        mask_i = Image.open(mask_path)  # Mask can be single-channel or multi-channel

        # Validate dimensions of the image and mask
        assert image_i.size == (self.images_width, self.images_height), (
            f"Unexpected size for image: {img_path}, found: {image_i.size}, expected: {(self.images_width, self.images_height)}."
        )
        assert mask_i.size == (self.images_width, self.images_height), (
            f"Unexpected size for mask: {mask_path}, found: {mask_i.size}, expected: {(self.images_width, self.images_height)}."
        )

        # Apply transformations if provided
        if self.transforms:
            image_i, mask_i = self.transforms(image_i, mask_i)

        # Convert mask to a tensor with normalized values [0, 1]
        mask_array = np.array(mask_i, dtype=np.float32)
        if mask_array.max() > 1:  # Normalize only if the max value is greater than 1
            mask_array = mask_array / 255.0  # Normalize to range [0, 1]
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32)

        # Convert image to tensor
        image_tensor = transforms.ToTensor()(image_i)

        return image_tensor, mask_tensor
        
import torch

class JointTransform:
    """
    A utility class to apply synchronized transformations to both images and masks 
    in a segmentation task. This ensures consistent augmentation and preprocessing.

    Args:
        joint_transform (callable, optional): Transformations applied jointly to both the 
                                              image and mask (e.g., rotation, flipping).
        image_transform (callable, optional): Transformations applied only to the image 
                                              (e.g., normalization).
    """

    def __init__(self, joint_transform=None, image_transform=None):
        self.joint_transform = joint_transform
        self.image_transform = image_transform

    def __call__(self, img, mask):
        """
        Applies the specified transformations to the image and mask.

        Args:
            img (PIL.Image.Image): The input image.
            mask (PIL.Image.Image): The corresponding ground truth mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The transformed image and mask as tensors.
        """
        # Apply joint transformations
        if self.joint_transform is not None:
            # Use a random seed for synchronized transformations
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            img = self.joint_transform(img)

            if mask is not None:
                torch.manual_seed(seed)  # Reuse the same seed for the mask
                mask = self.joint_transform(mask)

        # Apply image-specific transformations
        if self.image_transform is not None:
            img = self.image_transform(img)

        return img, mask


if __name__ == '__main__':
    
    # Define the path to the JSON configuration file
    config_file_path = 'config/config.json'

    # Open and read the JSON file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Define the joint transformations for both image and mask
    joint_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=config['test_patch_size'], scale=(0.75, 1), interpolation=transforms.InterpolationMode.NEAREST), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    joint_transform_test = transforms.Compose([transforms.ToTensor()])
    
    # Define the image-specific transformations
    image_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transformations = JointTransform(joint_transform=joint_transform_train, image_transform=image_transform)
    test_transformations = JointTransform(joint_transform=joint_transform_test, image_transform=image_transform)

    # train dataset
    set_i = 'train'
    train_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=train_transformations)
    
    img_i, mask_i = train_ds[38]
    pil_img_i = tensor_image_to_pil_image(img_i.clone())
    pil_mask_i = tensor_mask_to_pil_image(mask_i.clone())
    plt.figure()
    plt.imshow(pil_img_i)
    plt.figure()
    plt.imshow(pil_mask_i)
    plt.show()

    # Train dataloader
    dataloader_train = DataLoader(dataset=train_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_train)))

    set_i = 'test'
    test_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=test_transformations)
    
    img_i, mask_i = test_ds[28]
    pil_img_i = tensor_image_to_pil_image(img_i.clone())
    pil_mask_i = tensor_mask_to_pil_image(mask_i.clone())
    plt.figure()
    plt.imshow(pil_img_i)
    plt.figure()
    plt.imshow(pil_mask_i)
    plt.show()

    # Test dataloader
    dataloader_test = DataLoader(dataset=test_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_test)))