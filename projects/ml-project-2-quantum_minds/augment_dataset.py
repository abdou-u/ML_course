import os
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Constants
ROTATIONS_PNG = [0, 45, 90, 135, 180, 225, 270, 315]
FLIPS_PNG = [None, np.flipud, np.fliplr]

ROTATIONS_BMP = [0, 90, 180, 270]
FLIPS_BMP = [None, np.flipud]

TRIM_PIXELS = 8  # Number of pixels to trim from each side
BMP_RESIZE = (384, 384)

# Paths
TRAIN_DIR = "data\\merged_dataset\\train"
VALIDATION_DIR = "data\\merged_dataset\\validation"
AUGMENTED_TRAIN_DIR = "data\\merged_dataset_augmented\\train"
AUGMENTED_VALIDATION_DIR = "data\\merged_dataset_augmented\\validation"

# Helper functions
def create_directories(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def trim_image(image_array, pixels):
    return image_array[pixels:-pixels, pixels:-pixels]

def is_black_image(image):
    return np.all(image == 0)

def augment_image(image, groundtruth, base_name, ext, output_dirs, rotations, flips, skip_trim):
    augmented_images_dir, augmented_groundtruth_dir = output_dirs
    for flip_index, flip in enumerate(flips):
        flipped_image = flip(image) if flip else image
        flipped_groundtruth = flip(groundtruth) if flip else groundtruth

        for rotation in rotations:
            rotated_image = rotate(flipped_image, angle=rotation, reshape=False, mode='reflect')
            rotated_groundtruth = rotate(flipped_groundtruth, angle=rotation, reshape=False, mode='reflect')

            aug_image_name = f"{base_name}_flip{flip_index}_rot{rotation}{ext}"
            aug_groundtruth_name = f"{base_name}_flip{flip_index}_rot{rotation}{ext}"

            save_image(rotated_image, os.path.join(augmented_images_dir, aug_image_name), skip_trim=skip_trim)
            save_image(rotated_groundtruth, os.path.join(augmented_groundtruth_dir, aug_groundtruth_name), mode='L', skip_trim=skip_trim)

def save_image(image_array, path, mode=None, skip_trim=False):
    """
    Saves the image after optionally trimming it.
    
    :param image_array: Image array to save
    :param path: Path where the image will be saved
    :param mode: Mode for PIL conversion (e.g., 'L')
    :param skip_trim: Boolean to skip trimming for certain images
    """
    if not skip_trim:
        image_array = trim_image(image_array, TRIM_PIXELS)
    image = Image.fromarray(image_array.astype(np.uint8))
    if mode:
        image = image.convert(mode)
    image.save(path)

def process_images(input_dirs, output_dirs, is_validation=False):
    images_dir, groundtruth_dir = input_dirs
    augmented_images_dir, augmented_groundtruth_dir = output_dirs

    file_list = [f for f in os.listdir(images_dir) if f.endswith((".bmp", ".png"))]

    def process_file(filename):
        image_path = os.path.join(images_dir, filename)
        groundtruth_path = os.path.join(groundtruth_dir, filename)

        image = np.array(Image.open(image_path))
        groundtruth = np.array(Image.open(groundtruth_path))

        if is_black_image(groundtruth):
            return

        base_name, ext = os.path.splitext(filename)

        # Determine file type
        if filename.endswith(".bmp"):
            # Resize BMP images
            image = np.array(Image.fromarray(image).resize(BMP_RESIZE, Image.Resampling.LANCZOS))
            groundtruth = np.array(Image.fromarray(groundtruth).resize(BMP_RESIZE, Image.Resampling.LANCZOS))

            rotations = ROTATIONS_BMP
            flips = FLIPS_BMP
            skip_trim = True  # No trimming for BMP files
        else:
            rotations = ROTATIONS_PNG
            flips = FLIPS_PNG
            skip_trim = False  # Trim for other formats

        if is_validation:
            save_image(image, os.path.join(augmented_images_dir, filename), skip_trim=skip_trim)
            save_image(groundtruth, os.path.join(augmented_groundtruth_dir, filename), mode='L', skip_trim=skip_trim)
        else:
            augment_image(image, groundtruth, base_name, ext, output_dirs, rotations, flips, skip_trim)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, file_list), total=len(file_list), desc=f"Processing {'validation' if is_validation else 'train'} images"))


def main():
    train_images_dir = os.path.join(TRAIN_DIR, "images")
    train_groundtruth_dir = os.path.join(TRAIN_DIR, "groundtruth")
    augmented_train_images_dir = os.path.join(AUGMENTED_TRAIN_DIR, "images")
    augmented_train_groundtruth_dir = os.path.join(AUGMENTED_TRAIN_DIR, "groundtruth")

    validation_images_dir = os.path.join(VALIDATION_DIR, "images")
    validation_groundtruth_dir = os.path.join(VALIDATION_DIR, "groundtruth")
    augmented_validation_images_dir = os.path.join(AUGMENTED_VALIDATION_DIR, "images")
    augmented_validation_groundtruth_dir = os.path.join(AUGMENTED_VALIDATION_DIR, "groundtruth")

    create_directories(augmented_train_images_dir, augmented_train_groundtruth_dir, augmented_validation_images_dir, augmented_validation_groundtruth_dir)

    process_images(
        input_dirs=(train_images_dir, train_groundtruth_dir),
        output_dirs=(augmented_train_images_dir, augmented_train_groundtruth_dir),
        is_validation=False
    )

    process_images(
        input_dirs=(validation_images_dir, validation_groundtruth_dir),
        output_dirs=(augmented_validation_images_dir, augmented_validation_groundtruth_dir),
        is_validation=True
    )

if __name__ == "__main__":
    main()
