import cv2
import os
import numpy as np
import random

def zoom_and_resize(image_path, labels_path, output_image_path, output_labels_path, zoom_factor):
    """
    Zooms into a random zone of paired images and resizes them back to 608x608 pixels.
    
    Args:
        image_path (str): Path to the input image.
        labels_path (str): Path to the corresponding label image.
        output_image_path (str): Path to save the zoomed and resized image.
        output_labels_path (str): Path to save the zoomed and resized label.
        zoom_factor (float): Factor to zoom into the image (>1 zooms in).
    """
    # Load images
    image = cv2.imread(image_path)
    labels = cv2.imread(labels_path)
    if image is None or labels is None:
        raise FileNotFoundError("Image or labels file not found.")

    h, w = image.shape[:2]
    crop_size = int(400 / zoom_factor)

    # Randomly select the top-left corner for the zoom area
    crop_x = random.randint(0, w - crop_size)
    crop_y = random.randint(0, h - crop_size)

    # Crop the region of interest (ROI)
    image_crop = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    labels_crop = labels[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    # Resize the cropped images back to 608x608
    image_resized = cv2.resize(image_crop, (400, 400), interpolation=cv2.INTER_LINEAR)
    labels_resized = cv2.resize(labels_crop, (400, 400), interpolation=cv2.INTER_NEAREST)

    # Save the results
    cv2.imwrite(output_image_path, image_resized)
    cv2.imwrite(output_labels_path, labels_resized)


def process_image_pairs(input_folder, output_folder, zoom_factor):
    """
    Processes all paired images (imageX.png and labelsX.png) in the input folder.
    
    Args:
        input_folder (str): Folder containing the images and labels.
        output_folder (str): Folder to save the processed images.
        zoom_factor (float): Factor to zoom into the image (>1 zooms in).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for X in range(101, 558):
        image_path = os.path.join(input_folder, f"image{X}.png")
        labels_path = os.path.join(input_folder, f"labels{X}.png")
        output_image_path = os.path.join(output_folder, f"zoomed_image{X}.png")
        output_labels_path = os.path.join(output_folder, f"zoomed_labels{X}.png")

        try:
            zoom_and_resize(image_path, labels_path, output_image_path, output_labels_path, zoom_factor)
            print(f"Processed pair: image{X}.png and labels{X}.png")
        except Exception as e:
            print(f"Error processing image{X}.png: {e}")

if __name__ == "__main__":
    # Tunable parameters
    directory = os.getcwd()
    input_folder = os.path.join(directory, "test")  # Folder containing imageX.png and labelsX.png
    output_folder = os.path.join(directory, "newdata")  # Folder to save the processed images
    zoom_factor = 2.5  # Zoom factor (example: 2x zoom)
    # Process all image pairs
    process_image_pairs(input_folder, output_folder, zoom_factor)
