"""
Utility functions for the HOG Computer Vision Homework project - Problem 1.

This module provides helper functions for:
- Image loading and I/O
- Visualization
"""

import cv2
import numpy as np


def load_grayscale_image(path: str) -> np.ndarray:
    """
    Load a single image as grayscale.

    This function reads an image file from the given path and converts it
    to grayscale format.

    Parameters
    ----------
    path : str
        Path to the image file (e.g., "data/test_images/example.jpg").

    Returns
    -------
    image : np.ndarray
        Grayscale image as a 2D NumPy array (dtype: uint8, shape: (height, width)).

    Notes
    -----
    Step-by-step implementation:
    1. Use cv2.imread() to read the image file
    2. Convert the image to grayscale using cv2.IMREAD_GRAYSCALE flag
    3. Check if the image was loaded successfully (not None)
    4. Return the grayscale image array
    """
    # !!! Burak: set image path when calling this later
    
    # Step 1: Read the image file using cv2.imread()
    # Use cv2.IMREAD_GRAYSCALE flag to load directly as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Step 2: Check if the image was loaded successfully
    # If img is None, raise an error with a helpful message
    if img is None:
        raise ValueError(f"Could not load image from: {path}")
    
    # Step 3: Return the grayscale image array
    return img


def show_side_by_side(
    title: str,
    left_image: np.ndarray,
    right_image: np.ndarray,
    output_dir: str = "data/results/"
) -> None:
    """
    INSTEAD OF SHOWING IN A WINDOW (WHICH CAUSES BLACK SCREEN),
    THIS FUNCTION SAVES THE SIDE-BY-SIDE RESULT AS A PNG FILE.
    """
    import os
    
    # Ensure results directory exists
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert grayscale to BGR if needed
    if len(left_image.shape) == 2:
        left_image_bgr = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
    else:
        left_image_bgr = left_image
    
    if len(right_image.shape) == 2:
        right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
    else:
        right_image_bgr = right_image
    
    # Resize both to same height
    h = max(left_image_bgr.shape[0], right_image_bgr.shape[0])
    left_resized = cv2.resize(
        left_image_bgr, 
        (int(left_image_bgr.shape[1] * (h / left_image_bgr.shape[0])), h)
    )
    right_resized = cv2.resize(
        right_image_bgr, 
        (int(right_image_bgr.shape[1] * (h / right_image_bgr.shape[0])), h)
    )
    
    # Combine images horizontally
    combined = np.hstack((left_resized, right_resized))
    
    # Save the output
    filename = title.replace(" ", "_").replace(":", "") + ".png"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, combined)
    
    print(f"[SAVED] {save_path}")


def load_images_from_folder(folder_path: str) -> list:
    """
    Load all images from a folder.
    
    This function scans a folder for image files and loads them.
    Returns a list of tuples: (image_path, image_array).
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing images.
    
    Returns
    -------
    images : list of tuples
        List of (image_path, image_array) tuples.
        image_path: str, full path to the image file
        image_array: np.ndarray, loaded image (BGR format from OpenCV)
    
    Notes
    -----
    Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """
    import os
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []
    
    if not os.path.exists(folder_path):
        return images
    
    # Get all files in folder
    for filename in os.listdir(folder_path):
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext in valid_extensions:
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                images.append((image_path, img))
    
    return images
