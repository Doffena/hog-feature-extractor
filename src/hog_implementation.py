"""
Problem 1: Manual HOG (Histogram of Oriented Gradients) Feature Extraction.

This module contains a skeleton implementation of HOG feature extraction.
The core computation functions are intentionally left as stubs with detailed
step-by-step comments. You must implement the actual HOG algorithm yourself.

IMPORTANT: Do NOT implement the core math here. Only add detailed comments
and raise NotImplementedError at the end of each function.
"""

from typing import Tuple
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Handle both relative and absolute imports
try:
    from .utils import load_grayscale_image, show_side_by_side
except ImportError:
    # Add src directory to path if running as script
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from utils import load_grayscale_image, show_side_by_side


def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute horizontal and vertical gradients of the input image.

    This function calculates the gradient magnitude and orientation at each
    pixel of the input grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Single-channel (grayscale) input image represented as a 2D NumPy array
        of dtype uint8.

    Returns
    -------
    gradient_magnitude : np.ndarray
        Array of the same spatial shape as ``image`` containing the gradient
        magnitude at each pixel.
    gradient_angle : np.ndarray
        Array of the same spatial shape as ``image`` containing the gradient
        orientation in degrees (range: 0 to 180 for unsigned gradients).

    Notes
    -----
    Step-by-step implementation guide:

    1. Convert image to float32 to avoid overflow during gradient computation
       - Use image.astype(np.float32) or np.float32(image)

    2. Compute gradients in x-direction (horizontal):
       - Option A: Use np.gradient(image, axis=1) for the x-direction
       - Option B: Use cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
       - Option C: Convolve with kernel [-1, 0, 1] using cv2.filter2D()

    3. Compute gradients in y-direction (vertical):
       - Option A: Use np.gradient(image, axis=0) for the y-direction
       - Option B: Use cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
       - Option C: Convolve with kernel [-1, 0, 1].T using cv2.filter2D()

    4. Compute gradient magnitude:
       - Use: magnitude = sqrt(Gx^2 + Gy^2)
       - Hint: np.sqrt(gx**2 + gy**2)

    5. Compute gradient angle in degrees:
       - Use: angle = arctan2(Gy, Gx) * 180 / pi
       - Hint: np.arctan2(gy, gx) * 180 / np.pi

    6. Map angles to [0, 180) range for unsigned gradients:
       - If angle < 0, add 180 to it
       - If angle >= 180, subtract 180 from it
       - This ensures all angles are in the [0, 180) range
    """
    # Step 1: Convert image to float32 to avoid overflow
    image = image.astype(np.float32)
    
    # Step 2: Compute gradients in x-direction (horizontal)
    # Create kernel for horizontal gradient: [-1, 0, 1]
    kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
    # Convolve image with kernel_x to get horizontal gradients
    Gx = cv2.filter2D(image, -1, kernel_x)
    
    # Step 3: Compute gradients in y-direction (vertical)
    # Create kernel for vertical gradient: [-1, 0, 1] transposed
    kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    # Convolve image with kernel_y to get vertical gradients
    Gy = cv2.filter2D(image, -1, kernel_y)
    
    # Step 4: Compute gradient magnitude
    # Magnitude = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Step 5: Compute gradient angle in degrees
    # Angle = arctan2(Gy, Gx) converted to degrees
    angle = np.rad2deg(np.arctan2(Gy, Gx))
    
    # Step 6: Convert angles to range 0-180 for unsigned gradients
    # Use modulo operation to map angles to [0, 180) range
    angle = angle % 180
    
    # Return both magnitude and angle arrays
    return magnitude, angle


def create_cell_histogram(
    cell_magnitude: np.ndarray,
    cell_angle: np.ndarray,
    num_bins: int = 9
) -> np.ndarray:
    """
    Build the orientation histogram for a single cell.

    This function creates a histogram of gradient orientations for a single
    cell by accumulating gradient magnitudes into orientation bins.

    Parameters
    ----------
    cell_magnitude : np.ndarray
        2D array containing gradient magnitudes for a single cell (e.g., 8x8 pixels).
    cell_angle : np.ndarray
        2D array containing gradient angles in degrees (range: 0 to 180) for the same cell.
    num_bins : int, optional
        Number of orientation bins. Default is 9.

    Returns
    -------
    histogram : np.ndarray
        1D NumPy array of length ``num_bins`` containing the orientation histogram.

    Notes
    -----
    Step-by-step implementation guide:

    1. Define bin edges between 0 and 180 degrees:
       - Create num_bins bins, each covering 180/num_bins degrees
       - Example for 9 bins: bin_width = 180 / 9 = 20 degrees
       - Bin edges: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
       - Bin centers: [10, 30, 50, 70, 90, 110, 130, 150, 170]

    2. Initialize histogram array:
       - Create a 1D array of zeros with length num_bins
       - Example: histogram = np.zeros(num_bins)

    3. For each pixel in the cell:
       a. Get its gradient magnitude and angle
       b. Find which bin(s) the angle belongs to
       c. Optionally use bilinear interpolation:
          - If angle falls between two bins, split the magnitude vote
          - Weight by distance to each bin center
          - This provides smoother histograms

    4. Simple approach (without interpolation):
       - Calculate bin index: bin_idx = int(angle / (180 / num_bins))
       - Make sure bin_idx is in valid range [0, num_bins-1]
       - Add the pixel's magnitude to that bin

    5. Return the histogram array
    """
    # Step 1: Create histogram array initialized with zeros
    histogram = np.zeros(num_bins, dtype=np.float32)
    
    # Step 2: Compute bin width
    # Each bin covers 180/num_bins degrees (e.g., 20 degrees for 9 bins)
    bin_width = 180 / num_bins
    
    # Step 3: Loop over every pixel in the cell
    for i in range(cell_angle.shape[0]):
        for j in range(cell_angle.shape[1]):
            # Step 4: For each pixel, get its angle and magnitude
            angle = cell_angle[i, j]
            magnitude = cell_magnitude[i, j]
            
            # Step 5: Compute bin index
            # Use integer division to find which bin this angle belongs to
            bin_index = int(angle // bin_width)
            
            # Step 6: Clamp bin index to valid range [0, num_bins-1]
            bin_index = min(num_bins - 1, max(0, bin_index))
            
            # Step 7: Add magnitude to that bin
            histogram[bin_index] += magnitude
    
    # Step 8: Return the histogram
    return histogram


def normalize_block(
    block_histogram: np.ndarray,
    method: str = "L2",
    eps: float = 1e-5
) -> np.ndarray:
    """
    Normalize a block histogram using the specified normalization method.

    This function normalizes a concatenated block histogram to make it
    invariant to lighting variations.

    Parameters
    ----------
    block_histogram : np.ndarray
        1D array containing concatenated cell histograms for a block.
    method : str, optional
        Normalization method. Options: "L2", "L1", "L2-Hys". Default is "L2".
    eps : float, optional
        Small epsilon value to avoid division by zero. Default is 1e-5.

    Returns
    -------
    normalized : np.ndarray
        Normalized block histogram.

    Notes
    -----
    Normalization methods:

    L2 normalization:
        normalized = h / sqrt(||h||^2 + eps^2)
        where ||h||^2 is the L2 norm squared (sum of squares)
        Steps:
        1. Calculate L2 norm squared: norm_squared = np.sum(h**2)
        2. Calculate denominator: denom = np.sqrt(norm_squared + eps**2)
        3. Normalize: normalized = h / denom

    L1 normalization:
        normalized = h / (||h||_1 + eps)
        where ||h||_1 is the L1 norm (sum of absolute values)
        Steps:
        1. Calculate L1 norm: norm_l1 = np.sum(np.abs(h))
        2. Calculate denominator: denom = norm_l1 + eps
        3. Normalize: normalized = h / denom

    L2-Hys (L2 with clipping and renormalization):
        1. Apply L2 normalization first
        2. Clip values to a maximum (e.g., 0.2): clipped = np.clip(normalized, 0, 0.2)
        3. Renormalize using L2 again
    """
    # Step 1: Convert the input histogram to float32
    h = block_histogram.astype(np.float32)
    
    # Step 2: If method == "L2"
    if method == "L2":
        # Compute L2 norm squared
        norm_sq = np.sum(h ** 2)
        # Compute denominator
        denom = np.sqrt(norm_sq + eps ** 2)
        # If denom == 0, just return h (to avoid division by zero issues)
        if denom == 0:
            return h
        # Else return h / denom
        else:
            return h / denom
    
    # Step 3: If method == "L1"
    elif method == "L1":
        # Compute L1 norm
        norm_l1 = np.sum(np.abs(h))
        # Compute denominator
        denom = norm_l1 + eps
        # If denom == 0, just return h
        if denom == 0:
            return h
        # Else return h / denom
        else:
            return h / denom
    
    # Step 4: If method == "L2-Hys"
    elif method == "L2-Hys":
        # First do L2 normalization as in step 2 to get h_l2
        norm_sq = np.sum(h ** 2)
        denom = np.sqrt(norm_sq + eps ** 2)
        if denom == 0:
            h_l2 = h
        else:
            h_l2 = h / denom
        # Clip values
        h_clipped = np.clip(h_l2, 0, 0.2)
        # Recompute L2 norm squared on h_clipped
        norm_sq = np.sum(h_clipped ** 2)
        # Compute denominator
        denom = np.sqrt(norm_sq + eps ** 2)
        # If denom == 0, return h_clipped
        if denom == 0:
            return h_clipped
        # Else return h_clipped / denom
        else:
            return h_clipped / denom
    
    # Step 5: For any other method value
    else:
        # Just return h without changes
        return h


def compute_hog_descriptor(
    image: np.ndarray,
    cell_size: Tuple[int, int] = (8, 8),
    block_size: Tuple[int, int] = (2, 2),
    num_bins: int = 9
) -> np.ndarray:
    """
    Compute the full HOG descriptor for an image.

    This is the main function that computes the complete HOG feature vector
    for an input image by processing cells and blocks.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (2D array).
    cell_size : tuple of int, optional
        Size of each cell in pixels as (height, width). Default is (8, 8).
    block_size : tuple of int, optional
        Size of each block in number of cells as (height_in_cells, width_in_cells).
        Default is (2, 2), meaning a block contains 2x2 = 4 cells.
    num_bins : int, optional
        Number of orientation bins. Default is 9.

    Returns
    -------
    descriptor : np.ndarray
        1D NumPy array representing the complete HOG feature vector.

    Notes
    -----
    Step-by-step implementation guide:

    1. Compute gradients:
       - Call compute_gradients(image) to get gradient_magnitude and gradient_angle

    2. Split image into cells:
       - Divide the image into non-overlapping cells of size cell_size
       - Calculate how many cells fit in height and width directions
       - Example: If image is 64x128 and cell_size is (8, 8):
         * num_cells_y = 64 / 8 = 8
         * num_cells_x = 128 / 8 = 16

    3. For each cell, compute its histogram:
       - Extract the cell's magnitude and angle patches from the gradient arrays
       - Call create_cell_histogram() to get the cell's orientation histogram
       - Store all cell histograms in a 3D array: (num_cells_y, num_cells_x, num_bins)

    4. Group cells into blocks:
       - Slide a block window over the grid of cells
       - Each block contains block_size[0] x block_size[1] cells
       - Blocks typically overlap (stride = 1 cell)
       - Calculate how many blocks fit: 
         * num_blocks_y = num_cells_y - block_size[0] + 1
         * num_blocks_x = num_cells_x - block_size[1] + 1
       - For each block:
         a. Extract the block's cell histograms
         b. Concatenate the histograms of all cells in the block into a 1D array
         c. Call normalize_block() to normalize the concatenated histogram
         d. Store the normalized block feature vector

    5. Flatten all normalized block histograms:
       - Concatenate all block feature vectors into a single 1D array
       - This is your final HOG descriptor

    Example dimensions (for reference):
    - Image: 64x128 pixels
    - Cell size: 8x8
    - Number of cells: 8x16 = 128 cells
    - Block size: 2x2 cells
    - Blocks per dimension: (8-2+1) x (16-2+1) = 7x15 = 105 blocks
    - Features per block: 2x2x9 = 36 features
    - Total descriptor length: 105 x 36 = 3780 features
    """
    # Step 1: Compute gradients
    magnitude, angle = compute_gradients(image)
    
    # Step 2: Compute number of cells
    cell_h, cell_w = cell_size
    img_h, img_w = image.shape
    num_cells_y = img_h // cell_h
    num_cells_x = img_w // cell_w
    
    # Step 3: Create cell_histograms = zeros((num_cells_y, num_cells_x, num_bins))
    cell_histograms = np.zeros((num_cells_y, num_cells_x, num_bins), dtype=np.float32)
    
    # Step 4: For each cell
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # extract cell_mag & cell_angle using slicing
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_ang = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # compute histogram = create_cell_histogram(...)
            histogram = create_cell_histogram(cell_mag, cell_ang, num_bins)
            # store in cell_histograms
            cell_histograms[i, j] = histogram
    
    # Step 5: Compute number of blocks
    block_h, block_w = block_size
    num_blocks_y = num_cells_y - block_h + 1
    num_blocks_x = num_cells_x - block_w + 1
    
    # Step 6: For each block
    block_features = []
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            # extract block_cells = cell_histograms[i:i+block_h, j:j+block_w, :]
            block_cells = cell_histograms[i:i+block_h, j:j+block_w, :]
            # flatten → block_vector
            block_vector = block_cells.flatten()
            # normalize → normalized_block = normalize_block(block_vector)
            normalized_block = normalize_block(block_vector)
            # append to block_features list
            block_features.append(normalized_block)
    
    # Step 7: descriptor = concatenate(block_features)
    descriptor = np.concatenate(block_features)
    
    # Step 8: return descriptor
    return descriptor


def visualize_hog(
    image: np.ndarray,
    cell_size: Tuple[int, int] = (8, 8),
    num_bins: int = 9,
    scale_factor: float = 1.0,
    output_dir: str = "data/results/hog_vis/"
) -> np.ndarray:
    """
    Visualize HOG features by drawing orientation lines on the image.

    This function creates a visualization of HOG features by drawing lines
    representing the dominant gradient orientations in each cell.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    cell_size : tuple of int, optional
        Size of each cell in pixels. Default is (8, 8).
    num_bins : int, optional
        Number of orientation bins. Default is 9.
    scale_factor : float, optional
        Scaling factor for line lengths. Default is 1.0.

    Returns
    -------
    vis_image : np.ndarray
        Visualization image with HOG features overlaid (can be color or grayscale).

    Notes
    -----
    Step-by-step implementation guide:

    1. Compute HOG descriptor and per-cell histograms:
       - You need to compute histograms for each cell (similar to compute_hog_descriptor)
       - But keep the per-cell histograms instead of just the final descriptor
       - Steps: compute gradients, split into cells, compute cell histograms

    2. Create visualization canvas:
       - Option A: Create a blank canvas (same size as image, or larger)
       - Option B: Convert original image to color and use as background
       - Example: vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    3. For each cell:
       a. Get the cell's center coordinates:
          - center_x = (j + 0.5) * cell_w
          - center_y = (i + 0.5) * cell_h
       b. For each orientation bin:
          - Calculate the bin's orientation angle (center of the bin)
          - Get the bin's magnitude value from the histogram
          - Draw a line from the cell center:
            * Direction: orientation angle
            * Length: proportional to the bin's magnitude (scaled by scale_factor)
            * Calculate line endpoints using trigonometry:
              end_x = center_x + length * cos(angle)
              end_y = center_y + length * sin(angle)
            * Use cv2.line() to draw the line

    4. Drawing approach:
       - Use cv2.line(vis_image, (start_x, start_y), (end_x, end_y), color, thickness)
       - You can use different colors for different bins or use a single color
       - Line thickness can be 1 or 2 pixels

    5. Return the visualization image
    """
    # Step 1: Compute gradients
    magnitude, angle = compute_gradients(image)
    
    # Step 2: Compute image and cell dimensions
    img_h, img_w = image.shape
    cell_h, cell_w = cell_size
    
    # Step 3: Compute number of cells
    num_cells_y = img_h // cell_h
    num_cells_x = img_w // cell_w
    
    # Step 4: Allocate per-cell histograms
    cell_histograms = np.zeros((num_cells_y, num_cells_x, num_bins), dtype=np.float32)
    
    # Step 5: Fill per-cell histograms using create_cell_histogram
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_ang = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist = create_cell_histogram(cell_mag, cell_ang, num_bins)
            cell_histograms[i, j] = hist
    
    # Step 6: Create visualization canvas
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Step 7: Compute bin width
    bin_width = 180 / num_bins
    
    # Step 8: For each cell and each bin, draw a line using cv2.line
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            center_x = int((j + 0.5) * cell_w)
            center_y = int((i + 0.5) * cell_h)
            hist = cell_histograms[i, j]
            
            for k in range(num_bins):
                bin_angle = (k + 0.5) * bin_width
                bin_mag = hist[k]
                
                # scale length
                length = bin_mag * scale_factor
                
                # convert angle to radians
                rad = np.deg2rad(bin_angle)
                
                # compute end point (y uses minus because image origin is top-left)
                end_x = int(center_x + length * np.cos(rad))
                end_y = int(center_y - length * np.sin(rad))
                
                # draw white line
                cv2.line(vis, (center_x, center_y), (end_x, end_y), (255, 255, 255), 1)
    
    # Step 9: Save visualization image and return
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"hog_vis_{cell_size[0]}_{num_bins}.png")
    cv2.imwrite(save_path, vis)
    print(f"[SAVED] {save_path}")
    
    return vis


def save_gradient_outputs(image_path: str, output_dir: str = None):
    """
    Save Gx, Gy, magnitude, and orientation outputs for a given image.
    
    This function computes gradients and saves all intermediate outputs
    as separate image files for visualization and analysis.
    
    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_dir : str, optional
        Output directory. If None, uses data/results/{image_name}/gradients/
    """
    import os
    
    # Load image
    try:
        image = load_grayscale_image(image_path)
    except Exception as e:
        print(f"[ERROR] Could not load image: {e}")
        return
    
    # Compute gradients
    magnitude, angle = compute_gradients(image)
    
    # Compute Gx and Gy separately for visualization
    image_float = image.astype(np.float32)
    kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    Gx = cv2.filter2D(image_float, -1, kernel_x)
    Gy = cv2.filter2D(image_float, -1, kernel_y)
    
    # Normalize for visualization (0-255 range)
    def normalize_for_display(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min > 0:
            normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)
        return normalized
    
    Gx_display = normalize_for_display(Gx)
    Gy_display = normalize_for_display(Gy)
    magnitude_display = normalize_for_display(magnitude)
    # Orientation: map 0-180 degrees to 0-255
    orientation_display = ((angle / 180.0) * 255).astype(np.uint8)
    
    # Create output directory
    if output_dir is None:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join("data", "results", img_name, "gradients")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), image)
    print(f"[SAVED] {os.path.join(output_dir, f'{base_name}_original.png')}")
    
    # Save Gx (horizontal gradient)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_Gx.png"), Gx_display)
    print(f"[SAVED] {os.path.join(output_dir, f'{base_name}_Gx.png')}")
    
    # Save Gy (vertical gradient)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_Gy.png"), Gy_display)
    print(f"[SAVED] {os.path.join(output_dir, f'{base_name}_Gy.png')}")
    
    # Save magnitude
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_magnitude.png"), magnitude_display)
    print(f"[SAVED] {os.path.join(output_dir, f'{base_name}_magnitude.png')}")
    
    # Save orientation
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_orientation.png"), orientation_display)
    print(f"[SAVED] {os.path.join(output_dir, f'{base_name}_orientation.png')}")
    
    # Create combined visualization (all outputs in one image)
    # Resize all images to same size for grid layout
    target_size = (400, 400)  # (width, height)
    
    original_resized = cv2.resize(image, target_size)
    Gx_resized = cv2.resize(Gx_display, target_size)
    Gy_resized = cv2.resize(Gy_display, target_size)
    magnitude_resized = cv2.resize(magnitude_display, target_size)
    orientation_resized = cv2.resize(orientation_display, target_size)
    
    # Convert grayscale to BGR for text overlay
    original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
    Gx_bgr = cv2.cvtColor(Gx_resized, cv2.COLOR_GRAY2BGR)
    Gy_bgr = cv2.cvtColor(Gy_resized, cv2.COLOR_GRAY2BGR)
    magnitude_bgr = cv2.cvtColor(magnitude_resized, cv2.COLOR_GRAY2BGR)
    orientation_bgr = cv2.cvtColor(orientation_resized, cv2.COLOR_GRAY2BGR)
    
    # Add labels to each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    
    cv2.putText(original_bgr, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(Gx_bgr, "Gx (Horizontal)", (10, 30), font, font_scale, color, thickness)
    cv2.putText(Gy_bgr, "Gy (Vertical)", (10, 30), font, font_scale, color, thickness)
    cv2.putText(magnitude_bgr, "Magnitude", (10, 30), font, font_scale, color, thickness)
    cv2.putText(orientation_bgr, "Orientation", (10, 30), font, font_scale, color, thickness)
    
    # Create 2x3 grid (2 rows, 3 columns)
    # Row 1: Original, Gx, Gy
    row1 = np.hstack([original_bgr, Gx_bgr, Gy_bgr])
    # Row 2: Magnitude, Orientation, (empty space or repeat original)
    row2 = np.hstack([magnitude_bgr, orientation_bgr, np.zeros_like(original_bgr)])
    
    # Combine rows
    combined = np.vstack([row1, row2])
    
    # Save combined visualization
    combined_path = os.path.join(output_dir, f"{base_name}_gradients_combined.png")
    cv2.imwrite(combined_path, combined)
    print(f"[SAVED] {combined_path}")
    
    # Alternative: 2x3 grid (all 5 outputs in one image)
    # Row 1: Original, Gx, Gy
    row1_alt = np.hstack([original_bgr, Gx_bgr, Gy_bgr])
    # Row 2: Magnitude, Orientation, (empty space for balance)
    empty_space = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    cv2.putText(empty_space, "Empty", (10, 30), font, font_scale, color, thickness)
    row2_alt = np.hstack([magnitude_bgr, orientation_bgr, empty_space])
    
    # Combine rows (both rows have same width: 3 * 400 = 1200)
    combined_alt = np.vstack([row1_alt, row2_alt])
    combined_alt_path = os.path.join(output_dir, f"{base_name}_gradients_all.png")
    cv2.imwrite(combined_alt_path, combined_alt)
    print(f"[SAVED] {combined_alt_path}")
    
    print(f"\n✓ Tüm gradient çıktıları kaydedildi: {output_dir}")


def run_problem1_tests():
    """
    Run a series of tests for Problem 1:

    - Use at least 5 test images

    - Show original and HOG visualization

    - Print HOG descriptor length

    - Try different cell_size and num_bins values
    """
    # !!! Burak: put your actual test image filenames here
    test_image_paths = [
        "data/test_images/square.jpeg",      # simple square
        "data/test_images/circle.jpeg",      # simple circle
        "data/test_images/triangle.png",    # simple triangle
        "data/test_images/object.png",      # real object with edges
        "data/test_images/silhouette.jpg",  # human silhouette
    ]
    
    # Display menu for image selection
    print("\n" + "="*50)
    print("HOG Feature Extraction - Image Selection")
    print("="*50)
    print("\nAvailable test images:")
    for i, img_path in enumerate(test_image_paths, 1):
        img_name = os.path.basename(img_path)
        print(f"  {i}. {img_name}")
    print(f"  0. Exit")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect an image (0-{len(test_image_paths)}): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                print("Exiting...")
                return
            elif 1 <= choice_num <= len(test_image_paths):
                # Process selected image
                selected_paths = [test_image_paths[choice_num - 1]]
                break
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(test_image_paths)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Try different HOG parameter settings
    hog_configs = [
        {"cell_size": (8, 8), "num_bins": 9},
        {"cell_size": (8, 8), "num_bins": 6},
        {"cell_size": (16, 16), "num_bins": 9},
    ]
    
    # Process selected image
    img_path = selected_paths[0]
    print(f"\n=== Testing image: {img_path} ===")
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"  [WARNING] Image not found: {img_path}")
        print(f"  Skipping this image...")
        return
    
    # Get image name without extension for folder name
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_base_dir = os.path.join("data", "results", img_name)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"  Output directory: {output_base_dir}/")
    
    # Load image in grayscale
    try:
        image = load_grayscale_image(img_path)
    except Exception as e:
        print(f"  [ERROR] Could not load image: {e}")
        return
    
    # Save gradient outputs (Gx, Gy, magnitude, orientation)
    print(f"\n  -> Gradient çıktıları kaydediliyor (Gx, Gy, magnitude, orientation)...")
    save_gradient_outputs(img_path, output_dir=os.path.join(output_base_dir, "gradients"))
    
    for cfg in hog_configs:
        cell_size = cfg["cell_size"]
        num_bins = cfg["num_bins"]
        print(f"  -> cell_size={cell_size}, num_bins={num_bins}")
        
        # Compute descriptor
        descriptor = compute_hog_descriptor(
            image,
            cell_size=cell_size,
            block_size=(2, 2),
            num_bins=num_bins,
        )
        print(f"     Descriptor length: {descriptor.shape[0]}")
        
        # Visualize HOG (using default scale_factor)
        hog_vis_dir = os.path.join(output_base_dir, "hog_vis")
        vis = visualize_hog(
            image,
            cell_size=cell_size,
            num_bins=num_bins,
            scale_factor=0.05,
            output_dir=hog_vis_dir,
        )
        
        # Show original vs HOG visualization
        show_side_by_side(
            title=f"HOG - cell={cell_size}, bins={num_bins}",
            left_image=image,
            right_image=vis,
            output_dir=output_base_dir,
        )
    
    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)


if __name__ == "__main__":
    """
    Main demo section for testing HOG implementation (Problem 1).
    """
    run_problem1_tests()
