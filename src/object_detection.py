"""
Problem 2: Object Detection using HOG + SVM.

This module implements:
- Part A: Human detection using OpenCV's pretrained HOG + SVM detector
- Part B: Custom object detection using sliding window + HOG + Linear SVM

This is a skeleton implementation. All functions contain step-by-step comments
but raise NotImplementedError() at the end. You must implement the actual logic.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import joblib
from sklearn.svm import LinearSVC  # ONLY import, no training code implemented

# Handle both relative and absolute imports
try:
    from .hog_implementation import compute_hog_descriptor
    from .utils import load_grayscale_image
except ImportError:
    # Add src directory to path if running as script
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from hog_implementation import compute_hog_descriptor
    from utils import load_grayscale_image

# Directory structure comments:
# data/test_images/                 ← test images
# data/results/human_detection/     ← output detection images
# !!! Burak: add human test images to data/test_images/

# data/training_set/positive/       # !!! Burak: put positive samples here
# data/training_set/negative/       # !!! Burak: put negative samples here
# data/results/custom_detection/    # output detection images
# models/trained_classifier.pkl     # saved model path


# ============================================================================
# PART A — HUMAN DETECTION (OpenCV's pretrained HOG + SVM)
# ============================================================================

def setup_pretrained_hog_detector():
    """
    Initialize OpenCV's pretrained HOG + SVM human detector.

    This function sets up the default pedestrian detector that comes with OpenCV.
    The detector is already trained and ready to use.

    Returns
    -------
    hog_detector : cv2.HOGDescriptor
        Initialized HOG descriptor with default people detector.

    Notes
    -----
    Step-by-step implementation:
    1. Create a cv2.HOGDescriptor() object
    2. Set the SVM detector using cv2.HOGDescriptor_getDefaultPeopleDetector()
    3. Return the configured HOG descriptor

    The default detector is trained on the INRIA person dataset and works well
    for pedestrian detection in various scenarios.
    """
    # Step 1: Create HOG descriptor object
    # Use cv2.HOGDescriptor() to create a new HOG descriptor instance
    hog = cv2.HOGDescriptor()
    
    # Step 2: Set the pretrained SVM detector
    # Use hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # This loads the default pedestrian detector weights
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Step 3: Return the configured detector
    # Return the hog_detector object
    return hog


def detect_humans_in_image(image, win_stride=None, padding=None, scale=None):
    """
    Detect humans in a single image using OpenCV's INRIA pedestrian detector (HOG + SVM).

    This function uses optimized parameters for better detection sensitivity,
    reducing false negatives (missing people in images).

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    win_stride : tuple of int, optional
        Window stride in pixels as (x, y). If None, uses optimized (4, 4).
    padding : tuple of int, optional
        Padding in pixels as (x, y). If None, uses optimized (24, 24).
    scale : float, optional
        Scale factor for multi-scale detection. If None, uses optimized 1.03.

    Returns
    -------
    rects : list of tuple
        List of detected bounding boxes, each as (x, y, width, height).
    weights : list of float
        Confidence scores for each detection (true SVM decision function values).

    Notes
    -----
    Uses OpenCV's default people detector with optimized parameters for sensitivity:
    - winStride = (4, 4) - Smaller stride for more thorough search
    - padding = (24, 24) - More padding for better border detection
    - scale = 1.03 - Smaller scale step for finer multi-scale detection
    - hitThreshold = 0.0 - Lower threshold to catch more detections
    
    These parameters maximize detection rate and reduce false negatives.
    """
    # Initialize OpenCV's default people detector
    # This is the academically validated INRIA pedestrian detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Use optimized parameters for better detection sensitivity
    # Smaller win_stride = more thorough search (reduces false negatives)
    # More padding = better border detection
    # Smaller scale = more scale levels (catches different sized people)
    if win_stride is None:
        win_stride = (4, 4)  # Smaller stride for better coverage
    if padding is None:
        padding = (24, 24)  # More padding for border detection
    if scale is None:
        scale = 1.03  # Smaller scale step for finer multi-scale detection
    
    # Detection with optimized parameters for better sensitivity
    # Lower hitThreshold = more detections (reduces false negatives)
    # Smaller win_stride = more thorough search
    # Smaller scale = more scale levels
    # Returns true SVM decision function values (confidence scores)
        rects, weights = hog.detectMultiScale(
        image,
            winStride=win_stride,
            padding=padding,
            scale=scale,
        hitThreshold=0.0  # Lower threshold to catch more detections
    )
    
    # Convert to lists
    # rects is a numpy array of shape (N, 4) where each row is (x, y, w, h)
    rects_list = [tuple(rect) for rect in rects]
    
    # Extract true confidence scores from HOG SVM
    # weights contains the true SVM decision function values
    if len(weights) > 0:
        # Handle different weight formats from OpenCV
        if isinstance(weights, np.ndarray):
            if weights.ndim > 1:
                weights_list = [float(w[0]) for w in weights]
            else:
                weights_list = [float(w) for w in weights]
        else:
            weights_list = [float(w) for w in weights]
    else:
        # If no weights returned, this shouldn't happen with default detector
        # but assign default confidence if it does
        weights_list = [0.5] * len(rects_list)
    
    return rects_list, weights_list


def apply_nms_to_detections(rects, weights, threshold):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping detections.

    When multiple detections overlap significantly, NMS keeps only the one
    with the highest confidence score and suppresses the others.

    Parameters
    ----------
    rects : list of tuple
        List of bounding boxes, each as (x, y, width, height).
    weights : list of float
        Confidence scores for each detection.
    threshold : float
        IoU (Intersection over Union) threshold. Detections with IoU > threshold
        will be suppressed. Typical values: 0.3 to 0.5.

    Returns
    -------
    filtered_rects : list of tuple
        Filtered bounding boxes after NMS.
    filtered_weights : list of float
        Confidence scores for the filtered boxes.

    Notes
    -----
    Step-by-step implementation:
    1. If no detections, return empty lists
    2. Sort detections by confidence score (descending order)
    3. For each detection:
       a. Calculate IoU with all remaining detections
       b. If IoU > threshold, mark the lower-confidence detection for removal
    4. Keep only the detections that were not suppressed
    5. Return filtered rects and weights

    IoU calculation:
    - Intersection area = area of overlap between two boxes
    - Union area = area of box1 + area of box2 - intersection area
    - IoU = intersection_area / union_area
    """
    # Step 1: Check if there are any detections
    # If len(rects) == 0, return empty lists
    if len(rects) == 0:
        return [], []
    
    # Step 2: Sort detections by confidence (highest first)
    # Create a list of (score, rect) tuples and sort by score descending
    # Use sorted() with key parameter or zip and sort
    if len(weights) == 0:
        # If no weights, use dummy scores
        scores = [1.0] * len(rects)
    else:
        scores = weights
    
    # Create list of (score, rect, index) tuples
    detections = [(scores[i], rects[i], i) for i in range(len(rects))]
    # Sort by score descending
    detections.sort(key=lambda x: x[0], reverse=True)
    
    # Step 3: Initialize list to track which detections to keep
    # keep = []  # list of indices to keep
    keep = []
    
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    
    # Step 4: For each detection (starting with highest confidence):
    #   a. Calculate IoU with all remaining detections
    #   b. If IoU > threshold, suppress the lower-confidence detection
    #   c. Keep the current detection
    while len(detections) > 0:
        # Keep the highest confidence detection
        current_score, current_rect, current_idx = detections.pop(0)
        keep.append(current_idx)
        
        # Remove overlapping detections
        detections = [
            (score, rect, idx) for score, rect, idx in detections
            if calculate_iou(current_rect, rect) <= threshold
        ]
    
    # Step 5: Extract filtered rects and weights
    # filtered_rects = [rects[i] for i in keep]
    # filtered_weights = [weights[i] for i in keep]
    filtered_rects = [rects[i] for i in keep]
    if len(weights) > 0:
        filtered_weights = [weights[i] for i in keep]
    else:
        filtered_weights = []
    
    # Step 6: Return filtered results
    return filtered_rects, filtered_weights


def visualize_detections_on_image(image, rects, weights):
    """
    Draw bounding boxes and confidence scores on the image.

    This function visualizes the detection results by drawing rectangles
    around detected objects and displaying their confidence scores.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR format).
    rects : list of tuple
        List of bounding boxes, each as (x, y, width, height).
    weights : list of float
        Confidence scores for each detection.

    Returns
    -------
    result_image : np.ndarray
        Image with bounding boxes and labels drawn.

    Notes
    -----
    Step-by-step implementation:
    1. Create a copy of the input image (to avoid modifying the original)
    2. For each detection:
       a. Extract (x, y, w, h) from the bounding box
       b. Draw a rectangle using cv2.rectangle()
       c. Get the confidence score (if available)
       d. Draw text label with confidence score using cv2.putText()
    3. Return the annotated image
    """
    # Step 1: Create a copy of the input image
    # result = image.copy()
    result = image.copy()
    
    # Step 2: Loop through each detection
    # for i, (x, y, w, h) in enumerate(rects):
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        
        # Step 3: Draw bounding box rectangle
        # Use cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        # Color: (0, 255, 0) for green, thickness: 2
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Step 4: Get confidence score (if available)
        # if i < len(weights):
        #     score = weights[i]
        # else:
        #     score = 0.0
        if i < len(weights):
            score = weights[i]
        else:
            score = 0.0
        
        # Step 5: Draw text label with confidence score
        # Use cv2.putText(result, text, (x, y-10), font, fontScale, color, thickness)
        # Text format: "score: 0.82" (true SVM decision function value)
        # Position: (x, y-10) to place text above the box
        label = f"score: {score:.2f}"
                cv2.putText(
            result,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # Step 6: Return the annotated image
    return result


def run_human_detection_single_image(image_path):
    """
    Run human detection on a single image and save the result.

    This is a complete pipeline function that loads an image, detects humans,
    applies NMS, visualizes results, and saves the output.

    Parameters
    ----------
    image_path : str
        Path to the input image file.

    Returns
    -------
    None
        Saves the result image to disk.

    Notes
    -----
    Step-by-step implementation:
    1. Load the image using load_grayscale_image() or cv2.imread()
    2. Set detection parameters (win_stride, padding, scale)
    3. Call detect_humans_in_image() to get detections
    4. Apply NMS using apply_nms_to_detections()
    5. Visualize results using visualize_detections_on_image()
    6. Save the result image to data/results/human_detection/
    7. Print detection count and save path
    """
    # !!! Burak: set your image path here
    
    # Step 1: Load the input image
    # image = load_grayscale_image(image_path) or cv2.imread(image_path)
    # Convert to BGR if needed for visualization
    image = cv2.imread(image_path)
    if image is None:
        # Try loading as grayscale and convert to BGR
        image = load_grayscale_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Step 2: Set detection parameters
    # Note: detect_humans_in_image() now uses optimized parameters by default
    # Pass None to use optimized values, or specify custom values
    
    # Step 3: Detect humans in the image (uses optimized parameters)
    # rects, weights = detect_humans_in_image(image, win_stride, padding, scale)
    rects, weights = detect_humans_in_image(image, None, None, None)
    
    # Step 4: Apply Non-Maximum Suppression
    # nms_threshold = 0.3  # IoU threshold for NMS
    # filtered_rects, filtered_weights = apply_nms_to_detections(rects, weights, nms_threshold)
    nms_threshold = 0.3
    filtered_rects, filtered_weights = apply_nms_to_detections(rects, weights, nms_threshold)
    
    # Step 5: Visualize detections
    # result_image = visualize_detections_on_image(image, filtered_rects, filtered_weights)
    result_image = visualize_detections_on_image(image, filtered_rects, filtered_weights)
    
    # Step 6: Create output directory
    # output_dir = "data/results/human_detection/"
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = "data/results/human_detection/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 7: Save the result image
    # output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    # cv2.imwrite(output_path, result_image)
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, result_image)
    
    # Step 8: Print results
    # print(f"Detected {len(filtered_rects)} person(s) in {image_path}")
    # print(f"Result saved to: {output_path}")
    print(f"Detected {len(filtered_rects)} person(s) in {image_path}")
    print(f"Result saved to: {output_path}")


def evaluate_threshold_effects(image, thresholds=[0.0, 0.3, 0.5, 0.7]):
    """
    Runs detection with multiple hitThreshold values and compares results.
    
    This function evaluates the effect of different hitThreshold values on:
    - Number of detections
    - Confidence distribution
    - False positives (estimated)
    - False negatives (estimated, if ground truth available)

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    thresholds : list of float, optional
        List of hitThreshold values to test. Default: [0.0, 0.3, 0.5, 0.7]

    Returns
    -------
    results : dict
        Dictionary containing results for each threshold with:
        - 'threshold': threshold value
        - 'num_detections': number of detections
        - 'avg_confidence': average confidence score
        - 'min_confidence': minimum confidence score
        - 'max_confidence': maximum confidence score
        - 'confidence_distribution': list of all confidence scores

    Notes
    -----
    Step-by-step implementation:
    1. Initialize HOG detector
    2. For each threshold value:
       a. Run detection with that threshold
       b. Apply NMS
       c. Collect statistics (count, scores, etc.)
    3. Print comparison table
    4. Return results dictionary
    """
    # Step 1: Initialize HOG detector
    # Use OpenCV's default people detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Fixed optimal parameters
    win_stride = (8, 8)
    padding = (16, 16)
    scale = 1.05
    nms_threshold = 0.3
    
    # Step 2: Test each threshold value
    results = []
    
    print(f"\n{'='*80}")
    print(f"THRESHOLD EFFECTS EVALUATION")
    print(f"{'='*80}")
    print(f"Testing {len(thresholds)} different hitThreshold values...")
    print(f"Image shape: {image.shape}")
    print(f"{'='*80}\n")
    
    for threshold in thresholds:
        # Run detection with current threshold
        rects, weights = hog.detectMultiScale(
            image,
            winStride=win_stride,
            padding=padding,
            scale=scale,
            hitThreshold=threshold
        )
        
        # Apply NMS
        rects_list = [tuple(rect) for rect in rects]
        if len(weights) > 0:
            if isinstance(weights, np.ndarray):
                if weights.ndim > 1:
                    weights_list = [float(w[0]) for w in weights]
                else:
                    weights_list = [float(w) for w in weights]
            else:
                weights_list = [float(w) for w in weights]
        else:
            weights_list = []
        
        # Apply NMS to filter overlapping detections
        filtered_rects, filtered_weights = apply_nms_to_detections(
            rects_list, weights_list, nms_threshold
        )
        
        # Calculate statistics
        num_detections = len(filtered_rects)
        if len(filtered_weights) > 0:
            avg_confidence = np.mean(filtered_weights)
            min_confidence = np.min(filtered_weights)
            max_confidence = np.max(filtered_weights)
            confidence_distribution = filtered_weights
        else:
            avg_confidence = 0.0
            min_confidence = 0.0
            max_confidence = 0.0
            confidence_distribution = []
        
        # Store results
        result = {
            'threshold': threshold,
            'num_detections': num_detections,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'confidence_distribution': confidence_distribution
        }
        results.append(result)
    
    # Step 3: Print comparison table
    print(f"{'Threshold':<12} {'Detections':<12} {'Avg Conf':<12} {'Min Conf':<12} {'Max Conf':<12}")
    print("-" * 80)
    for result in results:
        print(f"{result['threshold']:<12.2f} {result['num_detections']:<12} "
              f"{result['avg_confidence']:<12.2f} {result['min_confidence']:<12.2f} "
              f"{result['max_confidence']:<12.2f}")
    
    # Step 4: Print analysis
    print(f"\n{'='*80}")
    print(f"ANALYSIS:")
    print(f"  Lower threshold (0.0): More detections, may include false positives")
    print(f"  Higher threshold (0.7): Fewer detections, higher confidence, may miss some persons")
    print(f"  Optimal threshold (0.5): Balanced detection (academically validated)")
    
    # Calculate differences
    if len(results) > 1:
        first_detections = results[0]['num_detections']
        last_detections = results[-1]['num_detections']
        print(f"\n  Detection difference (threshold {thresholds[0]:.1f} vs {thresholds[-1]:.1f}): "
              f"{first_detections - last_detections} detections")
    
    print(f"{'='*80}\n")
    
    # Step 5: Return results
    return results


def run_human_detection_dataset():
    """
    Run human detection on multiple images from a dataset folder.

    This function processes all images in a folder, detects humans in each,
    and saves all results to an output folder.

    Returns
    -------
    None
        Saves all result images to disk.

    Notes
    -----
    Step-by-step implementation:
    1. Define input folder path (data/test_images/ or similar)
    2. Define output folder path (data/results/human_detection/)
    3. Get list of all image files in input folder
    4. For each image:
       a. Load the image
       b. Run detection pipeline (detect -> NMS -> visualize)
       c. Save result to output folder
    5. Print summary statistics
    """
    # !!! Burak: set your input and output folder paths here
    
    # Step 1: Define input and output directories
    # input_folder = "data/test_images/"  # !!! Burak: adjust path
    # output_folder = "data/results/human_detection/"
    # os.makedirs(output_folder, exist_ok=True)
    input_folder = "data/test_images/"
    output_folder = "data/results/human_detection/"
    
    # Create separate folders for detected and not detected images
    detected_folder = os.path.join(output_folder, "detected")
    not_detected_folder = os.path.join(output_folder, "not_detected")
    os.makedirs(detected_folder, exist_ok=True)
    os.makedirs(not_detected_folder, exist_ok=True)
    
    # Step 2: Get list of image files - .jpg and .png files for human detection test
    # Filter .jpg and .png files (test images like 000001.jpg, 000096.jpg, etc.)
    # Must process at least 10 images
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    image_files.sort()  # Sort for consistent processing order
    
    # Check if we have at least 10 images
    if len(image_files) < 10:
        print(f"⚠ Warning: Only {len(image_files)} images found. Assignment requires at least 10 images.")
        else:
        print(f"✓ Found {len(image_files)} images (requirement: at least 10)")
    
    print(f"\n{'='*80}")
    print(f"HUMAN DETECTION - Using OpenCV Pretrained HOG + SVM Model")
    print(f"{'='*80}")
    print(f"Found {len(image_files)} test images (.jpg/.png files)")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"  - Detected images: {detected_folder}")
    print(f"  - Not detected images: {not_detected_folder}")
    print(f"Detection parameters (OPTIMIZED for better sensitivity):")
    print(f"  - winStride: (4, 4) [smaller = more thorough search]")
    print(f"  - padding: (24, 24) [more = better border detection]")
    print(f"  - scale: 1.03 [smaller = more scale levels]")
    print(f"  - hitThreshold: 0.0 [lower = more detections, reduces false negatives]")
    print(f"  - NMS IoU threshold: 0.3")
    print(f"  - Min confidence after NMS: 0.1 [filters very weak detections]")
    print(f"{'='*80}\n")

    # Step 3: Initialize detection parameters
    # Note: detect_humans_in_image() now uses optimized parameters internally
    # Parameters are optimized for better detection accuracy
    nms_threshold = 0.3
    
    # Statistics tracking
    total_images = 0
    total_detections = 0
    images_with_persons = 0
    images_without_persons = 0
    detection_results = []
    
    # Step 4: Process each image
    # for image_file in image_files:
    #     image_path = os.path.join(input_folder, image_file)
    #     
    #     # Load image
    #     image = cv2.imread(image_path)
    #     
    #     # Detect humans (uses optimized parameters internally)
    #     rects, weights = detect_humans_in_image(image, None, None, None)
    #     
    #     # Apply NMS
    #     filtered_rects, filtered_weights = apply_nms_to_detections(rects, weights, nms_threshold)
    #     
    #     # Visualize
    #     result_image = visualize_detections_on_image(image, filtered_rects, filtered_weights)
    #     
    #     # Save result
    #     output_path = os.path.join(output_folder, f"detected_{image_file}")
    #     cv2.imwrite(output_path, result_image)
    #     
    #     print(f"Processed {image_file}: {len(filtered_rects)} person(s) detected")
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Warning: Could not load {image_file}, skipping...")
            continue
        
        total_images += 1
        
        # Detect humans using optimized model (better sensitivity parameters)
        # Parameters: win_stride=(4,4), padding=(24,24), scale=1.03, hitThreshold=0.0
        # These parameters maximize detection rate (reduce false negatives)
        rects, weights = detect_humans_in_image(image, None, None, None)
        
        # Apply NMS to filter overlapping detections
        filtered_rects, filtered_weights = apply_nms_to_detections(rects, weights, nms_threshold)
        
        # Optional: Filter very low confidence detections (if needed)
        # Keep detections with confidence >= 0.1 (very permissive to reduce false negatives)
        min_confidence = 0.1
        if len(filtered_weights) > 0:
            high_conf_indices = [i for i, w in enumerate(filtered_weights) if w >= min_confidence]
            filtered_rects = [filtered_rects[i] for i in high_conf_indices]
            filtered_weights = [filtered_weights[i] for i in high_conf_indices]
        
        # Count detections
        num_detections = len(filtered_rects)
        total_detections += num_detections
        
        # Track statistics
        if num_detections > 0:
            images_with_persons += 1
            status = "✓ PERSON DETECTED"
        else:
            images_without_persons += 1
            status = "✗ NO PERSON"
        
        # Calculate average score
        avg_score = np.mean(filtered_weights) if len(filtered_weights) > 0 else 0.0
        
        # Store results with all confidence scores
        detection_results.append({
            'image': image_file,
            'detections': num_detections,
            'status': status,
            'scores': filtered_weights if len(filtered_weights) > 0 else [],
            'avg_score': avg_score,
            'fp': 0,  # False positives (to be manually determined if ground truth available)
            'fn': 0   # False negatives (to be manually determined if ground truth available)
        })

        # Visualize detections
        result_image = visualize_detections_on_image(image, filtered_rects, filtered_weights)
        
        # Save result to appropriate folder based on detection status
        if num_detections > 0:
            # Save to detected folder
            output_path = os.path.join(detected_folder, f"detected_{image_file}")
        else:
            # Save to not_detected folder
            output_path = os.path.join(not_detected_folder, f"not_detected_{image_file}")
        cv2.imwrite(output_path, result_image)
        
        # Print per-image results with all confidence values
        if num_detections > 0:
            scores_str = ", ".join([f"{s:.2f}" for s in filtered_weights])
            print(f"{status:20} | {image_file:20} | {num_detections:2} person(s) | Avg score: {avg_score:.2f}")
            print(f"  Confidence scores: [{scores_str}]")
        else:
            print(f"{status:20} | {image_file:20} | {num_detections:2} person(s)")
    
    # Step 5: Print detailed statistics table
    print(f"\n{'='*80}")
    print(f"DETECTION SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Separate detected and non-detected images
    detected_images = [r for r in detection_results if r['detections'] > 0]
    non_detected_images = [r for r in detection_results if r['detections'] == 0]
    
    # Print TESPIT EDİLENLER (Detected)
    print(f"\n{'='*80}")
    print(f"✓ TESPİT EDİLENLER ({len(detected_images)} görüntü)")
    print(f"{'='*80}")
    if len(detected_images) > 0:
        print(f"\n{'Image':<25} {'Persons':<10} {'AvgScore':<12} {'FP':<6} {'FN':<6}")
        print("-" * 80)
        for result in detected_images:
            image_name = result['image']
            num_det = result['detections']
            avg_sc = result['avg_score']
            fp = result['fp']
            fn = result['fn']
            print(f"{image_name:<25} {num_det:<10} {avg_sc:<12.2f} {fp:<6} {fn:<6}")
    else:
        print("  Tespit edilen görüntü yok.")
    
    # Print TESPİT EDİLEMEYENLER (Non-detected)
    print(f"\n{'='*80}")
    print(f"✗ TESPİT EDİLEMEYENLER ({len(non_detected_images)} görüntü)")
    print(f"{'='*80}")
    if len(non_detected_images) > 0:
        print(f"\n{'Image':<25} {'Persons':<10} {'AvgScore':<12} {'FP':<6} {'FN':<6}")
        print("-" * 80)
        for result in non_detected_images:
            image_name = result['image']
            num_det = result['detections']
            avg_sc = result['avg_score']
            fp = result['fp']
            fn = result['fn']
            print(f"{image_name:<25} {num_det:<10} {'-':<12} {fp:<6} {fn:<6}")
    else:
        print("  Tespit edilemeyen görüntü yok.")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"ÖZET İSTATİSTİKLER:")
    print(f"{'='*80}")
    print(f"  Toplam işlenen görüntü:     {total_images}")
    print(f"  Toplam tespit edilen kişi:  {total_detections}")
    print(f"  Tespit edilen görüntü:      {images_with_persons} ({images_with_persons/total_images*100:.1f}%)")
    print(f"  Tespit edilemeyen görüntü:  {images_without_persons} ({images_without_persons/total_images*100:.1f}%)")
    if total_detections > 0:
        avg_per_image = total_detections / images_with_persons if images_with_persons > 0 else 0
        print(f"  Görüntü başına ortalama tespit: {avg_per_image:.2f}")
        all_scores = [s for r in detection_results for s in r['scores']]
        if len(all_scores) > 0:
            print(f"  Genel ortalama confidence:    {np.mean(all_scores):.2f}")
            print(f"  Min confidence:                {np.min(all_scores):.2f}")
            print(f"  Max confidence:                {np.max(all_scores):.2f}")
    
    print(f"\nSonuçlar kaydedildi:")
    print(f"  - Tespit edilenler: {detected_folder}")
    print(f"  - Tespit edilemeyenler: {not_detected_folder}")
    print(f"{'='*80}\n")


# ============================================================================
# PART B — CUSTOM OBJECT DETECTION
# Sliding Window + HOG + Linear SVM (skeleton only)
# ============================================================================

def slide_window_over_image(image, window_size, stride):
    """
    Slide a fixed-size window across an image.

    This function generates all possible window positions by sliding a window
    of fixed size across the image with a specified stride.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    window_size : tuple of int
        Size of the sliding window as (height, width) in pixels.
    stride : tuple of int
        Step size for sliding as (step_y, step_x) in pixels.
        Smaller stride = more windows but slower processing.

    Yields
    ------
    (x, y, window_patch) : tuple
        Top-left coordinates (x, y) and the image patch as a NumPy array.

    Notes
    -----
    Step-by-step implementation:
    1. Get image dimensions (height, width)
    2. Extract window dimensions (window_h, window_w)
    3. Extract stride values (stride_y, stride_x)
    4. Loop through all possible window positions:
       a. Calculate window position: y from 0 to (img_h - window_h), step by stride_y
       b. Calculate window position: x from 0 to (img_w - window_w), step by stride_x
       c. Extract window patch: image[y:y+window_h, x:x+window_w]
       d. Yield (x, y, window_patch)
    """
    # Step 1: Get image dimensions
    # img_h, img_w = image.shape[:2]  # Get height and width
    img_h, img_w = image.shape[:2]
    
    # Step 2: Extract window and stride dimensions
    # window_h, window_w = window_size
    # stride_y, stride_x = stride
    window_h, window_w = window_size
    stride_y, stride_x = stride
    
    # Step 3: Loop through all possible window positions
    # for y in range(0, img_h - window_h + 1, stride_y):
    #     for x in range(0, img_w - window_w + 1, stride_x):
    for y in range(0, img_h - window_h + 1, stride_y):
        for x in range(0, img_w - window_w + 1, stride_x):
            # Step 4: Extract window patch
            # window_patch = image[y:y+window_h, x:x+window_w]
            window_patch = image[y:y+window_h, x:x+window_w]
            
            # Step 5: Yield the window position and patch
            # yield (x, y, window_patch)
            yield (x, y, window_patch)


def extract_hog_for_window(patch):
    """
    Extract HOG features from a single image patch.

    This function computes the HOG descriptor for a small image window.
    The patch should be resized to a standard size (e.g., 64x128) before
    computing HOG features.

    Parameters
    ----------
    patch : np.ndarray
        Image patch (window) as a 2D NumPy array.

    Returns
    -------
    hog_features : np.ndarray
        1D NumPy array containing the HOG feature vector.

    Notes
    -----
    Step-by-step implementation:
    1. Resize patch to standard size (e.g., 64x128 for pedestrian detection)
    2. Ensure patch is grayscale (convert if needed)
    3. Call compute_hog_descriptor() with appropriate parameters:
       - cell_size: typically (8, 8)
       - block_size: typically (2, 2)
       - num_bins: typically 9
    4. Return the HOG feature vector
    """
    # Step 1: Define standard window size
    # standard_size = (64, 128)  # (height, width) for pedestrian-like objects
    # Or use (64, 64) for square objects
    standard_size = (64, 128)  # (height, width)
    
    # Step 2: Resize patch to standard size
    # Use cv2.resize(patch, (width, height)) to resize the patch
    # resized_patch = cv2.resize(patch, (standard_size[1], standard_size[0]))
    resized_patch = cv2.resize(patch, (standard_size[1], standard_size[0]))
    
    # Step 3: Ensure patch is grayscale
    # if len(resized_patch.shape) == 3:
    #     resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY)
    if len(resized_patch.shape) == 3:
        resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Compute HOG descriptor
    # Use compute_hog_descriptor() from hog_implementation module
    # hog_features = compute_hog_descriptor(
    #     resized_patch,
    #     cell_size=(8, 8),
    #     block_size=(2, 2),
    #     num_bins=9
    # )
    hog_features = compute_hog_descriptor(
        resized_patch,
        cell_size=(8, 8),
        block_size=(2, 2),
        num_bins=9
    )
    
    # Step 5: Return the feature vector
    return hog_features


def train_custom_svm_classifier():
    """
    Train a Linear SVM classifier for custom object detection.

    This function loads positive and negative training samples, extracts
    HOG features from each, and trains a Linear SVM classifier.

    Returns
    -------
    classifier : LinearSVC
        Trained SVM classifier.

    Notes
    -----
    Step-by-step implementation:
    1. Load positive training samples from data/training_set/positive/
    2. Load negative training samples from data/training_set/negative/
    3. For each positive sample:
       a. Resize to standard window size
       b. Extract HOG features using extract_hog_for_window()
       c. Add to feature matrix X with label y=1
    4. For each negative sample:
       a. Resize to standard window size
       b. Extract HOG features using extract_hog_for_window()
       c. Add to feature matrix X with label y=0
    5. Create feature matrix X (shape: num_samples x feature_dim)
    6. Create label vector y (shape: num_samples)
    7. Train LinearSVC classifier: classifier.fit(X, y)
    8. Save the trained model to models/trained_classifier.pkl
    9. Return the classifier
    """
    # !!! Burak: set your training data paths here
    # positive_folder = "data/training_set/positive/"
    # negative_folder = "data/training_set/negative/"
    positive_folder = "data/training_set/positive/"
    negative_folder = "data/training_set/negative/"
    
    # Step 1: Initialize lists for features and labels
    # X = []  # list of feature vectors
    # y = []  # list of labels (1 for positive, 0 for negative)
    X = []
    y = []
    
    # Step 2: Load positive training samples
    # Use os.listdir() or glob to get all image files in positive_folder
    # for image_file in positive_images:
    #     image_path = os.path.join(positive_folder, image_file)
    #     patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     
    #     # Extract HOG features
    #     features = extract_hog_for_window(patch)
    #     X.append(features)
    #     y.append(1)  # positive label
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if os.path.exists(positive_folder):
        positive_files = [
            f for f in os.listdir(positive_folder)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]
        for image_file in positive_files:
            image_path = os.path.join(positive_folder, image_file)
            patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if patch is not None:
                features = extract_hog_for_window(patch)
                X.append(features)
                y.append(1)
        print(f"Loaded {len(positive_files)} positive samples")
    else:
        print(f"Warning: Positive folder not found: {positive_folder}")
    
    # Step 3: Load negative training samples
    # Similar to step 2, but use negative_folder and label y=0
    if os.path.exists(negative_folder):
        negative_files = [
            f for f in os.listdir(negative_folder)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]
        for image_file in negative_files:
            image_path = os.path.join(negative_folder, image_file)
            patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if patch is not None:
                features = extract_hog_for_window(patch)
                X.append(features)
                y.append(0)
        print(f"Loaded {len(negative_files)} negative samples")
    else:
        print(f"Warning: Negative folder not found: {negative_folder}")
    
    # Step 4: Convert to numpy arrays
    # X = np.array(X)
    # y = np.array(y)
    if len(X) == 0:
        raise ValueError("No training samples found! Please add images to positive and negative folders.")
    
    X = np.array(X)
    y = np.array(y)
    
    # Step 5: Train Linear SVM classifier
    # classifier = LinearSVC(C=1.0, max_iter=10000)
    # classifier.fit(X, y)
    classifier = LinearSVC(C=1.0, max_iter=20000)
    classifier.fit(X, y)
    
    # Step 6: Evaluate training accuracy (optional)
    # accuracy = classifier.score(X, y)
    # print(f"Training accuracy: {accuracy:.4f}")
    accuracy = classifier.score(X, y)
    print(f"Training accuracy: {accuracy:.4f}")

    # Step 7: Save the trained model
    # import joblib or pickle
    # model_path = "models/trained_classifier.pkl"
    # os.makedirs("models/", exist_ok=True)
    # joblib.dump(classifier, model_path)  # or use pickle
    # print(f"Model saved to: {model_path}")
    model_path = "models/trained_classifier.pkl"
    os.makedirs("models/", exist_ok=True)
    joblib.dump(classifier, model_path)
    print(f"Model saved to: {model_path}")
    
    # Step 8: Return the classifier
    return classifier


def build_template_from_positive_images(positive_folder="data/training_set/positive/"):
    """
    Build HOG template from positive training images (hazır model yaklaşımı).
    
    Pozitif görüntülerden HOG özelliklerini çıkarıp ortalama template oluşturur.
    Eğitim yapmadan, template matching ile tespit yapılır.
    
    Parameters
    ----------
    positive_folder : str
        Path to folder containing positive training images.
    
    Returns
    -------
    template_features : np.ndarray
        Average HOG feature vector from positive images.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not os.path.exists(positive_folder):
        raise FileNotFoundError(f"Positive folder not found: {positive_folder}")
    
    positive_files = [
        f for f in os.listdir(positive_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    
    if len(positive_files) == 0:
        raise ValueError(f"No positive images found in {positive_folder}")
    
    # Extract HOG features from all positive images
    all_features = []
    for image_file in positive_files:
        image_path = os.path.join(positive_folder, image_file)
        patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if patch is not None:
            features = extract_hog_for_window(patch)
            all_features.append(features)
    
    if len(all_features) == 0:
        raise ValueError("Could not extract features from positive images")
    
    # Calculate average HOG features (template)
    template_features = np.mean(all_features, axis=0)
    
    print(f"Template created from {len(all_features)} positive images")
    print(f"Template feature dimension: {template_features.shape}")
    
    return template_features


def build_negative_template(negative_folder="data/training_set/negative/"):
    """
    Build HOG template from negative training images.
    
    Negatif görüntülerden HOG özelliklerini çıkarıp ortalama template oluşturur.
    Bu template, pozitif template ile karşılaştırma yapmak için kullanılır.
    
    Parameters
    ----------
    negative_folder : str
        Path to folder containing negative training images.
    
    Returns
    -------
    template_features : np.ndarray or None
        Average HOG feature vector from negative images, or None if no negatives.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not os.path.exists(negative_folder):
        return None
    
    negative_files = [
        f for f in os.listdir(negative_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    
    if len(negative_files) == 0:
        return None
    
    # Extract HOG features from all negative images
    all_features = []
    for image_file in negative_files:
        image_path = os.path.join(negative_folder, image_file)
        patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if patch is not None:
            features = extract_hog_for_window(patch)
            all_features.append(features)
    
    if len(all_features) == 0:
        return None
    
    # Calculate average HOG features (template)
    template_features = np.mean(all_features, axis=0)
    
    print(f"Negative template created from {len(all_features)} negative images")
    
    return template_features


def detect_custom_objects_with_svm(image, classifier=None, score_threshold=0.0):
    """
    Detect if custom object exists in an image using sliding window + HOG + SVM classifier.
    
    Bu fonksiyon SVM classifier kullanarak daha güçlü tespit yapar.
    Template matching'den çok daha güçlü ve doğru sonuçlar verir.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    classifier : LinearSVC, optional
        Trained SVM classifier. If None, loads from saved model.
    score_threshold : float
        Minimum SVM decision function score threshold.

    Returns
    -------
    has_object : bool
        True if object detected, False otherwise.
    max_score : float
        Maximum SVM decision score found.
    result_image : np.ndarray
        Image with text label (UÇAK VAR or UÇAK YOK).
    """
    # Step 1: Load classifier if not provided
    if classifier is None:
        model_path = "models/trained_classifier.pkl"
        if os.path.exists(model_path):
            classifier = joblib.load(model_path)
            print(f"✓ SVM model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"SVM model not found: {model_path}. Please train the model first.")
    
    # Step 2: Define detection parameters
    window_sizes = [(64, 64), (96, 96), (128, 128), (64, 128)]  # Farklı boyutlar
    stride = (16, 16)  # Pencere adımı
    
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Step 3: Sliding Window - Her pencere için SVM ile sınıflandırma
    max_score = -float('inf')  # SVM decision scores can be negative
    all_scores = []
    
    # Farklı pencere boyutları için tarama
    for window_size in window_sizes:
    window_h, window_w = window_size

        # Skip if image is smaller than window
        if gray_image.shape[0] < window_h or gray_image.shape[1] < window_w:
            continue
        
        # Slide window across image
        for x, y, window_patch in slide_window_over_image(gray_image, window_size, stride):
            # Extract HOG features for this window
            try:
                features = extract_hog_for_window(window_patch)
                
                # Use SVM to predict (decision_function gives raw scores)
                # Positive scores indicate positive class (object present)
                score = classifier.decision_function([features])[0]
                all_scores.append(score)
                
                # Track maximum score
                if score > max_score:
                    max_score = score
                    
            except Exception as e:
                # Skip if feature extraction fails
                continue
    
    # Step 4: Determine if object exists based on threshold
    # SVM decision function: positive values = positive class, negative = negative class
    has_object = max_score > score_threshold
    
    # Prepare result image
    result_image = image.copy()
    if len(image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    # Draw text label in top-right corner (sağ üst köşe), smaller size
    if has_object:
        label = "ucak tespit edildi"
        color = (0, 255, 0)  # Green
        # Draw bounding box around entire image (tek bir bounding box)
        cv2.rectangle(result_image, (0, 0), (result_image.shape[1], result_image.shape[0]), (0, 255, 0), 3)
    else:
        label = "ucak yok"
        color = (0, 0, 255)  # Red
    
    # Calculate text position (sağ üst köşe)
    font_scale = 0.4  # Daha küçük yazı
    thickness = 2
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = result_image.shape[1] - text_size[0] - 7  # Sağdan 10px içeride
    text_y = 15  # Yukarıdan 30px
    
    # Draw text label
    cv2.putText(
        result_image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    # Add score info below (daha küçük)
    score_text = f"Score: {max_score:.2f}"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    score_x = result_image.shape[1] - score_size[0] - 10
    score_y = text_y + 25
    cv2.putText(
        result_image,
        score_text,
        (score_x, score_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )
    
    return has_object, max_score, result_image


def detect_custom_objects_in_image(image, template_features=None, negative_template=None, score_threshold=0.65):
    """
    Detect if custom object exists in an image using sliding window + HOG template matching.

    Tüm görüntü için "NESNE VAR" veya "NESNE YOK" kararı verir.
    Sliding window ile tarama yapılır, eğer herhangi bir pencere threshold'u geçerse nesne var demektir.

    Sliding Window yaklaşımı:
    1. Farklı boyutlarda ve konumlarda pencereler ile görüntüyü tarar
    2. Her pencere için HOG özellikleri çıkarır
    3. Her pencere için sınıflandırıcı skorunu hesaplar (template ile benzerlik)
    4. Eşik değerinin üzerindeki pencereler varsa → "NESNE VAR"
    5. Eşik değerinin üzerinde pencere yoksa → "NESNE YOK"

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    template_features : np.ndarray, optional
        HOG template features. If None, loads from positive images.
    negative_template : np.ndarray, optional
        HOG template from negative images. Used for better discrimination.
    score_threshold : float
        Minimum similarity score threshold (0.0 to 1.0).

    Returns
    -------
    has_object : bool
        True if object detected, False otherwise.
    max_score : float
        Maximum similarity score found.
    result_image : np.ndarray
        Image with text label (UÇAK VAR or UÇAK YOK).
    """
    # Step 1: Load or build template features
    if template_features is None:
        positive_folder = "data/training_set/positive/"
        template_features = build_template_from_positive_images(positive_folder)
    
    # Step 2: Define detection parameters
    # Sliding window parameters - farklı boyutlarda pencere
    window_sizes = [(64, 64), (96, 96), (128, 128), (64, 128)]  # Farklı boyutlar
    stride = (16, 16)  # Pencere adımı
    
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Step 3: Sliding Window - Farklı boyutlarda ve konumlarda pencere taraması
    # Her pencere için HOG özellikleri çıkarılır ve template ile karşılaştırılır
    
    # Normalize templates for cosine similarity
    template_norm = template_features / (np.linalg.norm(template_features) + 1e-8)
    
    # Normalize negative template if available
    negative_norm = None
    if negative_template is not None:
        negative_norm = negative_template / (np.linalg.norm(negative_template) + 1e-8)
    
    # Track maximum score across all windows
    max_score = 0.0
    
    # Farklı pencere boyutları için tarama (farklı boyutlarda pencere)
    for window_size in window_sizes:
        window_h, window_w = window_size

        # Skip if image is smaller than window
        if gray_image.shape[0] < window_h or gray_image.shape[1] < window_w:
            continue

        # Slide window across image (farklı konumlarda)
        for x, y, window_patch in slide_window_over_image(gray_image, window_size, stride):
            # Extract HOG features for this window
            try:
                features = extract_hog_for_window(window_patch)

                # Normalize features for cosine similarity
                features_norm = features / (np.linalg.norm(features) + 1e-8)
                
                # Calculate similarity score with positive template
                similarity_pos = np.dot(template_norm, features_norm)
                score_pos = (similarity_pos + 1.0) / 2.0  # Normalize to 0-1 range
                
                # If negative template exists, use difference score for better discrimination
                if negative_norm is not None:
                    similarity_neg = np.dot(negative_norm, features_norm)
                    score_neg = (similarity_neg + 1.0) / 2.0
                    # Use difference: positive score - negative score
                    # This gives better discrimination
                    score = score_pos - score_neg
                    # Normalize to 0-1 range (difference can be negative, so shift)
                    score = (score + 1.0) / 2.0
                else:
                    score = score_pos
                
                # Track maximum score
                if score > max_score:
                    max_score = score
                    
            except Exception as e:
                # Skip if feature extraction fails
                continue
    
    # Step 4: Determine if object exists based on threshold
    # Eşik değerinin üzerindeki pencereler varsa nesne var demektir
    has_object = max_score > score_threshold
    
    # Prepare result image
    result_image = image.copy()
    if len(image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    # Draw text label in top-right corner (sağ üst köşe), smaller size
    if has_object:
        label = "ucak tespit edildi"
        color = (0, 255, 0)  # Green
        # Draw bounding box around entire image (tek bir bounding box)
        cv2.rectangle(result_image, (0, 0), (result_image.shape[1], result_image.shape[0]), (0, 255, 0), 3)
    else:
        label = "ucak yok"
        color = (0, 0, 255)  # Red
    
    # Calculate text position (sağ üst köşe)
    font_scale = 0.4  # Daha küçük yazı
    thickness = 2
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = result_image.shape[1] - text_size[0] - 7  # Sağdan 10px içeride
    text_y = 15  # Yukarıdan 30px
    
    # Draw text label
        cv2.putText(
        result_image,
            label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    # Add score info below (daha küçük)
    score_text = f"Score: {max_score:.2f}"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    score_x = result_image.shape[1] - score_size[0] - 10
    score_y = text_y + 25
    cv2.putText(
        result_image,
        score_text,
        (score_x, score_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
        color,
        1
        )

    return has_object, max_score, result_image


def run_custom_detection_dataset_with_svm(classifier=None, score_threshold=0.0):
    """
    Run custom object detection on multiple test images using trained SVM classifier.
    
    Uses sliding window approach with HOG + SVM. This is much stronger than template matching.

    Parameters
    ----------
    classifier : LinearSVC, optional
        Trained SVM classifier. If None, loads from saved model.
    score_threshold : float
        Minimum SVM decision function score threshold.
    
    Returns
    -------
    None
        Saves all result images to disk.
    """
    input_folder = "data/test_images/"
    output_folder = "data/results/custom_detection/"
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    image_files.sort()
    
    print(f"Found {len(image_files)} test images")
    print(f"Output folder: {output_folder}\n")
    
    # Statistics
    total_images = 0
    total_detections = 0
    images_with_objects = 0
    images_without_objects = 0
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Warning: Could not load {image_file}, skipping...")
            continue
        
        total_images += 1
        
        # Detect objects using SVM classifier
        # Returns: has_object (bool), max_score (float), result_image
        has_object, max_score, result_image = detect_custom_objects_with_svm(
            image, 
            classifier=classifier,
            score_threshold=score_threshold
        )

        # Count detections
        if has_object:
            images_with_objects += 1
            total_detections += 1
            status = "✓ UÇAK VAR"
        else:
            images_without_objects += 1
            status = "✗ UÇAK YOK"

        # Save result
        output_path = os.path.join(output_folder, f"detected_{image_file}")
        cv2.imwrite(output_path, result_image)
        
        # Print per-image results
        print(f"{status:20} | {image_file:25} | Max score: {max_score:.2f}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ÖZET İSTATİSTİKLER:")
    print(f"{'='*80}")
    print(f"  Toplam işlenen görüntü:     {total_images}")
    print(f"  UÇAK VAR:                    {images_with_objects} ({images_with_objects/total_images*100:.1f}%)")
    print(f"  UÇAK YOK:                    {images_without_objects} ({images_without_objects/total_images*100:.1f}%)")
    print(f"\nSonuçlar kaydedildi: {output_folder}")
    print(f"{'='*80}\n")


def run_custom_detection_dataset_with_template(template_features, negative_template=None, score_threshold=0.65):
    """
    Run custom object detection on multiple test images using template matching.
    
    Uses sliding window approach with HOG template matching.
    
    Parameters
    ----------
    template_features : np.ndarray
        HOG template features from positive images.
    negative_template : np.ndarray, optional
        HOG template features from negative images. Improves discrimination.
    score_threshold : float
        Minimum similarity score threshold.
    
    Returns
    -------
    None
        Saves all result images to disk.
    """
    input_folder = "data/test_images/"
    output_folder = "data/results/custom_detection/"
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    image_files.sort()
    
    print(f"Found {len(image_files)} test images")
    print(f"Output folder: {output_folder}\n")
    
    # Statistics
    total_images = 0
    total_detections = 0
    images_with_objects = 0
    images_without_objects = 0
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Warning: Could not load {image_file}, skipping...")
            continue
        
        total_images += 1
        
        # Detect objects using template matching
        # Returns: has_object (bool), max_score (float), result_image
        has_object, max_score, result_image = detect_custom_objects_in_image(
            image, 
            template_features=template_features,
            negative_template=negative_template,
            score_threshold=score_threshold
        )

        # Count detections
        if has_object:
            images_with_objects += 1
            total_detections += 1
            status = "✓ UÇAK VAR"
        else:
            images_without_objects += 1
            status = "✗ UÇAK YOK"
        
        # Save result
        output_path = os.path.join(output_folder, f"detected_{image_file}")
        cv2.imwrite(output_path, result_image)
        
        # Print per-image results
        print(f"{status:20} | {image_file:25} | Max score: {max_score:.2f}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ÖZET İSTATİSTİKLER:")
    print(f"{'='*80}")
    print(f"  Toplam işlenen görüntü:     {total_images}")
    print(f"  UÇAK VAR:                    {images_with_objects} ({images_with_objects/total_images*100:.1f}%)")
    print(f"  UÇAK YOK:                    {images_without_objects} ({images_without_objects/total_images*100:.1f}%)")
    print(f"\nSonuçlar kaydedildi: {output_folder}")
    print(f"{'='*80}\n")


def run_custom_detection_dataset():
    """
    Run custom object detection on multiple test images (legacy function for trained model).

    This function processes all images in a test folder, detects objects
    in each, and saves all results to an output folder.

    Returns
    -------
    None
        Saves all result images to disk.
    """
    # !!! Burak: set your test and output folder paths here
    # input_folder = "data/test_images/"
    # output_folder = "data/results/custom_detection/"
    input_folder = "data/test_images/"
    output_folder = "data/results/custom_detection/"
    
    # Step 1: Create output directory
    # os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Step 2: Get list of image files
    # Use os.listdir() or glob to find all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    
    # Step 3: Process each image
    # for image_file in image_files:
    #     image_path = os.path.join(input_folder, image_file)
    #     
    #     # Load image
    #     image = cv2.imread(image_path)
    #     
    #     # Detect objects
    #     detections, result_image = detect_custom_objects_in_image(image)
    #     
    #     # Save result
    #     output_path = os.path.join(output_folder, f"detected_{image_file}")
    #     cv2.imwrite(output_path, result_image)
    #     
    #     print(f"Processed {image_file}: {len(detections)} object(s) detected")
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load {image_file}, skipping...")
            continue
        
        # Detect objects
        detections, result_image = detect_custom_objects_in_image(image)

        # Save result
        output_path = os.path.join(output_folder, f"detected_{image_file}")
        cv2.imwrite(output_path, result_image)
        
        print(f"Processed {image_file}: {len(detections)} object(s) detected")
    
    # Step 4: Print summary
    # print(f"\nProcessing complete! Results saved to: {output_folder}")
    print(f"\nProcessing complete! Results saved to: {output_folder}")


if __name__ == "__main__":
    """
    Main menu for Object Detection System.
    
    User can choose between:
    1. Human Detection (İnsan Tespiti) - Uses OpenCV's pretrained HOG + SVM model
    2. Custom Object Detection (Nesne Tespiti) - Uses custom trained HOG + SVM model
    """
    
    print("\n" + "="*80)
    print("NESNE TESPİT SİSTEMİ - HOG + SVM")
    print("="*80)
    print("\nLütfen yapmak istediğiniz işlemi seçin:")
    print("\n  1. İNSAN TESPİTİ (Hazır Model)")
    print("     - OpenCV'nin önceden eğitilmiş HOG + SVM modeli")
    print("     - İnsan/pedestrian tespiti yapar")
    print("     - Test görüntülerinde otomatik tespit")
    print("\n  2. NESNE TESPİTİ (Hazır Template Modeli)")
    print("     - Pozitif görüntülerden HOG template oluşturulur")
    print("     - Sliding window ile farklı boyutlarda ve konumlarda tarama")
    print("     - Her pencere için HOG özellikleri çıkarılır")
    print("     - Template ile benzerlik skoru hesaplanır")
    print("     - Eşik değerinin üzerindeki pencereler tespit edilir")
    print("     - NMS ile çakışan tespitler birleştirilir")
    print("     - Pozitif görüntüler gerekli (training_set/positive klasörü)")
    print("\n" + "="*80)
    
    while True:
        try:
            choice = input("\nSeçiminiz (1 veya 2, çıkmak için 0): ").strip()
            
            if choice == "0":
                print("\nÇıkılıyor...")
                break
            elif choice == "1":
    # ========================================================================
                # PART A: Human Detection (İnsan Tespiti)
    # ========================================================================
                print("\n" + "="*80)
                print("İNSAN TESPİTİ - Hazır OpenCV HOG + SVM Modeli")
                print("="*80)
                print("\nHazır model kullanılıyor...")
                print("Test görüntülerinde insan tespiti yapılıyor...\n")
                run_human_detection_dataset()
                print("\n" + "="*80)
                print("İnsan tespiti tamamlandı!")
                print("="*80)
                break
                
            elif choice == "2":
    # ========================================================================
                # PART B: Custom Object Detection (Nesne Tespiti) - Güçlü SVM Modeli
    # ========================================================================
                print("\n" + "="*80)
                print("NESNE TESPİTİ - Güçlü HOG + SVM Modeli")
                print("="*80)
                print("\nYaklaşım: Sliding Window + HOG + SVM Classifier")
                print("  - Pozitif ve negatif görüntülerden SVM eğitilir")
                print("  - Sliding window ile farklı boyutlarda ve konumlarda tarama")
                print("  - Her pencere için HOG özellikleri çıkarılır")
                print("  - SVM classifier ile sınıflandırma yapılır")
                print("  - Template matching'den çok daha güçlü ve doğru!")
                
                # Create training folders if they don't exist
                positive_folder = "data/training_set/positive/"
                negative_folder = "data/training_set/negative/"
                os.makedirs(positive_folder, exist_ok=True)
                os.makedirs(negative_folder, exist_ok=True)
                
                model_path = "models/trained_classifier.pkl"
                
                # Check if model exists
                if os.path.exists(model_path):
                    print(f"\n✓ Eğitilmiş SVM modeli bulundu: {model_path}")
                    use_existing = input("Mevcut modeli kullanmak ister misiniz? (e/h, varsayılan: e): ").strip().lower()
                    if use_existing != 'h':
                        try:
                            classifier = joblib.load(model_path)
                            print("✓ Model yüklendi!")
                            print("\nTest görüntülerinde nesne tespiti yapılıyor...")
                            print("(Sliding window + HOG + SVM)\n")
                            
                            # Run detection with SVM
                            run_custom_detection_dataset_with_svm(classifier=classifier)
                            
                        except Exception as e:
                            print(f"\n❌ Model yüklenirken hata oluştu: {e}")
                            import traceback
                            traceback.print_exc()
                        break
                
                # Check for training data
                pos_files = []
                neg_files = []
                if os.path.exists(positive_folder):
                    pos_files = [f for f in os.listdir(positive_folder) 
                                if os.path.splitext(f.lower())[1] in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}]
                if os.path.exists(negative_folder):
                    neg_files = [f for f in os.listdir(negative_folder) 
                                if os.path.splitext(f.lower())[1] in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}]
                
                if len(pos_files) > 0 and len(neg_files) > 0:
                    print(f"\n✓ Pozitif görüntüler: {len(pos_files)}")
                    print(f"✓ Negatif görüntüler: {len(neg_files)}")
                    print("\nSVM modeli eğitiliyor...")
                    print("(Bu işlem birkaç dakika sürebilir)\n")
                    try:
                        # Train SVM classifier
                        classifier = train_custom_svm_classifier()
                        
                        print("\n✓ Model eğitildi ve kaydedildi!")
                        print("\nTest görüntülerinde nesne tespiti yapılıyor...")
                        print("(Sliding window + HOG + SVM)\n")
                        
                        # Run detection with SVM
                        run_custom_detection_dataset_with_svm(classifier=classifier)
                        
                    except Exception as e:
                        print(f"\n❌ Hata oluştu: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"\n⚠ Eğitim verisi eksik!")
                    print(f"  - Pozitif örnekler: {len(pos_files)}")
                    print(f"  - Negatif örnekler: {len(neg_files)}")
                    print(f"\nKlasörler hazır:")
                    print(f"  ✓ {positive_folder}")
                    print(f"  ✓ {negative_folder}")
                    print("\nLütfen eğitim görüntülerini ekleyin:")
                    print("  - Nesne içeren görüntüler → positive/ klasörüne")
                    print("  - Nesne içermeyen görüntüler → negative/ klasörüne")
                    print("\nVerileri ekledikten sonra tekrar seçenek 2'yi çalıştırın.")
                
                print("\n" + "="*80)
                print("Nesne tespiti tamamlandı!")
                print("="*80)
                break
            else:
                print("\n⚠ Geçersiz seçim! Lütfen 1, 2 veya 0 girin.")
                
        except KeyboardInterrupt:
            print("\n\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"\n❌ Hata oluştu: {e}")
            print("Lütfen tekrar deneyin.")
    
    print("\n" + "="*80)
