"""
Problem 3: Image Classification using HOG features and SVM.

This module implements an image classification system that:
- Loads a multi-class dataset organized by folders
- Extracts HOG features (using either skimage or custom implementation)
- Trains an SVM classifier
- Evaluates performance with metrics and confusion matrix
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Handle both relative and absolute imports
if __name__ == "__main__":
    # Add parent directory to path for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.hog_implementation import compute_hog_descriptor
    from src.utils import load_images_from_folder
else:
    from .hog_implementation import compute_hog_descriptor
    from .utils import load_images_from_folder

import cv2
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

# Use scikit-image HOG as baseline (since Problem 3 allows ready-made libraries)
try:
    from skimage.feature import hog as skimage_hog
except ImportError:
    skimage_hog = None
    print("Warning: scikit-image not available. Only custom HOG will work (after Problem 1 is implemented).")


def load_dataset_from_folder(
    base_folder: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a multi-class image dataset organized by class folders.

    Parameters
    ----------
    base_folder : str
        Base folder containing subfolders, one per class.
        Expected structure:
        base_folder/
        ├── class_0/
        ├── class_1/
        ├── class_2/
        └── ...

    Returns
    -------
    X_images : np.ndarray
        Array of images (can be used for feature extraction later).
        Shape: (num_samples, height, width) for grayscale.
    y_labels : np.ndarray
        Integer labels for each image (0, 1, 2, ...).
    class_names : list of str
        List of class names (folder names).

    Notes
    -----
    This function loads images but does not extract features yet.
    Feature extraction is done separately to allow comparison between
    different HOG implementations.
    """
    # !!! Burak: organize data/training_set/ into one folder per class (e.g., cars, dogs, etc.)

    base_path = Path(base_folder)
    if not base_path.exists():
        raise ValueError(f"Base folder does not exist: {base_folder}")

    # Find all class folders
    class_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in: {base_folder}")

    class_names = [folder.name for folder in class_folders]
    print(f"Found {len(class_names)} classes: {class_names}")

    images = []
    labels = []

    for class_idx, class_folder in enumerate(class_folders):
        # Load images from this class folder
        class_images = load_images_from_folder(str(class_folder))

        for img_path, img in class_images:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            images.append(gray)
            labels.append(class_idx)

        print(f"Class {class_names[class_idx]} ({class_idx}): {len(class_images)} images")

    # Keep images as list (they may have different sizes)
    # We'll resize them during feature extraction if needed
    X_images = images  # List of images (not numpy array)
    y_labels = np.array(labels)

    print(f"Total dataset: {len(X_images)} images, {len(class_names)} classes")
    if len(X_images) > 0:
        print(f"Sample image shape: {X_images[0].shape}")

    return X_images, y_labels, class_names


def extract_features_hog_skimage(
    images,
    hog_params: dict = None,
    standard_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Extract HOG features using scikit-image's implementation.

    Parameters
    ----------
    images : list or np.ndarray
        List or array of grayscale images. Images can have different sizes.
    hog_params : dict, optional
        HOG parameters. If None, uses default values:
        {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'feature_vector': True
        }
    standard_size : tuple of int, optional
        Standard size to resize images to (height, width). Default is (128, 128).
        This ensures all images produce HOG features of the same length.

    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (num_samples, feature_dim).

    Notes
    -----
    This uses a ready-made HOG implementation (allowed for Problem 3).
    Images are resized to standard_size before feature extraction to ensure
    consistent feature vector lengths across different image sizes.
    """
    if skimage_hog is None:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")

    if hog_params is None:
        hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'feature_vector': True
        }

    features = []
    for img in images:
        # Resize image to standard size to ensure consistent feature vector length
        # cv2.resize expects (width, height), so we reverse the tuple
        resized_img = cv2.resize(img, (standard_size[1], standard_size[0]))
        
        # Ensure grayscale
        if len(resized_img.shape) == 3:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # HOG can handle different image sizes, but we resize for consistency
        hog_feat = skimage_hog(resized_img, **hog_params)
        features.append(hog_feat)

    return np.array(features)


def extract_features_hog_custom(
    images,
    cell_size: Tuple[int, int] = (8, 8),
    block_size: Tuple[int, int] = (2, 2),
    num_bins: int = 9,
    standard_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Extract HOG features using the custom implementation from Problem 1.

    Parameters
    ----------
    images : list or np.ndarray
        List or array of grayscale images. Images can have different sizes.
    cell_size : tuple of int, optional
        Cell size for HOG. Default is (8, 8).
    block_size : tuple of int, optional
        Block size in cells. Default is (2, 2).
    num_bins : int, optional
        Number of orientation bins. Default is 9.
    standard_size : tuple of int, optional
        Standard size to resize images to (height, width). Default is (128, 128).
        This ensures all images produce HOG features of the same length.

    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (num_samples, feature_dim).

    Notes
    -----
    Note: This path depends on my manual HOG implementation in hog_implementation.py.
    This function will raise NotImplementedError until Problem 1 is completed.
    
    Images are resized to standard_size before feature extraction to ensure
    consistent feature vector lengths across different image sizes.
    """
    features = []
    for img in images:
        try:
            # Resize image to standard size to ensure consistent feature vector length
            # cv2.resize expects (width, height), so we reverse the tuple
            resized_img = cv2.resize(img, (standard_size[1], standard_size[0]))
            
            # Ensure grayscale
            if len(resized_img.shape) == 3:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            hog_feat = compute_hog_descriptor(
                resized_img,
                cell_size=cell_size,
                block_size=block_size,
                num_bins=num_bins
            )
            features.append(hog_feat)
        except NotImplementedError as e:
            raise NotImplementedError(
                "Custom HOG implementation not yet complete. "
                "Please implement compute_hog_descriptor in hog_implementation.py first. "
                f"Original error: {e}"
            )

    return np.array(features)


def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = "linear"
):
    """
    Train an SVM classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (num_samples, feature_dim).
    y_train : np.ndarray
        Training labels.
    C : float, optional
        Regularization parameter. Default is 1.0.
    kernel : str, optional
        Kernel type: "linear" or "rbf". Default is "linear".

    Returns
    -------
    model
        Trained SVM classifier (LinearSVC or SVC).
    """
    print(f"Training SVM classifier (C={C}, kernel={kernel})...")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    if kernel == "linear":
        model = LinearSVC(C=C, max_iter=10000, random_state=42)
    elif kernel == "rbf":
        model = SVC(C=C, kernel='rbf', gamma='scale', random_state=42)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    return model


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Evaluate a trained classifier and print metrics.

    Parameters
    ----------
    model
        Trained SVM classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.
    class_names : list of str
        List of class names for reporting.

    Notes
    -----
    This function prints:
    - Accuracy
    - Confusion matrix
    - Classification report (precision, recall, F1-score per class)
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None
) -> None:
    """
    Plot and optionally save confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : list of str
        Class names for axis labels.
    save_path : str, optional
        Path to save the figure. If None, only displays.

    Notes
    -----
    This function requires matplotlib and seaborn.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available. Skipping confusion matrix plot.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    if save_path:
        plt.close()  # Close to avoid display issues
    else:
        plt.show()


def run_classification_experiment(
    base_folder: str,
    use_custom_hog: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    kernel: str = "linear",
    save_model_path: str = None,
    plot_confusion: bool = True
) -> None:
    """
    Run a complete classification experiment.

    Parameters
    ----------
    base_folder : str
        Base folder containing class subfolders.
    use_custom_hog : bool, optional
        If True, use custom HOG implementation (requires Problem 1 to be complete).
        If False, use scikit-image HOG. Default is False.
    test_size : float, optional
        Fraction of data to use for testing. Default is 0.2.
    random_state : int, optional
        Random seed for train/test split. Default is 42.
    C : float, optional
        SVM regularization parameter. Default is 1.0.
    kernel : str, optional
        SVM kernel type. Default is "linear".
    save_model_path : str, optional
        Path to save the trained model. If None, model is not saved.
    plot_confusion : bool, optional
        If True, plot confusion matrix. Default is True.

    Notes
    -----
    This function orchestrates the full pipeline:
    1. Load dataset
    2. Extract features (skimage or custom HOG)
    3. Split into train/test
    4. Train SVM
    5. Evaluate and print metrics
    6. Optionally plot confusion matrix
    """
    # !!! Burak: set base_folder for your classification dataset (e.g., "data/training_set")
    # !!! Burak: choose whether to start with skimage HOG (easier) or your custom HOG later

    print("="*60)
    print("Classification Experiment")
    print("="*60)
    print(f"Using {'custom' if use_custom_hog else 'scikit-image'} HOG implementation")
    print()

    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    X_images, y_labels, class_names = load_dataset_from_folder(base_folder)
    print()

    # Step 2: Extract features
    print("Step 2: Extracting HOG features...")
    if use_custom_hog:
        X_features = extract_features_hog_custom(X_images)
    else:
        X_features = extract_features_hog_skimage(X_images)
    print(f"Feature extraction complete. Feature dimension: {X_features.shape[1]}")
    print()

    # Step 3: Split into train/test
    print("Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()

    # Step 4: Train classifier
    print("Step 4: Training classifier...")
    model = train_svm_classifier(X_train, y_train, C=C, kernel=kernel)
    print()

    # Step 5: Evaluate
    print("Step 5: Evaluating classifier...")
    cm = evaluate_classifier(model, X_test, y_test, class_names)

    # Step 6: Plot confusion matrix
    if plot_confusion:
        print("\nStep 6: Plotting confusion matrix...")
        save_path = None
        if save_model_path:
            # Save confusion matrix in same directory as model
            model_dir = Path(save_model_path).parent
            model_name = Path(save_model_path).stem
            cm_path = model_dir / f"{model_name}_confusion_matrix.png"
            save_path = str(cm_path)
        else:
            # Save to results folder if no model path specified
            os.makedirs("data/results/classification/", exist_ok=True)
            hog_type = "custom" if use_custom_hog else "skimage"
            save_path = f"data/results/classification/confusion_matrix_{hog_type}.png"
        plot_confusion_matrix(cm, class_names, save_path=save_path)

    # Step 7: Save model (optional)
    if save_model_path:
        model_dir = Path(save_model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_model_path)
        print(f"\nModel saved to: {save_model_path}")


def compare_hog_implementations(
    base_folder: str,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    kernel: str = "linear"
) -> None:
    """
    Compare scikit-image HOG vs custom HOG implementations.
    
    Parameters
    ----------
    base_folder : str
        Base folder containing class subfolders.
    test_size : float, optional
        Fraction of data to use for testing. Default is 0.2.
    random_state : int, optional
        Random seed for train/test split. Default is 42.
    C : float, optional
        SVM regularization parameter. Default is 1.0.
    kernel : str, optional
        SVM kernel type. Default is "linear".
    """
    print("\n" + "="*80)
    print("HOG İMPLEMENTASYONLARI KARŞILAŞTIRMASI")
    print("="*80)
    
    results = {}
    
    # Test 1: scikit-image HOG
    print("\n" + "-"*80)
    print("TEST 1: scikit-image HOG (Hazır Kütüphane)")
    print("-"*80)
    try:
        run_classification_experiment(
            base_folder=base_folder,
            use_custom_hog=False,
            test_size=test_size,
            random_state=random_state,
            C=C,
            kernel=kernel,
            save_model_path="models/classification_skimage.pkl",
            plot_confusion=False  # Save instead of showing
        )
        
        # Load results
        model = joblib.load("models/classification_skimage.pkl")
        X_images, y_labels, class_names = load_dataset_from_folder(base_folder)
        X_features = extract_features_hog_skimage(X_images)
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
        )
        skimage_accuracy = model.score(X_test, y_test)
        results['skimage'] = skimage_accuracy
        print(f"\n✓ scikit-image HOG Test Accuracy: {skimage_accuracy:.4f}")
    except Exception as e:
        print(f"\n❌ scikit-image HOG hatası: {e}")
        results['skimage'] = None
    
    # Test 2: Custom HOG
    print("\n" + "-"*80)
    print("TEST 2: Custom HOG (Problem 1 Implementasyonu)")
    print("-"*80)
    try:
        run_classification_experiment(
            base_folder=base_folder,
            use_custom_hog=True,
            test_size=test_size,
            random_state=random_state,
            C=C,
            kernel=kernel,
            save_model_path="models/classification_custom.pkl",
            plot_confusion=False  # Save instead of showing
        )
        
        # Load results
        model = joblib.load("models/classification_custom.pkl")
        X_images, y_labels, class_names = load_dataset_from_folder(base_folder)
        X_features = extract_features_hog_custom(X_images)
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
        )
        custom_accuracy = model.score(X_test, y_test)
        results['custom'] = custom_accuracy
        print(f"\n✓ Custom HOG Test Accuracy: {custom_accuracy:.4f}")
    except Exception as e:
        print(f"\n❌ Custom HOG hatası: {e}")
        import traceback
        traceback.print_exc()
        results['custom'] = None
    
    # Comparison Summary
    print("\n" + "="*80)
    print("KARŞILAŞTIRMA ÖZETİ")
    print("="*80)
    if results.get('skimage') is not None:
        print(f"scikit-image HOG Accuracy: {results['skimage']:.4f} ({results['skimage']*100:.2f}%)")
    if results.get('custom') is not None:
        print(f"Custom HOG Accuracy:       {results['custom']:.4f} ({results['custom']*100:.2f}%)")
    
    if results.get('skimage') is not None and results.get('custom') is not None:
        diff = results['custom'] - results['skimage']
        if diff > 0:
            print(f"\n✓ Custom HOG, scikit-image'den {abs(diff)*100:.2f}% daha iyi!")
        elif diff < 0:
            print(f"\n⚠ scikit-image HOG, Custom'dan {abs(diff)*100:.2f}% daha iyi.")
        else:
            print(f"\n✓ Her iki implementasyon da aynı sonucu verdi!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    """
    Main menu for Classification System (Problem 3).
    
    User can choose between:
    1. Classification with scikit-image HOG (baseline)
    2. Classification with custom HOG (Problem 1 implementation)
    3. Compare both implementations
    """
    
    print("\n" + "="*80)
    print("PROBLEM 3: SINIFLANDIRMA VE KARŞILAŞTIRMA")
    print("="*80)
    print("\nLütfen yapmak istediğiniz işlemi seçin:")
    print("\n  1. scikit-image HOG ile Sınıflandırma (Hazır Kütüphane)")
    print("     - scikit-image'in hazır HOG implementasyonu")
    print("     - Multi-class sınıflandırma")
    print("     - SVM classifier ile eğitim")
    print("\n  2. Custom HOG ile Sınıflandırma (Problem 1)")
    print("     - Kendi HOG implementasyonunuz")
    print("     - Multi-class sınıflandırma")
    print("     - SVM classifier ile eğitim")
    print("\n  3. Her İki Implementasyonu Karşılaştır")
    print("     - scikit-image vs Custom HOG")
    print("     - Performans karşılaştırması")
    print("     - Accuracy karşılaştırması")
    print("\n" + "="*80)
    
    # Check dataset structure
    base_folder = "data/training_set"
    if not os.path.exists(base_folder):
        print(f"\n⚠ Uyarı: Dataset klasörü bulunamadı: {base_folder}")
        print("Lütfen dataset klasörünü oluşturun:")
        print("  data/training_set/")
        print("    ├── class_1/  (örnek: uçak)")
        print("    ├── class_2/  (örnek: araba)")
        print("    └── class_3/  (örnek: kedi)")
        print("\nHer klasör içine o sınıfa ait görüntüleri koyun.")
        exit(1)
    
    while True:
        try:
            choice = input("\nSeçiminiz (1, 2, 3 veya çıkmak için 0): ").strip()
            
            if choice == "0":
                print("\nÇıkılıyor...")
                break
            elif choice == "1":
                # ========================================================================
                # scikit-image HOG Classification
                # ========================================================================
                print("\n" + "="*80)
                print("SINIFLANDIRMA - scikit-image HOG")
                print("="*80)
                print(f"\nDataset: {base_folder}")
                print("HOG: scikit-image (hazır kütüphane)")
                print("\nEğitim başlıyor...\n")
                
                try:
                    run_classification_experiment(
                        base_folder=base_folder,
                        use_custom_hog=False,
                        test_size=0.2,
                        random_state=42,
                        C=1.0,
                        kernel="linear",
                        save_model_path="models/classification_skimage.pkl",
                        plot_confusion=True
                    )
                    print("\n" + "="*80)
                    print("Sınıflandırma tamamlandı!")
                    print("="*80)
                except Exception as e:
                    print(f"\n❌ Hata oluştu: {e}")
                    import traceback
                    traceback.print_exc()
                break
                
            elif choice == "2":
                # ========================================================================
                # Custom HOG Classification
                # ========================================================================
                print("\n" + "="*80)
                print("SINIFLANDIRMA - Custom HOG (Problem 1)")
                print("="*80)
                print(f"\nDataset: {base_folder}")
                print("HOG: Custom implementation (Problem 1)")
                print("\nEğitim başlıyor...\n")
                
                try:
                    run_classification_experiment(
                        base_folder=base_folder,
                        use_custom_hog=True,
                        test_size=0.2,
                        random_state=42,
                        C=1.0,
                        kernel="linear",
                        save_model_path="models/classification_custom.pkl",
                        plot_confusion=True
                    )
                    print("\n" + "="*80)
                    print("Sınıflandırma tamamlandı!")
                    print("="*80)
                except Exception as e:
                    print(f"\n❌ Hata oluştu: {e}")
                    import traceback
                    traceback.print_exc()
                break
                
            elif choice == "3":
                # ========================================================================
                # Compare Both Implementations
                # ========================================================================
                print("\n" + "="*80)
                print("KARŞILAŞTIRMA - scikit-image vs Custom HOG")
                print("="*80)
                print(f"\nDataset: {base_folder}")
                print("\nHer iki implementasyon test ediliyor...")
                print("(Bu işlem biraz zaman alabilir)\n")
                
                try:
                    compare_hog_implementations(
                        base_folder=base_folder,
                        test_size=0.2,
                        random_state=42,
                        C=1.0,
                        kernel="linear"
                    )
                    print("\n" + "="*80)
                    print("Karşılaştırma tamamlandı!")
                    print("="*80)
                except Exception as e:
                    print(f"\n❌ Hata oluştu: {e}")
                    import traceback
                    traceback.print_exc()
                break
            else:
                print("\n⚠ Geçersiz seçim! Lütfen 1, 2, 3 veya 0 girin.")
                
        except KeyboardInterrupt:
            print("\n\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"\n❌ Hata oluştu: {e}")
            print("Lütfen tekrar deneyin.")
    
    print("\n" + "="*80)

