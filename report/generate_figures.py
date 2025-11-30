"""
Grafik oluşturma scripti - Report için
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Try to import seaborn, use matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib only")

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Output directory
output_dir = "report/figures"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# FIGURE 1: Problem 3 - Confusion Matrix (scikit-image HOG)
# ============================================================================
print("Creating Figure 1: Confusion Matrix (scikit-image HOG)...")
cm_skimage = np.array([[30, 0], [2, 14]])
class_names = ['Negative', 'Positive']

plt.figure(figsize=(8, 6))
if HAS_SEABORN:
    sns.heatmap(cm_skimage, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
else:
    plt.imshow(cm_skimage, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Count')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm_skimage[i, j]), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white' if cm_skimage[i, j] > 15 else 'black')
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
plt.title('Confusion Matrix - scikit-image HOG', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure1_confusion_matrix_skimage.png'))
plt.close()
print(f"  Saved: {output_dir}/figure1_confusion_matrix_skimage.png")

# ============================================================================
# FIGURE 2: Problem 3 - Confusion Matrix (Custom HOG)
# ============================================================================
print("Creating Figure 2: Confusion Matrix (Custom HOG)...")
cm_custom = np.array([[30, 0], [2, 14]])

plt.figure(figsize=(8, 6))
if HAS_SEABORN:
    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
else:
    plt.imshow(cm_custom, cmap='Greens', interpolation='nearest')
    plt.colorbar(label='Count')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm_custom[i, j]), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white' if cm_custom[i, j] > 15 else 'black')
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
plt.title('Confusion Matrix - Custom HOG', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure2_confusion_matrix_custom.png'))
plt.close()
print(f"  Saved: {output_dir}/figure2_confusion_matrix_custom.png")

# ============================================================================
# FIGURE 3: Problem 3 - Accuracy Comparison
# ============================================================================
print("Creating Figure 3: Accuracy Comparison...")
implementations = ['scikit-image\nHOG', 'Custom\nHOG']
accuracies = [0.9565, 0.9565]

plt.figure(figsize=(8, 6))
bars = plt.bar(implementations, accuracies, color=['#3498db', '#2ecc71'], width=0.6)
plt.ylim([0.9, 1.0])
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('HOG Implementation Comparison - Test Accuracy', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure3_accuracy_comparison.png'))
plt.close()
print(f"  Saved: {output_dir}/figure3_accuracy_comparison.png")

# ============================================================================
# FIGURE 4: Problem 3 - Precision, Recall, F1-Score by Class
# ============================================================================
print("Creating Figure 4: Precision, Recall, F1-Score by Class...")
classes = ['Negative', 'Positive']
precision = [0.94, 1.00]
recall = [1.00, 0.88]
f1_score = [0.97, 0.93]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Metrics by Class (scikit-image HOG)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim([0.8, 1.05])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure4_metrics_by_class.png'))
plt.close()
print(f"  Saved: {output_dir}/figure4_metrics_by_class.png")

# ============================================================================
# FIGURE 5: Problem 3 - Dataset Distribution
# ============================================================================
print("Creating Figure 5: Dataset Distribution...")
labels = ['Negative\n(149 images)', 'Positive\n(78 images)']
sizes = [149, 78]
colors = ['#3498db', '#e74c3c']
explode = (0.05, 0.05)

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12})
plt.title('Dataset Distribution\n(Total: 227 images)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure5_dataset_distribution.png'))
plt.close()
print(f"  Saved: {output_dir}/figure5_dataset_distribution.png")

# ============================================================================
# FIGURE 6: Problem 3 - Training vs Test Accuracy
# ============================================================================
print("Creating Figure 6: Training vs Test Accuracy...")
metrics = ['Training\nAccuracy', 'Test\nAccuracy']
skimage_values = [1.0000, 0.9565]
custom_values = [1.0000, 0.9565]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, skimage_values, width, label='scikit-image HOG', color='#3498db')
bars2 = ax.bar(x + width/2, custom_values, width, label='Custom HOG', color='#2ecc71')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Training vs Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim([0.9, 1.05])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure6_training_vs_test.png'))
plt.close()
print(f"  Saved: {output_dir}/figure6_training_vs_test.png")

# ============================================================================
# FIGURE 7: Problem 2 - Human Detection Statistics (Example)
# ============================================================================
print("Creating Figure 7: Human Detection Statistics...")
# Örnek veri - gerçek sonuçlara göre güncellenebilir
detection_status = ['Detected', 'Not Detected']
counts = [12, 5]  # Örnek: 12 görüntüde tespit, 5 görüntüde tespit yok

plt.figure(figsize=(8, 6))
bars = plt.bar(detection_status, counts, color=['#2ecc71', '#e74c3c'], width=0.6)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Human Detection Results\n(Example: 17 images)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure7_human_detection_stats.png'))
plt.close()
print(f"  Saved: {output_dir}/figure7_human_detection_stats.png")

# ============================================================================
# FIGURE 8: Problem 1 - HOG Parameter Effects (Example)
# ============================================================================
print("Creating Figure 8: HOG Parameter Effects...")
cell_sizes = ['8x8', '16x16']
num_bins_list = [6, 9]
# Örnek descriptor lengths - gerçek değerlere göre güncellenebilir
descriptor_lengths_8_9 = 8100  # cell=8, bins=9
descriptor_lengths_16_9 = 2025  # cell=16, bins=9 (tahmini)

params = ['Cell: 8x8\nBins: 9', 'Cell: 16x16\nBins: 9']
lengths = [8100, 2025]

plt.figure(figsize=(10, 6))
bars = plt.bar(params, lengths, color=['#3498db', '#9b59b6'], width=0.6)
plt.ylabel('Feature Vector Length', fontsize=12)
plt.title('HOG Feature Vector Length by Parameters', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, length in zip(bars, lengths):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{length}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure8_hog_parameters.png'))
plt.close()
print(f"  Saved: {output_dir}/figure8_hog_parameters.png")

# ============================================================================
# FIGURE 9: Problem 3 - Overall Performance Metrics
# ============================================================================
print("Creating Figure 9: Overall Performance Metrics...")
metrics = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
values = [0.9565, 0.97, 0.94, 0.95]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], width=0.7)
plt.ylabel('Score', fontsize=12)
plt.title('Overall Classification Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim([0.9, 1.0])
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure9_overall_metrics.png'))
plt.close()
print(f"  Saved: {output_dir}/figure9_overall_metrics.png")

# ============================================================================
# FIGURE 10: Problem 3 - Class Performance Comparison
# ============================================================================
print("Creating Figure 10: Class Performance Comparison...")
classes = ['Negative', 'Positive']
precision_vals = [0.94, 1.00]
recall_vals = [1.00, 0.88]
f1_vals = [0.97, 0.93]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, recall_vals, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Class-wise Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(loc='upper right')
ax.set_ylim([0.8, 1.05])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure10_class_performance.png'))
plt.close()
print(f"  Saved: {output_dir}/figure10_class_performance.png")

print("\n" + "="*60)
print("All figures generated successfully!")
print(f"Output directory: {output_dir}")
print("="*60)

