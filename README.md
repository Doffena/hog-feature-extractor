# HOG (Histogram of Oriented Gradients) Computer Vision


Hazırlayan: Burak Avcı  

---

Bu proje, Histogram of Oriented Gradients (HOG) özellik çıkarımı, nesne tespiti ve sınıflandırma için kapsamlı bir sistem uygular. HOG algoritması sıfırdan implement edilmiş, OpenCV'nin hazır modelleri ve özel eğitimli SVM sınıflandırıcıları ile entegre edilmiştir.

## Hızlı Başlangıç

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Problem 1'i Çalıştırın (HOG Görselleştirme)
```bash
python src/hog_implementation.py
# Menüden görüntü seçin (1-5)
```

### 3. Problem 2'yi Çalıştırın (Nesne Tespiti)
```bash
python src/object_detection.py
# Menüden "1" (İnsan Tespiti) veya "2" (Özel Nesne) seçin
```

### 4. Problem 3'ü Çalıştırın (Sınıflandırma)
```bash
python src/classification.py
# Menüden "1", "2" veya "3" seçin
```

### 5. Rapor Grafiklerini Oluşturun
```bash
python report/generate_figures.py
```

## Proje Özeti

Üç ana problemden oluşmaktadır:

1. **Problem 1: HOG Özellik Çıkarımı** - HOG algoritmasının sıfırdan implementasyonu
2. **Problem 2: Nesne Tespiti** - İnsan tespiti ve özel nesne tespiti (uçak)
3. **Problem 3: Sınıflandırma ve Karşılaştırma** - HOG + SVM ile görüntü sınıflandırma

## Kurulum

### Sistem Gereksinimleri

- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows, Linux, macOS
- **RAM**: En az 4GB (önerilen: 8GB)
- **Disk Alanı**: ~500MB (veri setleri dahil)

### Adım 1: Python Kurulumu

Python'un kurulu olduğundan emin olun:
```bash
python --version
# Python 3.8 veya üzeri olmalı
```

### Adım 2: Projeyi İndirin

```bash
git clone <repository-url>
cd hog
```

veya ZIP dosyasını indirip açın.

### Adım 3: Sanal Ortam Oluşturma (Önerilen)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Adım 4: Bağımlılıkları Yükleyin

**Yöntem 1: requirements.txt kullanarak (Önerilen)**
```bash
pip install -r requirements.txt
```

**Yöntem 2: Manuel kurulum**
```bash
pip install opencv-python numpy scikit-learn scikit-image matplotlib joblib
```

### Adım 5: Kurulumu Doğrulayın

```bash
python -c "import cv2, numpy, sklearn, skimage, matplotlib, joblib; print('Tüm kütüphaneler yüklü!')"
```

Eğer hata alırsanız, eksik kütüphaneleri tek tek yükleyin.

### Proje Yapısı

```
hog/
├── README.md
├── requirements.txt
├── src/
│   ├── hog_implementation.py      # Problem 1 – HOG implementasyonu
│   ├── object_detection.py        # Problem 2 – Nesne tespiti
│   ├── classification.py          # Problem 3 – Sınıflandırma
│   └── utils.py                   # Yardımcı fonksiyonlar
├── data/
│   ├── test_images/               # Test görüntüleri
│   ├── training_set/               # Eğitim verileri
│   │   ├── positive/              # Pozitif örnekler (nesne içeren)
│   │   └── negative/              # Negatif örnekler (nesne içermeyen)
│   └── results/                   # Çıktı görüntüleri
│       ├── human_detection/        # İnsan tespiti sonuçları
│       │   ├── detected/          # Tespit edilen görüntüler
│       │   └── not_detected/      # Tespit edilmeyen görüntüler
│       ├── custom_detection/      # Özel nesne tespiti sonuçları
│       └── {image_name}/          # HOG görselleştirmeleri
│           ├── gradients/         # Gradient çıktıları (Gx, Gy, magnitude, orientation)
│           └── hog_vis/          # HOG görselleştirmeleri
├── models/
│   ├── classification_skimage.pkl    # scikit-image HOG ile eğitilmiş model
│   ├── classification_custom.pkl     # Custom HOG ile eğitilmiş model
│   └── trained_classifier.pkl        # Özel nesne tespiti modeli
├── report/
│   ├── figures/                      # Rapor grafikleri
│   ├── problem3_rapor.md             # Problem 3 detaylı raporu
│   ├── report.pdf                    # Final rapor
│   └── generate_figures.py            # Grafik oluşturma scripti
└── notebooks/
    └── analysis.ipynb                 # Jupyter notebook (opsiyonel)
```

## Kullanım ve Çalıştırma Talimatları

### Problem 1: HOG Özellik Çıkarımı

HOG algoritmasını sıfırdan uygulama ve görselleştirme.

#### Çalıştırma

```bash
python src/hog_implementation.py
```

#### Adımlar

1. **Scripti çalıştırın:**
   ```bash
   python src/hog_implementation.py
   ```

2. **Menüden görüntü seçin:**
   - 1-5 arası bir sayı girin (mevcut test görüntüleri)
   - 0 ile çıkış yapın

3. **Otomatik işlem:**
   - Seçilen görüntü yüklenir
   - Gradient çıktıları (Gx, Gy, magnitude, orientation) hesaplanır
   - HOG özellikleri çıkarılır
   - Görselleştirmeler oluşturulur

#### Çıktılar

**Gradient Çıktıları:**
- `data/results/{image_name}/gradients/`
  - `{image_name}_original.png` - Orijinal görüntü
  - `{image_name}_Gx.png` - Horizontal gradient
  - `{image_name}_Gy.png` - Vertical gradient
  - `{image_name}_magnitude.png` - Gradient magnitude
  - `{image_name}_orientation.png` - Gradient orientation
  - `{image_name}_gradients_all.png` - **Tüm gradient çıktıları tek görselde**

**HOG Görselleştirmeleri:**
- `data/results/{image_name}/hog_vis/`
  - `hog_vis_8_9.png` - Cell=8x8, Bins=9
  - `hog_vis_8_6.png` - Cell=8x8, Bins=6
  - `hog_vis_16_9.png` - Cell=16x16, Bins=9

**Karşılaştırma Görselleri:**
- `data/results/{image_name}/HOG_-_cell=(8,_8),_bins=9.png` - Orijinal ve HOG yan yana

#### Örnek Kullanım

```bash
# Terminal'de
python src/hog_implementation.py

# Çıktı:
# ==================================================
# HOG Feature Extraction - Image Selection
# ==================================================
# Available test images:
#   1. square.jpeg
#   2. circle.jpeg
#   3. triangle.png
#   4. object.png
#   5. silhouette.jpg
#   0. Exit
# 
# Select an image (0-5): 1
# 
# === Testing image: data/test_images/square.jpeg ===
#   Output directory: data\results\square/
#   
#   -> Gradient çıktıları kaydediliyor...
# [SAVED] data\results\square\gradients\square_original.png
# [SAVED] data\results\square\gradients\square_Gx.png
# [SAVED] data\results\square\gradients\square_Gy.png
# [SAVED] data\results\square\gradients\square_magnitude.png
# [SAVED] data\results\square\gradients\square_orientation.png
# [SAVED] data\results\square\gradients\square_gradients_all.png
# ...
```

### Problem 2: Nesne Tespiti

#### 2.1. İnsan Tespiti (Hazır Model)

OpenCV'nin önceden eğitilmiş HOG + SVM modeli ile insan tespiti.

##### Çalıştırma

```bash
python src/object_detection.py
```

##### Adımlar

1. **Scripti çalıştırın:**
   ```bash
   python src/object_detection.py
   ```

2. **Menüden seçim yapın:**
   - `1` - İnsan Tespiti
   - `2` - Özel Nesne Tespiti
   - `0` - Çıkış

3. **İnsan tespiti için:**
   - Otomatik olarak `data/test_images/` klasöründeki tüm görüntüler işlenir
   - Her görüntüde insan tespiti yapılır
   - Sonuçlar otomatik olarak kaydedilir

##### Özellikler

- Multi-scale detection (farklı boyutlardaki insanları tespit)
- Non-Maximum Suppression (NMS) - Çakışan tespitleri filtreler
- Confidence score gösterimi (her tespit için)
- Detaylı istatistikler (toplam, ortalama, FP, FN)

##### Çıktılar

**Klasör Yapısı:**
```
data/results/human_detection/
├── detected/          # İnsan tespit edilen görüntüler
│   ├── detected_000025.jpg
│   ├── detected_000038.jpg
│   └── ...
└── not_detected/      # İnsan tespit edilmeyen görüntüler
    ├── not_detected_000001.jpg
    ├── not_detected_000027.jpg
    └── ...
```

**Konsol Çıktısı:**
```
TESPİT EDİLENLER (12 görüntü)
TESPİT EDİLMEYENLER (5 görüntü)

ÖZET İSTATİSTİKLER:
  Toplam işlenen görüntü:     17
  İnsan tespit edilen:         12 (70.6%)
  İnsan tespit edilmeyen:      5 (29.4%)
  Ortalama confidence score:   0.85
```

#### 2.2. Özel Nesne Tespiti (Uçak)

Sliding window + HOG + SVM ile özel nesne tespiti.

##### Eğitim Verisi Hazırlama

1. **Pozitif örnekler ekleyin:**
   ```
   data/training_set/positive/
   ├── airplane_001.jpg
   ├── airplane_002.jpg
   └── ... (nesne içeren görüntüler)
   ```

2. **Negatif örnekler ekleyin:**
   ```
   data/training_set/negative/
   ├── car_001.jpg
   ├── cat_001.jpg
   └── ... (nesne içermeyen görüntüler)
   ```

##### Çalıştırma

```bash
python src/object_detection.py
# Menüden "2" seçin
```

##### Adımlar

1. **Eğitim verisi kontrolü:**
   - Sistem otomatik olarak pozitif/negatif klasörlerini kontrol eder
   - Eğer model yoksa, otomatik eğitim başlar

2. **Model eğitimi (ilk çalıştırmada):**
   - Pozitif görüntülerden HOG özellikleri çıkarılır
   - Negatif görüntülerden HOG özellikleri çıkarılır
   - Linear SVM ile model eğitilir
   - Model `models/trained_classifier.pkl` olarak kaydedilir

3. **Tespit:**
   - Test görüntüleri işlenir
   - Her görüntü için "NESNE VAR" veya "NESNE YOK" kararı verilir
   - Sonuçlar görüntü üzerine yazılır

##### Özellikler

- Otomatik model eğitimi
- Sliding window yaklaşımı (farklı boyutlarda pencere)
- Multi-scale detection
- Görüntü seviyesinde sınıflandırma
- Confidence score gösterimi

##### Çıktılar

- `data/results/custom_detection/` - Tespit sonuçları
  - Her görüntüde "UÇAK VAR" veya "UÇAK YOK" yazısı
  - Max similarity score bilgisi
- `models/trained_classifier.pkl` - Eğitilmiş model (yeniden kullanım için)

##### Örnek Kullanım

```bash
python src/object_detection.py

# Çıktı:
# ================================================================================
# NESNE TESPİT SİSTEMİ - HOG + SVM
# ================================================================================
# 
# Lütfen yapmak istediğiniz işlemi seçin:
# 
#   1. İNSAN TESPİTİ (Hazır Model)
#   2. NESNE TESPİTİ (Hazır Template Modeli)
# 
# Seçiminiz (1 veya 2, çıkmak için 0): 2
# 
# Pozitif görüntüler bulundu: 78
# Negatif görüntüler bulundu: 149
# 
# SVM modeli eğitiliyor...
# Loaded 78 positive samples
# Loaded 149 negative samples
# Training accuracy: 1.0000
# Model saved to: models/trained_classifier.pkl
# 
# Test görüntülerinde nesne tespiti yapılıyor...
# UÇAK VAR     | airplane_0000.jpg      | Max score: 0.87
# UÇAK YOK     | 000001.jpg             | Max score: 0.23
# ...
```

### Problem 3: Sınıflandırma ve Karşılaştırma

HOG özellikleri kullanarak görüntü sınıflandırma ve implementasyon karşılaştırması.

#### Dataset Hazırlama

**Klasör Yapısı:**
```
data/training_set/
├── negative/    # Sınıf 0 görüntüleri (149 görüntü)
│   ├── car_001.jpg
│   ├── cat_001.jpg
│   └── ...
└── positive/    # Sınıf 1 görüntüleri (78 görüntü)
    ├── airplane_001.jpg
    ├── airplane_002.jpg
    └── ...
```

**Not:** Her klasör en az 20-30 görüntü içermelidir (daha iyi sonuçlar için).

#### Çalıştırma

```bash
python src/classification.py
```

#### Menü Seçenekleri

1. **scikit-image HOG ile Sınıflandırma**
   - Hazır kütüphane implementasyonu
   - Hızlı ve optimize edilmiş
   - Baseline olarak kullanılır

2. **Custom HOG ile Sınıflandırma**
   - Problem 1'deki manuel implementasyon
   - scikit-image ile karşılaştırma için

3. **Her İki Implementasyonu Karşılaştır**
   - Her iki yöntemi test eder
   - Performans karşılaştırması yapar
   - Accuracy farkını gösterir

#### Adımlar

1. **Scripti çalıştırın:**
   ```bash
   python src/classification.py
   ```

2. **Menüden seçim yapın:**
   - `1` - scikit-image HOG
   - `2` - Custom HOG
   - `3` - Karşılaştırma
   - `0` - Çıkış

3. **Otomatik işlem:**
   - Dataset yüklenir
   - HOG özellikleri çıkarılır
   - Train/test split yapılır (80/20)
   - SVM modeli eğitilir
   - Test setinde değerlendirme yapılır
   - Sonuçlar kaydedilir

#### Sonuçlar

**Performans Metrikleri:**
- **Test Accuracy**: 95.65% (44/46 doğru)
- **Training Accuracy**: 100.00% (181/181 doğru)
- **Feature Dimension**: 8100

**Confusion Matrix:**
```
                Predicted
              Negative  Positive
Actual Negative   30       0
       Positive    2      14
```

**Sınıf Bazlı Metrikler:**
- **Negative**: Precision=0.94, Recall=1.00, F1=0.97
- **Positive**: Precision=1.00, Recall=0.88, F1=0.93

#### Çıktılar

**Model Dosyaları:**
- `models/classification_skimage.pkl` - scikit-image HOG modeli
- `models/classification_custom.pkl` - Custom HOG modeli

**Görselleştirmeler:**
- `models/classification_skimage_confusion_matrix.png` - Confusion matrix
- `models/classification_custom_confusion_matrix.png` - Confusion matrix

**Konsol Çıktısı:**
```
Step 1: Loading dataset...
Found 2 classes: ['negative', 'positive']
Class negative (0): 149 images
Class positive (1): 78 images

Step 2: Extracting HOG features...
Feature extraction complete. Feature dimension: 8100

Step 3: Splitting into train/test sets...
Training set: 181 samples
Test set: 46 samples

Step 4: Training classifier...
Training accuracy: 1.0000

Step 5: Evaluating classifier...
Test Accuracy: 0.9565

Confusion Matrix:
[[30  0]
 [ 2 14]]

Classification Report:
              precision    recall  f1-score   support
    negative       0.94      1.00      0.97        30
    positive       1.00      0.88      0.93        16
```

#### Önemli Notlar

- **İlk çalıştırmada:** Model eğitimi birkaç dakika sürebilir
- **Sonraki çalıştırmalarda:** Eğitilmiş model varsa direkt kullanılır
- **Karşılaştırma:** Her iki implementasyon da aynı sonuçları verir (95.65% accuracy)

## Rapor Grafikleri

Rapor için profesyonel grafikler oluşturma.

### Çalıştırma

```bash
python report/generate_figures.py
```

### Oluşturulan Grafikler

1. **Figure 1**: Confusion Matrix (scikit-image HOG)
2. **Figure 2**: Confusion Matrix (Custom HOG)
3. **Figure 3**: Accuracy Comparison (Bar Chart)
4. **Figure 4**: Precision, Recall, F1-Score by Class
5. **Figure 5**: Dataset Distribution (Pie Chart)
6. **Figure 6**: Training vs Test Accuracy
7. **Figure 7**: Human Detection Statistics
8. **Figure 8**: HOG Parameter Effects
9. **Figure 9**: Overall Performance Metrics
10. **Figure 10**: Class Performance Comparison

### Çıktı

Tüm grafikler `report/figures/` klasörüne kaydedilir:
- **Format**: PNG
- **Çözünürlük**: 300 DPI (yüksek kalite)
- **Boyut**: Optimize edilmiş (rapor için uygun)

### Grafikleri Raporunuza Ekleme

Grafikler LaTeX/Overleaf ile uyumludur:
```latex
\includegraphics[width=0.8\textwidth]{report/figures/figure1_confusion_matrix_skimage.png}
```

## Özellikler

### Problem 1 Özellikleri
- Gradyan hesaplama (Gx, Gy, magnitude, orientation)
- Hücre bazlı histogram oluşturma
- Blok normalizasyonu (L2-Hys)
- HOG descriptor çıkarımı
- HOG görselleştirme
- Farklı parametrelerle test (cell size, num bins)
- Gradient çıktılarını tek görselde birleştirme

### Problem 2 Özellikleri
- OpenCV hazır HOG + SVM insan dedektörü
- Multi-scale detection
- Non-Maximum Suppression (NMS)
- Confidence score gösterimi
- Özel nesne tespiti (SVM eğitimi)
- Sliding window yaklaşımı
- Görüntü seviyesinde sınıflandırma
- Detaylı istatistikler ve analiz
### Problem 3 Özellikleri
- scikit-image HOG implementasyonu
- Custom HOG implementasyonu
- Linear SVM sınıflandırıcı
- Performans karşılaştırması
- Confusion matrix analizi
- Precision, Recall, F1-score metrikleri
- Otomatik model kaydetme/yükleme

## Sonuçlar

### Problem 3 - Sınıflandırma Sonuçları

**Test Accuracy:** 95.65% (44/46 doğru sınıflandırma)

**Confusion Matrix:**
```
                Predicted
              Negative  Positive
Actual Negative   30       0
       Positive    2      14
```

**Sınıf Bazlı Metrikler:**
- **Negative Class**: Precision=0.94, Recall=1.00, F1=0.97
- **Positive Class**: Precision=1.00, Recall=0.88, F1=0.93

**Önemli Bulgu:** Custom HOG implementasyonu, scikit-image HOG ile **tamamen aynı sonuçları** vermiştir (95.65% accuracy). Bu, implementasyonun doğruluğunu kanıtlamaktadır.

## Teknik Detaylar

### HOG Parametreleri
- **Cell Size**: 8x8 piksel
- **Block Size**: 2x2 hücre
- **Orientation Bins**: 9
- **Block Normalization**: L2-Hys
- **Feature Vector Length**: 8100 özellik (128x128 görüntü için)

### SVM Parametreleri
- **Kernel**: Linear
- **Regularization (C)**: 1.0
- **Max Iterations**: 10000

### Preprocessing
- Görüntüler 128x128 piksele resize edilir
- Grayscale dönüşümü yapılır
- Feature vektörleri normalize edilir

##  Veri Organizasyonu

### Test Görüntüleri
Test görüntülerini `data/test_images/` klasörüne yerleştirin:
- `.jpg`, `.jpeg`, `.png` formatları desteklenir

### Eğitim Verisi (Problem 2 - Özel Nesne Tespiti)
```
data/training_set/
├── positive/    # Nesne içeren görüntüler (örn: uçak)
└── negative/    # Nesne içermeyen görüntüler
```

### Eğitim Verisi (Problem 3 - Sınıflandırma)
```
data/training_set/
├── negative/    # Sınıf 0 görüntüleri
└── positive/    # Sınıf 1 görüntüleri
```

## Sorun Giderme

### Import Hataları
Eğer relative import hataları alırsanız:
```bash
# Direkt script olarak çalıştırın
python src/hog_implementation.py
python src/object_detection.py
python src/classification.py
```

### Model Bulunamadı
Eğer eğitilmiş model bulunamazsa:
- Problem 2: Önce modeli eğitin (menüden seçenek 2)
- Problem 3: Önce sınıflandırma deneyini çalıştırın

### Görüntü Yükleme Hataları
- Görüntü formatlarını kontrol edin (.jpg, .jpeg, .png)
- Dosya yollarının doğru olduğundan emin olun
- Görüntülerin bozuk olmadığını kontrol edin

##  Notlar

- Tüm kod Türkçe yorumlar içerir
- Menü sistemi kullanıcı dostu arayüz sağlar
- Çıktılar otomatik olarak organize edilir
- Detaylı istatistikler ve analizler sağlanır

##  Referanslar

1. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.
2. OpenCV Documentation: HOGDescriptor
3. Scikit-image Documentation: HOG Feature Extraction
4. Scikit-learn Documentation: SVM Classifier

##  Yazar

**Burak Avcı**  
Mail:burakavci0206@gmail.com
Ostim Teknik Üniversitesi - Yapay Zeka Mühendisliği

---
