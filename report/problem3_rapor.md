# Problem 3: Sınıflandırma ve Karşılaştırma Raporu

## 1. Giriş ve Amaç

Bu problemde, HOG (Histogram of Oriented Gradients) özelliklerini kullanarak bir görüntü sınıflandırma sistemi geliştirilmiş ve farklı HOG implementasyonları karşılaştırılmıştır. Sistem, scikit-image'in hazır HOG implementasyonu ile Problem 1'de geliştirilen custom HOG implementasyonunu kullanarak multi-class sınıflandırma yapmaktadır.

### 1.1. Problem Kapsamı

- **Amaç**: HOG özellikleri kullanarak görüntü sınıflandırma
- **Yöntem**: Support Vector Machine (SVM) classifier
- **Karşılaştırma**: scikit-image HOG vs Custom HOG implementasyonu
- **Değerlendirme**: Accuracy, Precision, Recall, F1-score metrikleri

## 2. Metodoloji

### 2.1. Dataset

**Dataset Yapısı:**
- **Toplam Görüntü Sayısı**: 227 görüntü
- **Sınıf Sayısı**: 2 sınıf (binary classification)
  - **negative**: 149 görüntü (65.6%)
  - **positive**: 78 görüntü (34.4%)
- **Train/Test Split**: 80% train (181 görüntü), 20% test (46 görüntü)
- **Görüntü Boyutları**: Farklı boyutlarda (örnek: 332x500 piksel)
- **Preprocessing**: Tüm görüntüler 128x128 piksele resize edilmiştir

### 2.2. HOG Özellik Çıkarımı

**HOG Parametreleri:**
- **Cell Size**: 8x8 piksel
- **Block Size**: 2x2 hücre
- **Orientation Bins**: 9
- **Block Normalization**: L2-Hys
- **Feature Vector Length**: 8100 özellik

**İki Farklı Implementasyon:**
1. **scikit-image HOG**: Hazır kütüphane implementasyonu
2. **Custom HOG**: Problem 1'de manuel olarak geliştirilen implementasyon

### 2.3. Sınıflandırıcı

**SVM (Support Vector Machine) Parametreleri:**
- **Kernel**: Linear
- **Regularization Parameter (C)**: 1.0
- **Random State**: 42 (reproducibility için)
- **Max Iterations**: 10000

## 3. Deney Sonuçları

### 3.1. scikit-image HOG Sonuçları

**Eğitim Sonuçları:**
- **Training Accuracy**: 100.00% (181/181 doğru sınıflandırma)
- **Training Set Size**: 181 görüntü
- **Feature Dimension**: 8100

**Test Sonuçları:**
- **Test Accuracy**: **95.65%** (44/46 doğru sınıflandırma)
- **Test Set Size**: 46 görüntü

**Confusion Matrix:**
```
                Predicted
              Negative  Positive
Actual Negative   30       0
       Positive    2      14
```

**Detaylı Metrikler:**
- **Negative Class:**
  - Precision: 0.94
  - Recall: 1.00
  - F1-score: 0.97
  - Support: 30

- **Positive Class:**
  - Precision: 1.00
  - Recall: 0.88
  - F1-score: 0.93
  - Support: 16

- **Overall:**
  - Accuracy: 0.9565
  - Macro Average: Precision=0.97, Recall=0.94, F1=0.95
  - Weighted Average: Precision=0.96, Recall=0.96, F1=0.96

### 3.2. Custom HOG Sonuçları

**Eğitim Sonuçları:**
- **Training Accuracy**: 100.00% (181/181 doğru sınıflandırma)
- **Training Set Size**: 181 görüntü
- **Feature Dimension**: 8100

**Test Sonuçları:**
- **Test Accuracy**: **95.65%** (44/46 doğru sınıflandırma)
- **Test Set Size**: 46 görüntü

**Confusion Matrix:**
```
                Predicted
              Negative  Positive
Actual Negative   30       0
       Positive    2      14
```

**Detaylı Metrikler:**
- **Negative Class:**
  - Precision: 0.94
  - Recall: 1.00
  - F1-score: 0.97
  - Support: 30

- **Positive Class:**
  - Precision: 1.00
  - Recall: 0.88
  - F1-score: 0.93
  - Support: 16

- **Overall:**
  - Accuracy: 0.9565
  - Macro Average: Precision=0.97, Recall=0.94, F1=0.95
  - Weighted Average: Precision=0.96, Recall=0.96, F1=0.96

## 4. Karşılaştırma ve Analiz

### 4.1. Implementasyon Karşılaştırması

**Önemli Bulgu**: Her iki HOG implementasyonu da **tamamen aynı sonuçları** vermiştir!

| Metrik | scikit-image HOG | Custom HOG | Fark |
|--------|------------------|------------|------|
| Test Accuracy | 95.65% | 95.65% | 0.00% |
| Training Accuracy | 100.00% | 100.00% | 0.00% |
| Feature Dimension | 8100 | 8100 | 0 |
| Confusion Matrix | [[30,0],[2,14]] | [[30,0],[2,14]] | Aynı |

### 4.2. Sonuç Analizi

**Başarılı Yönler:**
1. **Yüksek Accuracy**: %95.65 test accuracy ile başarılı bir sınıflandırma
2. **Mükemmel Training Performance**: %100 training accuracy
3. **Implementasyon Doğruluğu**: Custom HOG implementasyonu, scikit-image ile aynı sonuçları vererek doğruluğunu kanıtlamıştır
4. **Dengeli Metrikler**: Precision, Recall ve F1-score değerleri dengeli

**Hata Analizi:**
- **Toplam Hata**: 2 false negative (positive sınıfı negative olarak yanlış sınıflandırılmış)
- **False Positive**: 0 (negative sınıfı mükemmel şekilde sınıflandırılmış)
- **False Negative**: 2 (positive sınıfından 2 görüntü negative olarak sınıflandırılmış)

**Sınıf Bazlı Performans:**
- **Negative Class**: Mükemmel performans (Recall=1.00, Precision=0.94)
- **Positive Class**: İyi performans ancak biraz daha düşük recall (Recall=0.88, Precision=1.00)

### 4.3. Dataset Dengesizliği

**Gözlem:**
- Negative sınıfı: 149 görüntü (65.6%)
- Positive sınıfı: 78 görüntü (34.4%)
- **Dengesiz dataset** (2:1 oranı)

**Etkisi:**
- Positive sınıfının recall değeri (0.88) negative sınıfından (1.00) daha düşük
- Bu, dataset dengesizliğinden kaynaklanıyor olabilir
- Ancak genel accuracy (%95.65) yüksek kalmıştır

## 5. Teknik Detaylar

### 5.1. Preprocessing

**Görüntü Standardizasyonu:**
- Tüm görüntüler 128x128 piksele resize edilmiştir
- Bu, farklı boyutlardaki görüntülerden aynı uzunlukta HOG feature vektörleri elde etmek için gereklidir
- Grayscale dönüşümü yapılmıştır

**HOG Feature Extraction:**
- Her görüntü için 8100 boyutlu feature vektörü çıkarılmıştır
- Feature extraction süreci:
  1. Gradient hesaplama
  2. Cell histogram oluşturma
  3. Block normalization
  4. Feature vektörü oluşturma

### 5.2. Model Eğitimi

**SVM Eğitimi:**
- Linear kernel kullanılmıştır (non-linear veri için RBF kernel alternatif olabilir)
- C=1.0 regularization parametresi ile overfitting kontrol edilmiştir
- Stratified split kullanılarak sınıf dağılımı korunmuştur

### 5.3. Değerlendirme Metrikleri

**Kullanılan Metrikler:**
- **Accuracy**: Genel doğruluk oranı
- **Precision**: Pozitif tahminlerin doğruluğu
- **Recall**: Gerçek pozitiflerin ne kadarının yakalandığı
- **F1-score**: Precision ve Recall'un harmonik ortalaması
- **Confusion Matrix**: Detaylı hata analizi

## 6. Sonuç ve Değerlendirme

### 6.1. Ana Bulgular

1. **Implementasyon Doğruluğu**: Custom HOG implementasyonu, scikit-image ile tamamen aynı sonuçları vermiştir. Bu, Problem 1'deki implementasyonun doğruluğunu kanıtlamaktadır.

2. **Yüksek Performans**: %95.65 test accuracy ile başarılı bir sınıflandırma gerçekleştirilmiştir.

3. **Dengeli Metrikler**: Precision, Recall ve F1-score değerleri dengeli ve yüksektir.

4. **Sınıf Performansı**: Negative sınıfı mükemmel performans gösterirken, positive sınıfı biraz daha düşük recall'a sahiptir.

### 6.2. Öneriler

**İyileştirme Önerileri:**
1. **Dataset Dengesizliği**: Positive sınıfı için daha fazla görüntü eklenebilir veya data augmentation uygulanabilir
2. **Hyperparameter Tuning**: C parametresi ve kernel tipi için grid search yapılabilir
3. **Feature Engineering**: Farklı HOG parametreleri (cell size, block size) denenebilir
4. **Ensemble Methods**: Birden fazla modelin kombinasyonu denenebilir

**Gelecek Çalışmalar:**
- Multi-class classification için daha fazla sınıf eklenebilir
- Farklı feature extraction yöntemleri (SIFT, SURF) karşılaştırılabilir
- Deep learning tabanlı yöntemlerle karşılaştırma yapılabilir

### 6.3. Genel Değerlendirme

Problem 3 başarıyla tamamlanmıştır. Hem scikit-image hem de custom HOG implementasyonları kullanılarak yüksek performanslı bir sınıflandırma sistemi geliştirilmiştir. Custom HOG implementasyonunun scikit-image ile aynı sonuçları vermesi, implementasyonun doğruluğunu ve güvenilirliğini göstermektedir.

**Başarı Oranı**: %95.65 test accuracy ile başarılı bir sonuç elde edilmiştir.

---

## Ekler

### Ek A: Çıktı Dosyaları

**Model Dosyaları:**
- `models/classification_skimage.pkl` - scikit-image HOG ile eğitilmiş model
- `models/classification_custom.pkl` - Custom HOG ile eğitilmiş model

**Confusion Matrix Görselleri:**
- `models/classification_skimage_confusion_matrix.png`
- `models/classification_custom_confusion_matrix.png`

### Ek B: Kod Yapısı

**Ana Fonksiyonlar:**
- `load_dataset_from_folder()` - Dataset yükleme
- `extract_features_hog_skimage()` - scikit-image HOG feature extraction
- `extract_features_hog_custom()` - Custom HOG feature extraction
- `train_svm_classifier()` - SVM eğitimi
- `evaluate_classifier()` - Model değerlendirme
- `plot_confusion_matrix()` - Confusion matrix görselleştirme
- `run_classification_experiment()` - Tam pipeline
- `compare_hog_implementations()` - Karşılaştırma fonksiyonu

---

**Rapor Tarihi**: 2024  
**Hazırlayan**: Burak Avcı  
**Problem**: Problem 3 - Sınıflandırma ve Karşılaştırma

