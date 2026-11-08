# Food-11: ResNet-50 ile Yemek Kategorisi Sınıflandırma (MATLAB)

MATLAB'de **ResNet-50 tabanlı, 2 aşamalı transfer learning** ile yemek görüntülerini **11 kategoriye** ayıran CNN sınıflandırma projesi. Yapay Sinir Ağları dersi kapsamında geliştirilmiştir.

##  Proje Özeti

| | |
|---|---|
| **Veri seti** | Food-11 (11 yemek kategorisi) |
| **Model** | ResNet-50 (ImageNet ön eğitimli), transfer learning |
| **Girdi boyutu** | 224 × 224 × 3 |
| **Eğitim / Doğrulama / Test** | XXXX / XXXX / XXXX görüntü |
| **Test doğruluğu (Accuracy)** | %XX.XX |
| **Macro F1 / Weighted F1** | %XX.XX / %XX.XX |
| **Ortalama ROC AUC** | 0.XXX |

##  Sınıflar

1. Bread
2. Dairy product
3. Dessert
4. Egg
5. Fried food
6. Meat
7. Noodles/Pasta
8. Rice
9. Seafood
10. Soup
11. Vegetable/Fruit

> Not: Klasör adlarını kendi veri setindekilerle karşılaştırıp gerekirse güncelle.

##  Yöntem

### Model mimarisi
ResNet-50'nin son sınıflandırma katmanları (`fc1000`, softmax, classification) kaldırılıp yerine şunlar eklendi:

`avg_pool → Dropout (0.4) → Fully Connected (11) → Softmax → Classification (class weights)`

### İki aşamalı transfer learning

| | Phase 1 | Phase 2 |
|---|---|---|
| **Amaç** | Yeni sınıflandırıcı başlığını eğitmek | Derin katmanları ince ayarla (fine-tuning) |
| **Dondurulan katmanlar** | Tüm backbone | `res4` ve `res5` blokları açık |
| **Optimizer** | Adam | SGDM (momentum 0.9) |
| **Öğrenme oranı** | 1e-3 | 5e-5 (her 20 epoch'ta ×0.3) |
| **Epoch** | 10 | En fazla 40 |
| **Katman LR çarpanı** | FC katmanı ×10 | FC ×1, `res5` ×0.1, `res4` ×0.01 |
| **Early stopping** | - | Validation loss 5 epoch iyileşmezse durur |

Her iki aşamada mini-batch boyutu 32'dir.

### Sınıf dengesizliği
Eğitim setindeki sınıf dağılımına göre **class weights** hesaplanıp classification layer'a verilmiştir.

### Veri artırma (Data Augmentation)
- Yatay çevirme
- Rastgele döndürme (±30°)
- Rastgele ölçekleme (0.85 – 1.15)
- Rastgele kaydırma / shear (±10°)
- HSV renk değişimi (ton, doygunluk, parlaklık, kontrast)

## Sonuçlar

Değerlendirme metrikleri: Accuracy, sınıf bazlı Precision / Recall / F1, Macro & Weighted F1, Cohen's Kappa, ROC-AUC (One-vs-Rest).

### Karışıklık Matrisi
![Confusion Matrix](results/food11_confusion_matrix_v5.png)

### Eğitim Süreci
![Training Curves](results/food11_training_curves_v5.png)

### Sınıf Bazlı Precision, Recall ve F1
![F1 Scores](results/food11_f1_scores_v5.png)

### ROC Eğrileri
![ROC Curves](results/food11_roc_curves_v5.png)

### Grad-CAM (Ağın baktığı bölgeler)
![Grad-CAM](results/food11_gradcam_analysis_v5.png)

### Diğer çıktılar
- `food11_class_distribution_v5.png`: eğitim verisi sınıf dağılımı
- `food11_perclass_v5.png`: sınıf bazlı Precision & Recall
- `food11_confidence_v5.png`: doğru ve yanlış tahminlerin güven skoru dağılımı
- `food11_worst_predictions_v5.png`: en yüksek güvenle yapılan yanlış tahminler

## Gereksinimler

- MATLAB R20XX
- Deep Learning Toolbox
- Deep Learning Toolbox Model for ResNet-50 Network (eklenti)
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox (`perfcurve`, `confusionchart`)
- *(Opsiyonel)* Parallel Computing Toolbox ve CUDA destekli GPU. GPU yoksa kod otomatik olarak CPU'da çalışır.

##  Klasör Yapısı

```
├── README.md
├── food11cnn.m          # Eğitim, değerlendirme ve grafik kodu
├── results/             # Grafikler (PNG)
└── dataset/             # (Repoda yok) Veri seti
    ├── training/
    ├── validation/
    └── evaluation/
```

##  Nasıl Çalıştırılır?

1. Repoyu klonla:
  
2. Food-11 veri setini buradan : "https://www.kaggle.com/datasets/trolukovich/food11-image-dataset?select=validation" indir. `training`, `validation` ve `evaluation` klasörlerini `food11cnn.m` ile aynı dizine koy. Her klasörün altında sınıf adlarıyla alt klasörler olmalıdır.
3. MATLAB'de `food11cnn.m` dosyasını çalıştır. Kod sırasıyla modeli eğitir, test verisinde değerlendirir, grafikleri kaydeder ve `food11_resnet50_model_v5.mat` dosyasını oluşturur.

> Eğitim süresi: GPU ile yaklaşık 3 saat (GPU modeli: 3050TI).

## Hazır Modelle Tahmin

Eğitilmiş modeli indir (`food11_resnet50_model_v5.mat`, ~80 MB),

```matlab
load('food11_resnet50_model_v5.mat');   % trainedNet, classNames, ...

img = imread('ornek.jpg');
img = imresize(img, [224 224]);
if size(img, 3) == 1, img = repmat(img, [1 1 3]); end

[label, score] = classify(trainedNet, img);
fprintf('Tahmin: %s (%.1f%%)\n', char(label), max(score) * 100);
```

##  Geliştirme Fikirleri

- Farklı mimarilerle (EfficientNet, ResNet-101) karşılaştırılabilir
- App Designer ile resim yükleyip tahmin alan arayüz eklenebilir
- Gerçek zamanlı kamera ile sınıflandırma yapılabilir
