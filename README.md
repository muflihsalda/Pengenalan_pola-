# Pengenalan_pola

## Tugas_Pengolahan_citra

## Penjelasan Tugas 1: Implementasi KNN dari Nol (Fundamental)

Tugas ini bertujuan untuk memahami fondasi matematika dari algoritma K-Nearest Neighbors (KNN) dengan membangunnya secara manual tanpa bantuan library Machine Learning, kemudian memvalidasi hasilnya menggunakan library standar industri.

### Metodologi dan Alur Kerja
* **Dataset:** Menggunakan Diabetes Dataset yang diperoleh dari Kaggle (`diabetes.csv`).
* **Implementasi Custom:** Membangun class `KNNFromScratch` dengan menghitung jarak Euclidean secara manual antara data uji dan seluruh data latih menggunakan formula:
  
  $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

  Dilanjutkan dengan pengambilan indeks tetangga terdekat berdasarkan urutan jarak terkecil, dan melakukan voting mayoritas menggunakan metode `Counter`.
* **Validasi Sklearn:** Menggunakan `KNeighborsClassifier` dari library `scikit-learn` sebagai baseline pembanding untuk memastikan akurasi model custom sudah tepat.
* **Proces Evaluasi:** Menerapkan metode Stratified 5-Fold Cross Validation untuk membagi data secara adil dan menghindari bias. Pengujian dilakukan pada variasi nilai K, yaitu 1, 3, 5, 7, 10, dan 15.

### Hasil dan Analisis
Kedua model menghasilkan performa akurasi yang hampir berimpit di setiap nilai K, membuktikan bahwa logika matematika pada KNN kustom berhasil meniru performa standar Scikit-Learn. Secara umum, akurasi model meningkat seiring bertambahnya nilai K pada dataset diabetes ini, di mana akurasi optimal dicapai pada K=15 dengan persentase sekitar 75%.

<img width="2100" height="1407" alt="grafik_perbandingan_knn" src="https://github.com/user-attachments/assets/3cf27f04-efaf-409f-b327-e3786a14cab0" />

<img width="370" height="154" alt="5_fold_cross_validation" src="https://github.com/user-attachments/assets/1ea60c7a-272a-49ba-915b-bc6d9ca8aba8" />

---

## Penjelasan Tugas 2: Perbandingan Fitur untuk Klasifikasi Tekstur (Menengah)

Tugas ini berfokus pada tahap ekstraksi fitur citra untuk mengenali pola tekstur. Eksperimen dilakukan dengan membandingkan tiga metode ekstraksi fitur yang berbeda terhadap dua algoritma klasifikasi populer.

### Metodologi dan Alur Kerja
* **Dataset:** Representasi karakteristik visual dari DTD (Describable Textures Dataset) untuk mengklasifikasikan 3 jenis kelas tekstur visual yang berbeda.
* **Metode Ekstraksi Fitur:**
  1. *Local Binary Pattern (LBP):* Mengekstraksi informasi tekstur lokal dengan membandingkan nilai piksel pusat terhadap piksel tetangganya dalam bentuk biner.
  2. *Histogram of Oriented Gradients (HOG):* Menangkap informasi struktur dan bentuk objek berdasarkan distribusi arah gradien intensitas cahaya.
  3. *Gray-Level Co-occurrence Matrix (GLCM):* Mengekstraksi fitur statistik tekstur (seperti kontras, korelasi, energi, dan homogenitas) berdasarkan hubungan spasial antar piksel.
* **Algoritma Klasifikasi:** Fitur yang telah diekstrak kemudian diuji menggunakan dua jenis classifier dari Scikit-Learn, yaitu Support Vector Machine (SVM) dengan kernel RBF dan Random Forest Classifier (100 estimators).

### Hasil dan Analisis
Berdasarkan hasil pengujian, fitur HOG memberikan performa akurasi tertinggi pada kedua classifier (mencapai kisaran 55% - 58%) dibandingkan fitur LBP dan GLCM. Hal ini mengindikasikan bahwa distribusi arah gradien objek jauh lebih informatif dalam mengenali pola tekstur dataset DTD ini. Dari sisi classifier, Random Forest memberikan akurasi yang sedikit lebih stabil pada fitur LBP dan HOG.

<img width="2700" height="1500" alt="grafik_perbandingan_tekstur_tugas2" src="https://github.com/user-attachments/assets/729ced42-6945-4ade-a535-a9d975d60a2c" />

<img width="499" height="91" alt="Terminal_Klasifikasi_Tekstur" src="https://github.com/user-attachments/assets/e06b91b2-0740-4538-9cb5-8d1b412d0b54" />

---

## Penjelasan Tugas 3: Transfer Learning untuk Dataset Kustom (Lanjutan)

Tugas terakhir ini menerapkan paradigma Deep Learning untuk menyelesaikan masalah klasifikasi citra pada dataset kustom berskala kecil menggunakan teknik Transfer Learning.

### Metodologi dan Alur Kerja
* **Dataset:** Menggunakan dataset citra kustom yang terdiri dari 5 kelas berbeda dengan total 500 gambar (masing-masing kelas memiliki minimal 100 sampel gambar).
* **Arsitektur Model yang Dibandingkan:**
  1. *Transfer Learning (MobileNetV2):* Menggunakan arsitektur MobileNetV2 yang bobotnya (*weights*) telah dilatih menggunakan dataset raksasa ImageNet. Lapisan atas (*top layer*) diganti dan disesuaikan untuk mengenali 5 kelas baru, kemudian dilatih selama 10 epoch.
  2. *CNN dari Nol (Scratch):* Membangun arsitektur Convolutional Neural Network kustom sendiri dari awal (terdiri dari lapisan Convolutional, Pooling, dan Dense) untuk dilatih pada dataset yang sama sebagai pembanding baseline.

### Hasil dan Analisis
Model Transfer Learning menggunakan MobileNetV2 melesat cepat dengan akurasi akhir mencapai 96% pada epoch ke-10. Hal ini terjadi karena MobileNetV2 sudah memiliki kemampuan mengekstrak fitur visual dasar (seperti garis, tepi, dan bentuk) dari training ImageNet sebelumnya. Sebaliknya, model CNN dari Nol mengalami kendala konvergensi dan hanya mentok di akurasi 65% karena jumlah data latih (100 gambar per kelas) terlalu sedikit bagi model deep learning untuk belajar dari awal, sehingga rawan memicu kesalahan klasifikasi pada objek dengan kemiripan visual yang tinggi.

<img width="2306" height="1407" alt="grafik_transfer_learning_tugas3" src="https://github.com/user-attachments/assets/adf41a39-f2b5-4ee1-bd7a-4e86e8ed58ea" />


<img width="467" height="421" alt="terminal_grafik_transfer_learning_tugas3" src="https://github.com/user-attachments/assets/5a060907-bb82-4729-9633-b98196733938" />
