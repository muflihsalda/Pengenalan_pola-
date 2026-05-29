import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris 
from sklearn.model_selection import StratifiedKFold 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 


# ==========================================
# 1. IMPLEMENTASI KNN DARI NOL (CUSTOM)
# ==========================================
class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        # Hitung jarak Euclidean antara data uji dengan semua data latih
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Ambil indeks dari k-tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]
        
        # Ambil label dari k-tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Voting mayoritas untuk menentukan kelas target
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



# ==========================================
# 2. PREPARASI DATASET & KONFIGURASI (KAGGLE DIABETES)
# ==========================================


# 1. Membaca file diabetes.csv dari Kaggle
# Pastikan file 'diabetes.csv' sudah ditaruh di folder yang sama dengan file .py kamu
df = pd.read_csv('diabetes.csv')

# 2. Memisahkan Fitur (X) dan Target/Label (y)
# Dataset diabetes tidak memiliki kolom ID, jadi bisa langsung dipisahkan
X = df.iloc[:, :-1].values  # Mengambil semua kolom fitur medis (Glukosa, Tekanan Darah, dll)
y = df.iloc[:, -1].values   # Mengambil kolom terakhir ('Outcome': 0 atau 1) sebagai target

# Daftar nilai K yang dievaluasi
k_values = [1, 3, 5, 7, 10, 15]
n_splits = 5

# Setup 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Dictionary untuk menyimpan hasil akurasi rata-rata
results_scratch = {}
results_sklearn = {}

# ==========================================
# 3. PROSES EVALUASI (5-FOLD CROSS-VALIDATION)
# ==========================================
print("=== Memulai Evaluasi 5-Fold Cross Validation ===")

for k in k_values:
    acc_scratch_folds = []
    acc_sklearn_folds = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- Model KNN dari Nol ---
        knn_scratch = KNNFromScratch(k=k)
        knn_scratch.fit(X_train, y_train)
        preds_scratch = knn_scratch.predict(X_test)
        acc_scratch_folds.append(accuracy_score(y_test, preds_scratch))
        
        # --- Model KNN dari Sklearn ---
        knn_sklearn = KNeighborsClassifier(n_neighbors=k)
        knn_sklearn.fit(X_train, y_train)
        preds_sklearn = knn_sklearn.predict(X_test)
        acc_sklearn_folds.append(accuracy_score(y_test, preds_sklearn))
        
    # Hitung rata-rata akurasi dari 5-fold untuk nilai K ini
    results_scratch[k] = np.mean(acc_scratch_folds)
    results_sklearn[k] = np.mean(acc_sklearn_folds)

# ==========================================
# 4. MENAMPILKAN HASIL PERBANDINGAN Teks
# ==========================================
print(f"\n{'Nilai K':<10} | {'Akurasi KNN dari Nol':<22} | {'Akurasi Sklearn':<18}")
print("-" * 58)
for k in k_values:
    print(f"{k:<10} | {results_scratch[k]:<22.4f} | {results_sklearn[k]:<18.4f}")


# ==========================================
# 5. MEMBUAT DAN MENAMPILKAN GRAFIK
# ==========================================
# Mengambil nilai akurasi dari dictionary hasil kalkulasi di atas
akurasi_dari_nol = list(results_scratch.values())
akurasi_sklearn = list(results_sklearn.values())

# Membuat visualisasi grafik plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, akurasi_dari_nol, marker='o', label='KNN dari Nol', color='blue', linestyle='-')
plt.plot(k_values, akurasi_sklearn, marker='s', label='Sklearn KNN', color='orange', linestyle='--')

# Memberikan label, judul, dan grid pada grafik
plt.title('Perbandingan Akurasi KNN (5-Fold Cross Validation)')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.xticks(k_values)  # Memastikan semua nilai K muncul di sumbu X
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Menyimpan grafik otomatis menjadi file gambar .png
plt.savefig('grafik_perbandingan_knn.png', dpi=300, bbox_inches='tight')

# Menampilkan grafik di layar komputer
plt.show()