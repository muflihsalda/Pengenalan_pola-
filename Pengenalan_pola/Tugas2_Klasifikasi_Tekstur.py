import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SIMULASI EKSTRAKSI FITUR TEKSTUR (DTD DATASET)
# ==========================================
# Perbaikan: Menggunakan penambahan berbasis kolom yang sesuai dimensi fitur masing-masing
np.random.seed(42)
n_samples = 300  # Total sampel gambar

# Membuat label kelas tekstur (misal: 'banded', 'bumpy', 'woven')
y = np.random.choice([0, 1, 2], size=n_samples)

# Fitur 1: LBP (Dimensi 10)
X_lbp = np.random.randn(n_samples, 10)
X_lbp[:, :3] += np.eye(3)[y] * 0.5  # Ditambahkan hanya ke 3 kolom pertama agar sinkron

# Fitur 2: HOG (Dimensi 24)
X_hog = np.random.randn(n_samples, 24)
X_hog[:, :3] += np.eye(3)[y] * 0.8

# Fitur 3: GLCM (Dimensi 6)
X_glcm = np.random.randn(n_samples, 6)
X_glcm[:, :3] += np.eye(3)[y] * 0.3

# List metode ekstraksi dan classifier sesuai instruksi soal
fitur_metode = {'LBP': X_lbp, 'HOG': X_hog, 'GLCM': X_glcm}
classifiers = {
    'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# ==========================================
# 2. PROSES PELATIHAN & EVALUASI PERFORMA
# ==========================================
# ... (ke bawahnya tetap sama dengan kode Tugas 2 yang pertama) ...
hasil_akurasi = {clf_name: [] for clf_name in classifiers.keys()}
fitur_names = list(fitur_metode.keys())

for clf_name, clf in classifiers.items():
    for fitur_name, X_data in fitur_metode.items():
        # Split data 80% Latih, 20% Uji
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
        
        # Training model
        clf.fit(X_train, y_train)
        
        # Prediksi dan hitung akurasi
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        hasil_akurasi[clf_name].append(acc)

# ==========================================
# 3. MENAMPILKAN TABEL HASIL DI TERMINAL
# ==========================================
print("=== HASIL EVALUASI TUGAS 2: KLASIFIKASI TEKSTUR ===")
df_hasil = pd.DataFrame(hasil_akurasi, index=fitur_names)
print(df_hasil)

# ==========================================
# 4. VISUALISASI GRAFIK PERBANDINGAN (BATANG)
# ==========================================
x = np.arange(len(fitur_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, df_hasil['SVM'], width, label='SVM', color='#4A90E2')
rects2 = ax.bar(x + width/2, df_hasil['Random Forest'], width, label='Random Forest', color='#F5A623')

# Dekorasi Grafik
ax.set_ylabel('Tingkat Akurasi')
ax.set_title('Perbandingan Fitur Tekstur (LBP vs HOG vs GLCM) dan Classifier')
ax.set_xticks(x)
ax.set_xticklabels(fitur_names)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()

# Menambahkan angka akurasi di atas bar batang
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('grafik_perbandingan_tekstur_tugas2.png', dpi=300)
plt.show()