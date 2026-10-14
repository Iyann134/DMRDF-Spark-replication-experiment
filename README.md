# Replikasi DMRDF pada Apache Spark — Dataset Flipkart E-Commerce
> **Mata Kuliah:** Mahadata / Big Data (IF25-40402) · Institut Teknologi Sumatera (ITERA)  
> **Jurnal Acuan:** Narayanan, S., Samuel, P., & Chacko, M. (2020). *Improving prediction with enhanced Distributed Memory‑based Resilient Dataset Filter*. Journal of Big Data, 7, Article 13. https://doi.org/10.1186/s40537-020-00292-y

---

## Deskripsi Proyek

Repositori ini berisi **replikasi parsial** dan **adaptasi dataset** dari metode yang diusulkan oleh Narayanan et al. (2020) pada jurnal *Journal of Big Data* (Springer). Eksperimen ini mengimplementasikan:

1. **Feature Information Gain (FIG)** — Seleksi fitur signifikan berbasis entropi informasi dari data ulasan produk.
2. **Distributed Memory-based Resilient Dataset Filter (DMRDF)** — Reduksi redundansi ulasan terdistribusi menggunakan Apache Spark RDD (Resilient Distributed Dataset) melalui operasi `map()`, `filter()`, dan `reduceByKey()`.
3. **Klasifikasi Prediksi Pre-launch Produk** menggunakan Spark MLlib:
   - Logistic Regression
   - Support Vector Machine (LinearSVC)
   - Decision Tree Classifier

**Dataset yang Digunakan:** Flipkart E-Commerce Product Reviews (Kaggle: `PromptCloudHQ/flipkart-products`), 20.000 produk, ~38 MB.

---

## Struktur Direktori

```
Mahadata/
├── data/
│   └── flipkart_com-ecommerce_sample.csv   # Dataset utama Flipkart
├── src/
│   ├── __init__.py
│   ├── spark_session.py    # Inisialisasi SparkSession terkontrol
│   ├── preprocessing.py    # Data cleaning & Feature extraction pipeline
│   ├── dmrdf_filter.py     # Implementasi DMRDF (Spark RDD)
│   └── modeling.py         # Training & Evaluasi model MLlib
├── results/
│   ├── metrics_comparison.csv   # Tabel metrik hasil eksperimen
│   └── benchmark_plot.png       # Grafik perbandingan kinerja
├── DMRDF_Flipkart_Experiment.ipynb  # Jupyter Notebook lengkap (runnable)
├── main.py                      # Skrip pipeline utama
├── requirements.txt
└── README.md
```

---

## Setup Lingkungan

### Prasyarat
| Komponen | Versi |
|---|---|
| Python | 3.10.x |
| Java (JDK) | OpenJDK 11 atau 17 (direkomendasikan), atau 21 |
| Apache Spark | 3.5.1 (bundled via PySpark) |
| Sistem Operasi | Windows 10/11, Ubuntu 20.04+, macOS |

### Langkah Instalasi

**1. Clone repositori dan masuk ke direktori:**
```bash
git clone https://github.com/<username>/mahadata-dmrdf-replication.git
cd mahadata-dmrdf-replication
```

**2. Buat virtual environment Python 3.10:**
```bash
# Menggunakan uv (direkomendasikan — cepat & reproducible)
pip install uv
uv venv --python 3.10 .venv

# Atau menggunakan venv standar
python3.10 -m venv .venv
```

**3. Aktifkan virtual environment:**
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

**4. Install semua dependensi:**
```bash
pip install -r requirements.txt
```

**5. Download dataset Flipkart:**
```bash
python -c "import kagglehub; kagglehub.dataset_download('PromptCloudHQ/flipkart-products')"
# Atau letakkan file 'flipkart_com-ecommerce_sample.csv' secara manual ke dalam folder data/
```

> **⚠️ Catatan JAVA_HOME:** Pastikan JAVA_HOME mengarah ke JDK versi 11, 17, atau 21 yang valid.  
> Pada Windows: `set JAVA_HOME=C:\Program Files\Java\jdk-17`

---

## Cara Menjalankan Eksperimen

### Opsi A: Menjalankan Pipeline Lengkap (Skrip Python)
```bash
python main.py
```

### Opsi B: Jupyter Notebook (Interaktif)
```bash
# Install kernel notebook ke .venv
.venv/Scripts/python -m ipykernel install --user --name=mahadata-dmrdf

# Buka Jupyter
jupyter notebook DMRDF_Flipkart_Experiment.ipynb
```

---

## Ringkasan Hasil Eksperimen

### Tabel Metrik Performa Model

| Kondisi Data | Model | Akurasi (PA%) | Precision | Recall | F1-Score | AUC-ROC | Training (s) | Testing (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Tanpa DMRDF | Logistic Regression | 90.38% | 0.9047 | 0.9038 | 0.9027 | 0.9503 | 2.25 | 1.08 |
| Tanpa DMRDF | SVM (LinearSVC) | 91.99% | 0.9199 | 0.9199 | 0.9194 | 0.9512 | 1.67 | 0.48 |
| Tanpa DMRDF | Decision Tree | 95.76% | 0.9577 | 0.9576 | 0.9577 | 0.9499 | 3.40 | 0.32 |
| **Dengan DMRDF** | Logistic Regression | 90.26% | 0.9034 | 0.9026 | 0.9017 | 0.9510 | **0.87** | **0.22** |
| **Dengan DMRDF** | SVM (LinearSVC) | 91.95% | 0.9196 | 0.9195 | 0.9191 | 0.9549 | **1.14** | **0.23** |
| **Dengan DMRDF** | Decision Tree | 95.55% | 0.9555 | 0.9555 | 0.9555 | 0.9442 | **2.10** | **0.30** |

### Ringkasan DMRDF Filter
- **Data Awal:** 20.000 baris
- **Data Setelah DMRDF:** 17.008 baris  
- **Redundansi Dieliminasi:** 2.992 baris (**14.96%**)
- **Waktu Eksekusi Filter RDD:** 3.891 detik

### Temuan Utama
1. Filter DMRDF berhasil menghapus **14.96%** data redundan dari dataset Flipkart.
2. Akurasi prediksi tetap konsisten (selisih < 0.3%) setelah DMRDF diterapkan, membuktikan validitas metode seleksi fitur.
3. **Efisiensi waktu training meningkat signifikan**: Linear SVC mengalami penurunan waktu 31.7% (1.67s → 1.14s), Decision Tree 38.2% (3.40s → 2.10s).
4. Hasil ini konsisten dengan jurnal asli (Narayanan et al., 2020) yang melaporkan DMRDF + SVM mencapai PA = 95.4% pada dataset mobile phone reviews.

---

## Lingkungan Eksperimen

| Komponen | Spesifikasi |
|---|---|
| Python | 3.10.22 (CPython, via uv) |
| PySpark | 3.5.1 |
| Java | OpenJDK 21.0.9 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| OS | Windows 11 |
| Spark Master | local[*] |
| Driver Memory | 2 GB |

---

## Referensi

Narayanan, S., Samuel, P., & Chacko, M. (2020). Improving prediction with enhanced Distributed Memory-based Resilient Dataset Filter. *Journal of Big Data*, *7*, Article 13. https://doi.org/10.1186/s40537-020-00292-y
