import os
import sys

# ============================================================
# GUARD: Pastikan berjalan di Python 3.10 (.venv), bukan Python 3.12+
# PySpark 3.5.x tidak kompatibel dengan Python 3.12+ (cloudpickle RecursionError)
# ============================================================
if sys.version_info >= (3, 12):
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if os.path.exists(_venv_python):
        print(f"[GUARD] Terdeteksi Python {sys.version_info.major}.{sys.version_info.minor} "
              f"({sys.executable}) — tidak kompatibel dengan PySpark RDD.")
        print(f"[GUARD] Me-relaunch dengan: {_venv_python}")
        os.execv(_venv_python, [_venv_python] + sys.argv)
    else:
        print(f"[ERROR] Python {sys.version_info.major}.{sys.version_info.minor} terdeteksi, "
              f"namun .venv\\Scripts\\python.exe tidak ditemukan.")
        print(f"[ERROR] Buat dulu dengan: uv venv --python 3.10 .venv && .venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)

# Set variabel environment agar Spark worker menggunakan Python yang sama
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.spark_session import get_spark_session
from src.preprocessing import load_raw_dataset, clean_and_prepare_data
from src.dmrdf_filter import apply_dmrdf_rdd_filter
from src.modeling import run_experiment_suite

def generate_comparison_plots(df_metrics: pd.DataFrame, output_path: str = "results/benchmark_plot.png"):
    """
    Menghasilkan visualisasi komparatif metrik performa (Akurasi, F1, Waktu Eksekusi)
    antara kondisi Tanpa DMRDF vs Menggunakan DMRDF.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Akurasi & F1-Score
    df_melted = df_metrics.melt(
        id_vars=["model_name", "dataset_condition"],
        value_vars=["accuracy", "f1_score"],
        var_name="Metric",
        value_name="Score"
    )
    # Sesuaikan skala F1-score ke persen untuk visualisasi seimbang
    df_melted.loc[df_melted["Metric"] == "f1_score", "Score"] *= 100.0

    palette = {"Without_DMRDF": "#e74c3c", "With_DMRDF": "#2ecc71"}
    
    sns.barplot(
        data=df_metrics,
        x="model_name",
        y="accuracy",
        hue="dataset_condition",
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title("Perbandingan Akurasi Prediksi (PA %)\n(Baseline vs DMRDF Filter)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Model Klasifikasi", fontsize=11)
    axes[0].set_ylabel("Akurasi (%)", fontsize=11)
    axes[0].set_ylim(60, 102)
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f"{height:.1f}%",
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                             textcoords='offset points')

    # Plot 2: Total Waktu Eksekusi (Training + Testing)
    sns.barplot(
        data=df_metrics,
        x="model_name",
        y="total_time_sec",
        hue="dataset_condition",
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title("Perbandingan Efisiensi Waktu Eksekusi (detik)\n(In-Memory Spark MLlib Processing)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Model Klasifikasi", fontsize=11)
    axes[1].set_ylabel("Waktu Total (detik)", fontsize=11)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{height:.2f}s",
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                             textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[OK] Grafik benchmark komparatif berhasil disimpan ke: {output_path}")

def main():
    print("=" * 80)
    print("EKSPERIMEN REPLIKASI BIG DATA: METODE DMRDF PADA APACHE SPARK")
    print("Mata Kuliah: Mahadata (IF25-40402) - Institut Teknologi Sumatera (ITERA)")
    print("Jurnal Acuan: Narayanan et al. (2020), Journal of Big Data (Springer)")
    print("=" * 80)

    # 1. Inisialisasi SparkSession
    print("\n[1/6] Menginisialisasi Apache SparkSession...")
    spark = get_spark_session(app_name="DMRDF_Replication_ITERA", memory="2g")
    print(f"      Spark Version : {spark.version}")
    print(f"      Spark Master  : {spark.sparkContext.master}")

    # 2. Memuat Dataset Flipkart
    dataset_path = "data/flipkart_com-ecommerce_sample.csv"
    print(f"\n[2/6] Memuat Dataset Flipkart dari: {dataset_path}...")
    df_raw = load_raw_dataset(spark, dataset_path)
    print(f"      Total data mentah: {df_raw.count()} baris, {len(df_raw.columns)} kolom.")

    # 3. Praproses & Pembersihan Data Awal
    print("\n[3/6] Menjalankan Praproses & Pembersihan Awal (Spark SQL)...")
    df_cleaned = clean_and_prepare_data(df_raw)
    df_cleaned.cache()
    print(f"      Data terproses siap eksperimen: {df_cleaned.count()} baris.")

    # 4. Eksperimen Baseline: Model Machine Learning Tanpa Filter Redundansi (Without DMRDF)
    print("\n[4/6] Menjalankan Rangkaian Pengujian Model Baseline (Tanpa Filter DMRDF)...")
    results_baseline = run_experiment_suite(df_cleaned, experiment_label="Without_DMRDF")

    # 5. Eksperimen DMRDF: Redundancy Removal Terdistribusi dengan RDD
    print("\n[5/6] Menerapkan Replikasi Filter Redundansi DMRDF (Spark RDD Transformations)...")
    df_filtered, dmrdf_stats = apply_dmrdf_rdd_filter(df_cleaned, min_text_len=30, ig_threshold=0.3)
    
    print("\n" + "-" * 50)
    print("RINGKASAN FILTER REDUNDANSI DMRDF:")
    print(f"  * Jumlah Data Awal        : {dmrdf_stats['raw_count']} baris")
    print(f"  * Jumlah Setelah DMRDF    : {dmrdf_stats['filtered_count']} baris")
    print(f"  * Data Redundan Dieliminasi: {dmrdf_stats['redundancy_removed']} baris ({dmrdf_stats['reduction_pct']}%)")
    print(f"  * Waktu Eksekusi Filter RDD: {dmrdf_stats['execution_time_sec']} detik")
    print("-" * 50)

    # Eksperimen Model pada Data yang Difilter DMRDF
    print("\n      Melatih Model pada Dataset Bersih DMRDF (With DMRDF)...")
    results_dmrdf = run_experiment_suite(df_filtered, experiment_label="With_DMRDF")

    # 6. Agregasi Hasil, Penyimpanan, dan Visualisasi
    print("\n[6/6] Menyusun Tabel Metrik Komparatif & Menghasilkan Visualisasi...")
    all_results = results_baseline + results_dmrdf
    df_metrics = pd.DataFrame(all_results)
    
    os.makedirs("results", exist_ok=True)
    csv_output_path = "results/metrics_comparison.csv"
    df_metrics.to_csv(csv_output_path, index=False)
    print(f"      Hasil metrik disimpan ke: {csv_output_path}")

    # Cetak tabel ringkasan format teks
    print("\n" + "=" * 95)
    print(f"{'Kondisi Data':<15} | {'Model':<35} | {'Akurasi (%)':<11} | {'Precision':<9} | {'Recall':<8} | {'Waktu (s)':<9}")
    print("-" * 95)
    for _, row in df_metrics.iterrows():
        print(f"{row['dataset_condition']:<15} | {row['model_name']:<35} | {row['accuracy']:<11.2f} | {row['precision']:<9.4f} | {row['recall']:<8.4f} | {row['total_time_sec']:<9.3f}")
    print("=" * 95)

    # Buat grafik benchmark
    generate_comparison_plots(df_metrics, output_path="results/benchmark_plot.png")

    spark.stop()
    print("\n[SELESAI] Seluruh tahapan eksperimen replikasi berhasil dieksekusi dengan sukses!\n")

if __name__ == "__main__":
    main()
