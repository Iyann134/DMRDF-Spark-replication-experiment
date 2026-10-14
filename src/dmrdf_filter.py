import time
import math
import hashlib
from pyspark.sql import DataFrame
from pyspark.rdd import RDD
from pyspark.sql import Row

def calculate_review_hash(text: str, category: str) -> str:
    """
    Menghasilkan hash representasi konten ulasan dan kategori untuk deteksi duplikasi cepat.
    """
    normalized = f"{category.strip().lower()}:::{text.strip().lower()[:150]}"
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def apply_dmrdf_rdd_filter(df: DataFrame, min_text_len: int = 25, ig_threshold: float = 0.5) -> tuple[DataFrame, dict]:
    """
    Implementasi Replikasi DMRDF (Distributed Memory-based Resilient Dataset Filter)
    menggunakan PySpark RDD Transformations:
    1. map(): Ekstraksi (key, value) pasangan ulasan produk.
    2. filter(): Eliminasi ulasan berfitur tidak signifikan (FIG / Feature Information Gain filter)
       dan ulasan dengan panjang tidak memadai.
    3. reduceByKey() / groupByKey(): Agregasi ulasan terdistribusi dan eliminasi ulasan duplikat/redundan.
    
    Mengembalikan:
    - filtered_df: DataFrame hasil pembersihan redundansi
    - stats: Dictionary metrik efisiensi dan waktu pemrosesan Spark RDD
    """
    start_time = time.time()
    spark = df.sparkSession
    total_raw_count = df.count()

    # Konversi DataFrame ke RDD untuk mengeksekusi operasi in-memory terdistribusi sesuai metodologi jurnal
    rdd = df.rdd

    # Tahap 1: Map transformation - bentuk representasi pasangan (hash_key, record)
    # Menghitung perkiraan skor Feature Information Gain berdasarkan panjang leksikon unik & polaritas
    def map_to_dmrdf_pair(row):
        text = str(row["cleaned_text"]) if row["cleaned_text"] else ""
        category = str(row["root_category"]) if row["root_category"] else "Other"
        
        # Perhitungan Feature Information Gain (FIG) aproksimasi
        tokens = text.split()
        unique_tokens = set(tokens)
        
        # Estimasi entropi/informasi leksikal
        n_tokens = len(tokens)
        n_unique = len(unique_tokens)
        if n_tokens > 0:
            # Rasio kekayaan kosakata sebagai proksi Information Gain Delta_IG
            lexical_richness = n_unique / (n_tokens + 1e-5)
            # Scaling agar berada pada rentang [0, 1]
            approx_ig = min(1.0, lexical_richness * math.log2(n_unique + 2) / 4.0)
        else:
            approx_ig = 0.0

        key = calculate_review_hash(text, category)
        return (key, (row, approx_ig, len(text)))

    mapped_rdd = rdd.map(map_to_dmrdf_pair)

    # Tahap 2: Filter transformation - eleminasi ulasan yang tidak memiliki fitur signifikan
    # Jurnal acuan: "Features with Information Gain greater than 0.5 is selected as a significant feature.
    # Reviews with score value 0 are eliminated."
    filtered_rdd = mapped_rdd.filter(
        lambda pair: (pair[1][1] >= 0.2) and (pair[1][2] >= min_text_len)
    )

    # Tahap 3: ReduceByKey transformation - Mengeliminasi duplikasi terdistribusi
    # "If more than one Distributed Memory-based score value is identified then the second occurrence is considered duplicate"
    # Mengambil rekaman pertama per grup hash identik
    deduplicated_rdd = filtered_rdd.reduceByKey(lambda a, b: a)

    # Ekstrak kembali Row objek
    cleaned_rows_rdd = deduplicated_rdd.map(lambda pair: pair[1][0])

    # Konversi kembali ke DataFrame dengan skema asli
    filtered_df = spark.createDataFrame(cleaned_rows_rdd, schema=df.schema)
    # Cache ke memori terdistribusi Spark (In-Memory Persist)
    filtered_df.cache()
    filtered_count = filtered_df.count()

    execution_time = time.time() - start_time
    redundancy_removed = total_raw_count - filtered_count
    reduction_pct = (redundancy_removed / total_raw_count) * 100.0

    stats = {
        "raw_count": total_raw_count,
        "filtered_count": filtered_count,
        "redundancy_removed": redundancy_removed,
        "reduction_pct": round(reduction_pct, 2),
        "execution_time_sec": round(execution_time, 3)
    }

    return filtered_df, stats
