import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, regexp_replace, regexp_extract, lower,
    coalesce, lit, length, udf, trim
)
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, HashingTF, IDF,
    StringIndexer, VectorAssembler
)

def load_raw_dataset(spark, file_path: str) -> DataFrame:
    """
    Memuat dataset CSV mentah Flipkart dengan konfigurasi multiline dan escape quote yang tepat.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan pada path: {file_path}")

    df = spark.read \
        .option("header", "true") \
        .option("multiLine", "true") \
        .option("escape", "\"") \
        .csv(file_path)
    
    return df

def clean_and_prepare_data(df: DataFrame) -> DataFrame:
    """
    Pembersihan data awal:
    - Ekstraksi kategori utama dari kolom 'product_category_tree'
    - Parsing harga numerik 'retail_price' dan 'discounted_price'
    - Penanganan rating produk dan pembentukan label biner 'label' (Success: 1, Failure: 0)
    - Pembersihan teks 'description' dan 'product_name'
    """
    # 1. Ekstraksi kategori utama (root category) dari JSON string tree
    # Contoh format: ["Clothing >> Women's Clothing >> ..."] -> Clothing
    category_pattern = r'\[\"([^\">]+)'
    df_clean = df.withColumn(
        "root_category",
        regexp_extract(col("product_category_tree"), category_pattern, 1)
    )
    df_clean = df_clean.withColumn(
        "root_category",
        when(col("root_category") == "", lit("Other")).otherwise(trim(col("root_category")))
    )

    # 2. Parsing kolom harga numerik
    df_clean = df_clean.withColumn("retail_price_num", col("retail_price").cast(DoubleType()))
    df_clean = df_clean.withColumn("discounted_price_num", col("discounted_price").cast(DoubleType()))

    # Imputasi harga yang null dengan nilai default/median
    avg_price = 1000.0
    df_clean = df_clean.fillna({
        "retail_price_num": avg_price,
        "discounted_price_num": avg_price
    })

    # Hitung persentase diskon
    df_clean = df_clean.withColumn(
        "discount_ratio",
        when(col("retail_price_num") > 0,
             (col("retail_price_num") - col("discounted_price_num")) / col("retail_price_num")
        ).otherwise(lit(0.0))
    )

    # 3. Parsing kolom rating & pembentukan variabel target (Success = 1, Failure = 0)
    # Sesuai paper Narayanan et al. (2020): Rating scale [1-5], threshold rating >= 4.0 atau kesuksesan pre-launch
    rating_num = col("overall_rating").cast(DoubleType())
    df_clean = df_clean.withColumn("rating_numeric", rating_num)

    # Label target: jika rating >= 4.0 -> 1.0, jika rating < 4.0 -> 0.0
    # Jika rating null/no rating: manfaatkan kombinasi kepuasan produk (diskon kompetitif & ulasan deskripsi lengkap)
    has_high_rating = col("rating_numeric") >= 4.0
    has_low_rating = (col("rating_numeric") < 4.0) & (col("rating_numeric").isNotNull())
    
    # Untuk unrated products: proksi rating berbasis daya tarik diskon (> 35%) & spesifikasi kaya
    success_proxy = (col("discount_ratio") >= 0.35) & (length(col("description")) > 100)

    df_clean = df_clean.withColumn(
        "label",
        when(has_high_rating, lit(1.0))
        .when(has_low_rating, lit(0.0))
        .when(success_proxy, lit(1.0))
        .otherwise(lit(0.0))
    )

    # 4. Pembersihan teks ulasan / deskripsi produk
    df_clean = df_clean.withColumn(
        "cleaned_text",
        lower(coalesce(col("description"), col("product_name"), lit("")))
    )
    # Hapus karakter non-alphanumeric dasar
    df_clean = df_clean.withColumn(
        "cleaned_text",
        regexp_replace(col("cleaned_text"), r"[^a-zA-Z0-9\s]", " ")
    )

    return df_clean

def build_nlp_feature_pipeline(df: DataFrame, num_features: int = 500):
    """
    Ekstraksi fitur teks menggunakan Tokenizer, StopWordsRemover, TF-IDF,
    serta VectorAssembler untuk menggabungkan fitur teks dan numerik.
    """
    tokenizer = RegexTokenizer(
        inputCol="cleaned_text",
        outputCol="tokens",
        pattern=r"\s+",
        minTokenLength=3
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )

    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_tf",
        numFeatures=num_features
    )

    idf = IDF(
        inputCol="raw_tf",
        outputCol="tfidf_features"
    )

    indexer = StringIndexer(
        inputCol="root_category",
        outputCol="category_idx",
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=["tfidf_features", "retail_price_num", "discount_ratio", "category_idx"],
        outputCol="features",
        handleInvalid="skip"
    )

    return tokenizer, remover, hashing_tf, idf, indexer, assembler
