import time
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LinearSVC, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from src.preprocessing import build_nlp_feature_pipeline

def train_and_evaluate_model(
    model_name: str,
    classifier,
    train_df: DataFrame,
    test_df: DataFrame
) -> dict:
    """
    Melatih model klasifikasi Spark MLlib dan mengukur metrik evaluasi lengkap serta waktu eksekusi.
    """
    # 1. Pelatihan model & pengukuran Training Time
    start_train = time.time()
    trained_model = classifier.fit(train_df)
    train_time = time.time() - start_train

    # 2. Pengujian prediksi & pengukuran Testing Time
    start_test = time.time()
    predictions = trained_model.transform(test_df)
    predictions.cache()
    _ = predictions.count()  # Action untuk memastikan eksekusi graf komputasi Spark
    test_time = time.time() - start_test

    # 3. Perhitungan metrik evaluasi
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    prec_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    rec_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    accuracy = acc_evaluator.evaluate(predictions)
    precision = prec_evaluator.evaluate(predictions)
    recall = rec_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    
    try:
        auc = auc_evaluator.evaluate(predictions)
    except Exception:
        auc = 0.0

    return {
        "model_name": model_name,
        "accuracy": round(accuracy * 100.0, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "train_time_sec": round(train_time, 3),
        "test_time_sec": round(test_time, 3),
        "total_time_sec": round(train_time + test_time, 3)
    }

def run_experiment_suite(
    df: DataFrame,
    experiment_label: str = "With_DMRDF",
    train_ratio: float = 0.75,
    seed: int = 42
) -> list[dict]:
    """
    Mengeksekusi rangkaian eksperimen pelatihan model (Logistic Regression, Linear SVC, Decision Tree)
    menggunakan pipeline fitur NLP yang konsisten.
    """
    print(f"\n>>> Menyiapkan Feature Pipeline untuk eksperimen: [{experiment_label}] ...")
    tokenizer, remover, hashing_tf, idf, indexer, assembler = build_nlp_feature_pipeline(df)

    feature_pipeline = Pipeline(stages=[
        tokenizer, remover, hashing_tf, idf, indexer, assembler
    ])

    print(">>> Menyesuaikan pipeline transformasi fitur ke DataFrame...")
    pipeline_model = feature_pipeline.fit(df)
    assembled_df = pipeline_model.transform(df)

    # Split train-test sesuai proporsi jurnal asli: 75% train, 25% test
    train_df, test_df = assembled_df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    train_df.cache()
    test_df.cache()

    print(f">>> Dataset dibagi menjadi: Train = {train_df.count()} baris, Test = {test_df.count()} baris.")

    # Inisialisasi pengklasifikasi
    classifiers = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=20,
            regParam=0.1
        ),
        "Support Vector Machine (LinearSVC)": LinearSVC(
            featuresCol="features",
            labelCol="label",
            maxIter=20,
            regParam=0.1
        ),
        "Decision Tree": DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=5,
            maxBins=300
        )
    }

    results = []
    for name, clf in classifiers.items():
        print(f"--- Melatih {name} [{experiment_label}] ---")
        metrics = train_and_evaluate_model(name, clf, train_df, test_df)
        metrics["dataset_condition"] = experiment_label
        results.append(metrics)
        print(f"    Hasil: Akurasi={metrics['accuracy']}%, Precision={metrics['precision']}, Recall={metrics['recall']}, Waktu={metrics['total_time_sec']}s")

    return results
