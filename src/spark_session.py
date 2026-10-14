import os
import sys
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "DMRDF_Replication", memory: str = "2g", cores: str = "local[*]") -> SparkSession:
    """
    Inisialisasi SparkSession lokal yang terkontrol dengan konfigurasi memori ramah lingkungan lokal.
    Memastikan kompatibilitas worker Python dengan runtime aktif.
    """
    # Pastikan worker python merujuk ke interpreter aktif di dalam venv
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder \
        .appName(app_name) \
        .master(cores) \
        .config("spark.driver.memory", memory) \
        .config("spark.executor.memory", memory) \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark
