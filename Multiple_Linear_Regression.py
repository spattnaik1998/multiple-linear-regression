from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import sys

spark = SparkSession.builder.appName("TaxiDataAnalysis").config("Spark Shuffle Partition", 1000).getOrCreate()

df = spark.read.csv(sys.argv[1], header=False)

corrected_df = df.select(
    col("_c0").cast("string").alias("medallion"),
    col("_c1").cast("string").alias("hack_license"),
    col("_c2").cast("timestamp").alias("pickup_datetime"),
    col("_c3").cast("timestamp").alias("dropoff_datetime"),
    col("_c4").cast("integer").alias("trip_time_in_secs"),
    col("_c5").cast("float").alias("trip_distance"),
    col("_c6").cast("float").alias("pickup_longitude"),
    col("_c7").cast("float").alias("pickup_latitude"),
    col("_c8").cast("float").alias("dropoff_longitude"),
    col("_c9").cast("float").alias("dropoff_latitude"),
    col("_c10").cast("string").alias("payment_type"),
    col("_c11").cast("float").alias("fare_amount"),
    col("_c12").cast("float").alias("surcharge"),
    col("_c13").cast("float").alias("mta_tax"),
    col("_c14").cast("float").alias("tip_amount"),
    col("_c15").cast("float").alias("tolls_amount"),
    col("_c16").cast("float").alias("total_amount")
)

# Filter data using DataFrame API
corrected_df = corrected_df.filter(
    (col("trip_distance") >= 1) & (col("trip_distance") <= 50) &
    (col("fare_amount") >= 3) & (col("fare_amount") <= 200) &
    (col("tolls_amount") >= 3) &
    ((col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long")) >= 120) &
    ((col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long")) <= 3600)
)

corrected_df = corrected_df.dropna()

features = np.array(corrected_df.select("trip_time_in_secs", "trip_distance", "fare_amount", "tolls_amount").collect())
features = np.insert(features, 0, 1, axis=1)  # Add bias term
target = np.array(corrected_df.select("total_amount").rdd.flatMap(lambda x: x).collect())

learning_rate = 0.001
num_iterations = 100
parameters = np.full(features.shape[1], 0.1)

for iteration in range(num_iterations):
  predictions = np.dot(features, parameters)
  errors = predictions - target
  gradient = np.dot(features.T, errors) / len(target)
  parameters -= learning_rate * gradient

  cost = np.sum(errors ** 2) / (2 * len(target))
  print(f"Iteration {iteration + 1}: Cost={cost}")

print("Final Model Parameters:", parameters)

# Stop SparkSession
spark.stop()
