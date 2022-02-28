# Databricks notebook source
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y = True, as_frame=True)
df = spark.createDataFrame(X)

(
  df.write
  .format("delta") 
  .saveAsTable("digits")
)

# COMMAND ----------

spark.sql("SELECT * FROM digits").inputFiles()
