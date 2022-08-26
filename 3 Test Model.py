# Databricks notebook source
# MAGIC %md # Parameters

# COMMAND ----------

#CODIGO NUEVO

# COMMAND ----------

#Experiment
dbutils.widgets.text("experiment_name", "", label="1 - Experiment Name")
experiment_name = dbutils.widgets.get("experiment_name")
#Model
dbutils.widgets.text("model_name", "", label="2 - Model Name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md # Libraries

# COMMAND ----------

import mlflow.sklearn
import mlflow
import joblib

# COMMAND ----------

# MAGIC %md # Model Production From Experiment Run

# COMMAND ----------

# MAGIC %md List Experiment Runs

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# COMMAND ----------

runs = mlflow.search_runs(experiment_id, order_by=["metrics.accuracy_test DESC"])
runs

# COMMAND ----------

# MAGIC %md Get the Run ID of the Run with the highest metric

# COMMAND ----------

run_id = runs.iloc[0,:].run_id

# COMMAND ----------

from mlflow.tracking import MlflowClient
import os

# COMMAND ----------

# MAGIC %md Load model of the selected Run ID

# COMMAND ----------

client = MlflowClient()
local_dir = "/tmp/artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

#DOWNLOAD MODEL
model_path = client.download_artifacts(run_id, "model", local_dir)
#DOWNLOAD TRANSFORMERS
transformer_path = client.download_artifacts(run_id, "transformers", local_dir)

# COMMAND ----------

'Model', os.listdir(model_path), 'Transformer', os.listdir(transformer_path)

# COMMAND ----------

model = joblib.load(os.path.join(model_path, 'model.pkl'))
scaler = joblib.load(os.path.join(transformer_path, 'scaler.pkl'))

# COMMAND ----------

# MAGIC %md Score model

# COMMAND ----------

X, _ = load_digits(return_X_y=True)

# COMMAND ----------

X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

# COMMAND ----------

# MAGIC %md # Model Production From Registered Model

# COMMAND ----------

# MAGIC %md Load registered model

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

model = mlflow.pyfunc.load_model("models:/aa_digits/Production")

# COMMAND ----------

# MAGIC %md Score model

# COMMAND ----------

from sklearn.datasets import load_digits
X, y = load_digits(return_X_y = True)

# COMMAND ----------

model.predict(X)
