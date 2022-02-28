# Databricks notebook source
# MAGIC %md # Parameters

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
from mlflow.tracking import MlflowClient
import os

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# COMMAND ----------

# MAGIC %md # Get Data

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# COMMAND ----------

runs = mlflow.search_runs(experiment_id, order_by=["metrics.accuracy_test DESC"])
runs

# COMMAND ----------

run_id = runs.iloc[0,:].run_id

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri = model_uri,
                      name=model_name)
