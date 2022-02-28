# Databricks notebook source
df = spark.sql("SELECT * FROM digits")
display(df)

# COMMAND ----------

# MAGIC %md # Parameters

# COMMAND ----------

dbutils.widgets.text("experiment_name", "", label="1 - Experiment Name")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------

# MAGIC %md # Libraries

# COMMAND ----------

import mlflow.sklearn
import mlflow

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# COMMAND ----------

from matplotlib import pyplot as plt
%matplotlib inline 

# COMMAND ----------

# MAGIC %md # Get Data

# COMMAND ----------

X, y = load_digits(return_X_y=True)

# COMMAND ----------

def display_numbers(X,y):
  n = 10
  plt.figure(figsize=(20, 20))
  for i in range(n):
    plt.subplot(1,n,i+1)
    plt.imshow(X[i,:].reshape((8,8)), cmap='gray_r')
    plt.title(y[i])
    plt.axis('off')
  plt.tight_layout()
  plt.show()
  
display_numbers(X, y)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

# COMMAND ----------

display_numbers(X_train, y_train)

# COMMAND ----------

# MAGIC %md # Feature Engineering

# COMMAND ----------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

display_numbers(X_train_scaled, y_train)

# COMMAND ----------

# MAGIC %md # Model Training

# COMMAND ----------

# MAGIC %md Set Experiment

# COMMAND ----------

# MAGIC %md
# MAGIC <div>
# MAGIC     <center>
# MAGIC     <img src="https://raw.githubusercontent.com/danielp-db/mlflow-notebook/main/images/set_create_experiment.png" width="500"/>
# MAGIC     </center>
# MAGIC </div>

# COMMAND ----------

try:
  mlflow.set_experiment(experiment_name)
except:
  mlflow.create_experiment(experiment_name)
  mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md Train the model

# COMMAND ----------

# MAGIC %md
# MAGIC <div>
# MAGIC     <center>
# MAGIC     <img src="https://raw.githubusercontent.com/danielp-db/mlflow-notebook/main/images/register_run_databricks.png" width="500"/>
# MAGIC     </center>
# MAGIC </div>

# COMMAND ----------

with mlflow.start_run():
  model = LogisticRegression()
  
  model.fit(X_train_scaled, y_train)

  #LOG MODEL
  mlflow.sklearn.log_model(model,
                           artifact_path = "model")

  mlflow.log_param("alpha", 0.5)
  
  #LOG METRICS
  y_train_pred = model.predict(X_train_scaled)
  accuracy_train = balanced_accuracy_score(y_train, y_train_pred)
  mlflow.log_metric("accuracy_train", accuracy_train)
  
  y_test_pred = model.predict(X_test_scaled)
  accuracy_test = balanced_accuracy_score(y_test, y_test_pred)
  mlflow.log_metric("accuracy_test", accuracy_test)
  
  #LOG TRAINING ARTIFACTS
  conf_mat_train = plot_confusion_matrix(model, X_train_scaled, y_train, colorbar=False)
  plt.savefig("confusion_matrix_train.jpg")
  mlflow.log_artifact("confusion_matrix_train.jpg", artifact_path="confusion_matrix")
  
  #LOG TEST ARTIFACTS
  conf_mat_test = plot_confusion_matrix(model, X_test_scaled, y_test, colorbar=False)
  plt.savefig("confusion_matrix_test.jpg")
  mlflow.log_artifact("confusion_matrix_test.jpg", artifact_path="confusion_matrix")
  
  #LOG TRANSFORMERS
  joblib.dump(scaler, "scaler.pkl")
  mlflow.log_artifact("scaler.pkl", artifact_path="transformers")

# COMMAND ----------

# MAGIC %md # Go to notebook 2 before running next cell...

# COMMAND ----------

# MAGIC %md # Model training and registration

# COMMAND ----------

# MAGIC %md
# MAGIC <div>
# MAGIC     <center>
# MAGIC     <img src="https://raw.githubusercontent.com/danielp-db/mlflow-notebook/main/images/register_model_databricks.png" width="500"/>
# MAGIC     </center>
# MAGIC </div>

# COMMAND ----------

model_name = "digits_model"

# COMMAND ----------

with mlflow.start_run():
  model.fit(X_train_scaled, y_train)

  #LOG MODEL
  mlflow.sklearn.log_model(model,
                           artifact_path = "model",
                           registered_model_name=model_name) #This is the only line that changes

  #LOG METRICS
  y_train_pred = model.predict(X_train_scaled)
  accuracy_train = balanced_accuracy_score(y_train, y_train_pred)
  mlflow.log_metric("accuracy_train", accuracy_train)
  
  y_test_pred = model.predict(X_test_scaled)
  accuracy_test = balanced_accuracy_score(y_test, y_test_pred)
  mlflow.log_metric("accuracy_test", accuracy_test)
  
  #LOG TRAINING ARTIFACTS
  conf_mat_train = plot_confusion_matrix(model, X_train_scaled, y_train, colorbar=False)
  plt.savefig("confusion_matrix_train.jpg")
  mlflow.log_artifact("confusion_matrix_train.jpg", artifact_path="confusion_matrix")
  
  #LOG TEST ARTIFACTS
  conf_mat_test = plot_confusion_matrix(model, X_test_scaled, y_test, colorbar=False)
  plt.savefig("confusion_matrix_test.jpg")
  mlflow.log_artifact("confusion_matrix_test.jpg", artifact_path="confusion_matrix")
  
  #LOG TRANSFORMERS
  joblib.dump(scaler, "scaler.pkl")
  mlflow.log_artifact("scaler.pkl", artifact_path="transformers")
