# Week 2 - Experiment Tracking

**Table of contents**

1. [Introduction to MLOps](#part-1)
2. [Configure enviromnent](#part-2)
3. [Train the first ML model](#part-3)
4. [MLOps Maturity model](#part-4)

## Part 1: Introduction <a id='part-1'></a>

+ Important concepts to know:
  + **ML Experiment**: the process of building ML models.
  + **Experiment run**: each trial in an ML experiment.
  + **Run artifact**: any file that is associated with an ML run.
  + **Experiment metadata**.

+ What is Experiment Tracking?
  + Experiment tracking is the process of keeping track of all the **relevant information** from an **ML experiment**, includes:
    + Source code;
    + Environment;
    + Data;
    + Model;
    + Hyperparameters;
    + Metrics.
  
+ Why is Experimentr Tracking important?
  + Reproducibility.
  + Organization.
  + Optimization.

+ Why should not tracking in Spreadsheet, Excel??
  + Error prone.
  + No standard format.
  + Visibility and Collaboration.

+ **MLflow**:
  + A Python open-source library for the Machine Learning lifecycle. Including:
    + Tracking.
    + Models.
    + Model Registry.
    + Projects.
  
  + MLflow allows to organize experiment into runs, and keep track of: 
    + Parameters.
    + Metrics.
    + Metadata.
    + Artifacts.
    + Models.
  
  + MLflow also automatically logs extra information for the run:
    + Source code.
    + Version of the code (git commit).
    + Start and end time.
    + Author.
  
## Part 2: Getting started with MLflow

### 2.1. First experiment

+ Installation:

  ```bash
  pip install mlflow
  ```

+ Launch the MLflow UI:

  ```bash
  mlfow ui
  ```
  + Go to URL: http://127.0.0.1:5000/

+ Start the MLflow server with SQLite as backend:

  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

+ Import MLflow in Python:

  ```python
  import mlflow
  
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("nyc-taxi-experiment")
  ```

+ To wrap each model training in a MLflow run and log experiment information:

  ```python
  with mlflow.start_run():
    mlflow.set_tag("developer", "nda")
    mlflow.log_param("train_data_path", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
    mlflow.log_param("val_data_path", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet")

    mlflow.log_param("mode", "linear_reg")
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    print("mse: ", mse)
    print("rmse: ", rmse)
  ```

+ Go to MLflow UI to view the experiment with logged information.


### 2.2. Experiment tracking

