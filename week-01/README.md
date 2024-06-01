# Week 1 - Introduction to MLOps

### Table of contents

1. [Introduction to MLOps](#part-1)
2. [Configure enviromnent](#part-2)
3. [Train the first ML model](#part-3)
4. [MLOps Maturity model](#part-4)


## Part 1: Introduction <a id='part-1'></a>

+ **MLOps** is a set of practices for automating machine learning solutions with everything to operate together to run the models on production.

+ Best practices for MLOps:
  + Design.
  + Train.
  + Operate.


## Part 2: Configure Environment <a id='part-2'></a>

+ Use **GitHub Codespaces** to configure the environment for the course.
+ In dedicated repository for the course, select Code -> Create codespace on main.
+ The codespace on browser will be created and opened.
+ Click File -> Open in VS Code Desktop to use local VS Code to connect the machine (make sure we have the Github Codespaces extention).

+ Download Anaconda for the codespace machine:

  ```bash
  wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
  bash Anaconda3-2024.02-1-Linux-x86_64.sh 
  ```

+ Open a new terminal

  ```bash
  bash
  ```

+ Check if Python is using by Conda:

  ```bash
  which python
  ```

+ Open a new Jupyter Notebook for running scripts in Python:

  ```bash
  jupyter notebook
  ```

  + Make sure the port is forwarding from the codespace to the local machine.

+ Try to read NYC Taxi Trip data (Yellow 2024):

  ```python
  import pandas as pd
  df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
  df.head()
  ```

+ The codespace machine will be deleleted if there is no activity or there is no code un-committed.

+ The freemium of codespace is 120 hours per month.

## Part 3: Training the first ML model <a id='part-3'></a>

+ Using the Yellow Taxi Trip data (January 2024), we will transform and clean the data before applying the model.

+ **Clean the data:**

  + Parsing the datetime column:

    ```python
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    ```

  + Calculate the trip duration:

    ```python
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda t: t.total_seconds() / 60)
    ```

  + Filter the correct data:

    ```python
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    ```

+ **Preprocessing**:
  + Columns to use for the model:

    ```python
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    ```

  + Preprocesing columns to list of dictionaries:

    ```python
    from sklearn.feature_extraction import DictVectorizer

    dv = DictVectorizer()
    train_dicts = df[categorical + numerical].to_dict(orient="records")

    X_train = dv.fit_transform(train_dicts)
    ```

  + Label data:

    ```python
    target = 'duration'
    y_train = df[target].values
    ```
  
  + Wrap up the code into resuable functions:

    ```python
    def read_data(url: str) -> pd.DataFrame:
      df = pd.read_parquet(url)
      
      df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
      df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
      
      df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
      df.duration = df.duration.apply(lambda t: t.total_seconds() / 60)
      
      df = df[(df.duration >= 1) & (df.duration <= 60)]

      categorical = ['PULocationID', 'DOLocationID']
      numerical = ['trip_distance']

      df[categorical] = df[categorical].astype(str)
      
      return df
    ```
  
  + Query train and test data:

    ```python
    train_data = read_data("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
    test_data = read_data("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet")
    ```

+ **Train model**:
  + Use Linear Regression model of scikit-learn:

    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    ```
  
  + Visualize the prediction vs actual values:

    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.distplot(y_pred, label='prediction')
    sns.distplot(y_train, label='actual')

    plt.legend()
    ```
  
  + Evaluate the model: use MSE and RMSE metrics:

    ```python
    from sklearn.metrics import mean_squared_error

    print("mse: ", mean_squared_error(y_train, y_pred))
    print("rmse: ", mean_squared_error(y_train, y_pred, squared=False))
    ```

+ **Save the model:**
  + Need to save the (trained) DictVectorizer and Linear Regression model:

    ```python
    import pickle

    with open('./models/lin_reg.bin', 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    ```

## Part 4: MLOps Maturity model <a id='part-4'></a>

Reference: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model

+ **Level 0**: No MLOps.

  + All code in Jupyter Notebook for model experiment.
  + No automation.
  + Full human interation.
  + ML: Proof of Concept (PoC).

+ **Level 1**: DevOps, but no MLOps.

  + Model releases are automated.
  + Best practices from Devops, Software engineering, but no machine learning:
    + Unit & Integration testing.
    + CI/CD.
    + OPs metrics.
  + No Experiment tracking.
  + No Reproducibility.
  + Data Scientist separated from Engineering.

+ **Level 2**: Automated training.

  + Parameterize training code.
  + Training pipeline.
  + Experiment tracking.
  + Trained models stored in Model Registry.
  + Low fiction deployment
  + Data Scientists wok with Engineeeing.

+ **Level 3**: Automated deployment.
  
  + Easy to deploy model, limit human effort.
  + Deloyment pipeline.
  + Model monitoring.
  + A/B testing.

+ **Level 4**: Full MLOps automation.

  + Automated model training.
  + Automated model deployment.
  + No human interaction.
