# Week 1 - Introduction to MLOps

### Table of contents

1. [Introduction to GCP](#part-1)
2. [Introduction to Docker](#part-2)
3. [SQL Refresher](#part-3)
4. [Set up Google Cloud environment](#part-4)
5. [Infrastruture as Code with Terraform](#part-5)

    [Additional resources](#resource)

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

## Part 3: Training the first ML model

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


  + 
