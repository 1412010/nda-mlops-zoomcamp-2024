# Week 3 - Workflow Orchestration

**Table of contents**

1. [Introduction to MLOps](#part-1)
2. [Configure enviromnent](#part-2)
3. [Train the first ML model](#part-3)
4. [MLOps Maturity model](#part-4)

## Part 1: Machine Learning pipeline <a id='part-1'></a>


+ **ML pipeline**: A sequence of steps to process data and train the ML models.

+ To schedule the pipeline for running and define dependencies of each step, we need **orchestration** tool for the ML/Data pipeline: 
  + Airflow
  + Prefect
  + **Mage**

### How Mage helps MLOps

#### 1. Data preparation

+ Mage offers features to build, run, and manage data pipelines for data transformation and integration, including pipeline orchestration, notebook environments, data integrations, and streaming pipelines for real-time data.

#### 2. Training and deployment

+ Mage helps prepare data, train machine learning models, and deploy them with accessible API endpoints.

#### 3. Standardize complex processes

+ Mage simplifies MLOps by providing a unified platform for data pipelining, model development, deployment, versioning, CI/CD, and maintenance, allowing developers to focus on model creation while improving efficiency and collaboration.

## Part 2: Mage setup

+ Quick Start to run Mage on Windows machine.

  1. Clone the following respository containing the complete code for this module:

      ```bash
      git clone https://github.com/mage-ai/mlops.git
      cd mlops
      ```

  1. Launch Mage and the database service (PostgreSQL):

      ```bash
      ./scripts/start.sh
      ```

      If don't have bash in your enviroment, modify the following command and run it:

      ```bash
      PROJECT_NAME=mlops \
          MAGE_CODE_PATH=/home/src \
          SMTP_EMAIL=$SMTP_EMAIL \
          SMTP_PASSWORD=$SMTP_PASSWORD \
          docker compose up
      ```

      It is ok if you get this warning, you can ignore it  
      `The "PYTHONPATH" variable is not set. Defaulting to a blank string.`

  1. The subproject that contains all the pipelines and code is named
    [`unit_3_observability`](https://github.com/mage-ai/mlops/tree/master/mlops/unit_3_observability)

+ Run example pipeline:

  1. Open [`http://localhost:6789`](http://localhost:6789) in your browser.

  1. In the top left corner of the screen next to the Mage logo and **`mlops`** project name,
    click the project selector dropdown and choose the **`unit_0_setup`** option.

  1. Click on the pipeline named **`example_pipeline`**.
  1. Click on the button labeled **`Run @once`**.

## Part 3. Working with Mage

### 3.1. Data preparation

+ Create a new project on Mage:
  + In Mage editor, choose File -> `Create a new Mage project`  -> Give a name: `unit_1_data_prep`

  + Register the project: Go to Settings -> Enter project name -> Register.

  + Create a new pipeline `+New pipeline` -> Standard pipeline -> enter name: **Data preparation** and some descriptions.

+ Ingest data:
  + Create a new block: All blocks -> Data Loader -> Base template (generic) -> give a name
  + Put the code for ingesting Yellow Taxi Trip data: 
    ```python
    import requests
    from io import BytesIO
    import pandas as pd
    from typing import List

    @data_loader
    def load_data(*args, **kwargs) -> pd.DataFrame:
        """
        Template code for loading data from any source.

        Returns:
            Anything (e.g. data frame, dictionary, array, int, str, etc.)
        """
        dfs: List[pd.DataFrame] = []

        for year, months in [(2024, (1, 3))]:
            for month in range(*months):
                response = requests.get(
                    'https://d37ci6vzurychx.cloudfront.net/trip-data/'
                    f'yellow_tripdata_{year}-{month:02d}.parquet'
                )

                if response.status_code != 200:
                    raise Exception(response.text)
                
                df = pd.read_parquet(BytesIO(response.content))
                dfs.append(df)

        return pd.concat(dfs)
    ```

  + Execute the block.

  + We can create some charts for theh data on the Charts view on the right side.

+ Utilities for transforming data:
  + Create new files: 
    + `utils/data_preparation/cleaning.py`
    + `utils/data_preparation/feature_engineering.py`
    + `utils/data_preparation/feature_selector.py`
    + `utils/data_preparation/splitter.py`
    + `utils/data_preparation/encoder.py`

  + Make sure to add `__init__.py` file in the utils folder.

+ Transform the data: 
  + Create a new **Transformer** block with name `prepare`.
  + Code for transformation: 
    ```python
    from typing import Tuple
    import pandas as pd

    from mlops.utils.data_preparation.cleaning import clean
    from mlops.utils.data_preparation.feature_engineering import combine_features
    from mlops.utils.data_preparation.feature_selector import select_features
    from mlops.utils.data_preparation.splitters import split_on_value


    @transformer
    def transform(
        df: pd.DataFrame, *args, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_on_feature = kwargs.get('split_on_feature', 'lpep_pickup_datetime')
        split_on_feature_value = kwargs.get('split_on_feature_value', '2024-02-01')
        target = kwargs.get('target', 'duration')

        df = clean(df)
        df = combine_features(df)
        df = select_features(df, features=[split_on_feature, target])

        df_train, df_val = split_on_value(
            df,
            split_on_feature,
            split_on_feature_value,
        )

        return df, df_train, df_val
    ```

  + Create new Global variables for easy configuration:
    + `split_on_feature`: lpep_pickup_datetime
    + `split_on_feature_value`: 2024-02-01
    + `target`: duration

  + Execute the block.

+ Click to create **Histogram** chart to view and check the skewness of the data. Code for the chart:
  ```python
  import pandas as pd

  from mage_ai.shared.parsers import convert_matrix_to_dataframe

  if isinstance(df_1, list) and len(df_1) >= 1:
      item = df_1[0]
      if isinstance(item, pd.Series):
          item = item.to_frame()
      elif not isinstance(item, pd.DataFrame):
          item = convert_matrix_to_dataframe(item)
      df_1 = item

  columns = df_1.columns
  col = 'trip_distance'
  x = df_1[df_1[col] <= 20][col]

  ```

+ Export data: 
  + Create a new block: **Data Exporter** with name `build`
  + Code for the block: 
    ```python
    from typing import Tuple
    import pandas as pd
    from scipy.sparse._csr import csr_matrix
    from sklearn.base import BaseEstimator

    from mlops.utils.data_preparation.encoders import vectorize_features
    from mlops.utils.data_preparation.feature_selector import select_features

    if 'data_exporter' not in globals():
        from mage_ai.data_preparation.decorators import data_exporter


    @data_exporter
    def export_data(
        data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], *args, **kwargs
    ) -> Tuple[
        csr_matrix,
        csr_matrix,
        csr_matrix,
        pd.Series,
        pd.Series,
        pd.Series,
        BaseEstimator,
    ]:
        df, df_train, df_val = data
        target = kwargs.get('target', 'duration')

        X, _, _ = vectorize_features(select_features(df))
        y: pd.Series = df[target]

        X_train, X_val, dv = vectorize_features(
            select_features(df_train),
            select_features(df_val),
        )
        y_train = df_train[target]
        y_val = df_val[target]

        return X, X_train, X_val, y, y_train, y_val, dv

    ```
  
  + Execute the block.

+ Add test for the final data:
  + Full data: 
    ```python
    @test
    def test_dataset(
        X: csr_matrix,
        X_train: csr_matrix,
        X_val: csr_matrix,
        y: pd.Series,
        y_train: pd.Series,
        y_val: pd.Series,
        *args,
    ) -> None:
        assert(
            X.shape[0] == 105870
        ), f'Entire dataset should have 105870 examples, but has {X.shape[0]}'
        assert(
            X.shape[1] == 7027
        ), f'Entire dataset should have 7027 features, but has {X.shape[0]}'
        assert(
            len(y.index) == X.shape[0]
        ), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'
    ```
  
  + Do similar for train and validation data.

### 3.2. Training data

+ 







## Resources

1. [Code for example data pipeline](https://github.com/mage-ai/mlops/tree/master/mlops/unit_0_setup)
1. [The definitive end-to-end machine learning (ML lifecycle) guide and tutorial for data engineers](https://mageai.notion.site/The-definitive-end-to-end-machine-learning-ML-lifecycle-guide-and-tutorial-for-data-engineers-ea24db5e562044c29d7227a67e70fd56?pvs=4).