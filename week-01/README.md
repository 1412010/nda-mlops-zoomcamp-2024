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
