# EDA to MLOps with ZenML & MLflow 🚀

This project showcases an end-to-end machine learning pipeline built using **ZenML**, **MLflow**, and **pandas**. It starts from **exploratory data analysis (EDA)** and progresses to a **production-grade MLOps workflow**, demonstrating key components like data preprocessing, model training, evaluation, and experiment tracking.

This project follows a modular, template-based structure to demonstrate a **scalable and production-ready machine learning workflow**


## 🔧 Environment Setup

You can use either Python’s built-in `venv` or Anaconda’s `conda` to manage dependencies.


### ✅ Option 1: Using `venv`

#### 1. Create virtual environment
```bash
python -m venv mlOps
```
#### 2. Activate Environment
```bash
source mlOps/bin/activate
```
#### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

### ✅ Option 2: Using `conda`

#### 1. Create virtual environment
```bash
conda create -n mlOps python=3.11
```
#### 2. Activate Environment
```bash
conda activate mlOps
```
#### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

## 🧪 MLflow Integration with ZenML

To run deployment-related scripts (e.g., `run_deployment.py`), you must configure a ZenML stack that includes **MLflow** as both the **experiment tracker** and **model deployer**.

### 1. Install MLflow Integration

Install the MLflow integration for ZenML:

```bash
zenml integration install mlflow -y
```

### 2. Register an MLflow Experiment Tracker and Deployer
```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
```

### 3. Register and Set the ZenML Stack
```bash
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## 📊 Exploratory Data Analysis (EDA)

This project includes an interactive EDA tool for analyzing structured tabular datasets.

Once you set up the environment, you can explore your dataset interactively using the `run_data_analysis.py` script.
 
The default dataset used is a **House Price Predictor dataset**, which includes numerical and categorical features like lot area, number of rooms, year built, and sale price.

You can use your own dataset in `.csv` or `.zip` format as well.


### Dataset Format

You can provide your dataset in one of the following formats:

- A **CSV** file (e.g., `data.csv`)
- A **ZIP** file containing a CSV (e.g., `data.zip` with `data.csv` inside)

### Running the Script

To start the EDA process, run:

```bash
python run_data_analysis.py
```

The script will prompt:

```bash
Enter path to dataset file (zip or csv)
```
Once the dataset is loaded, you'll be shown a list of available analysis options, such as:

```bash
Select Analysis

1. Data Inspection
2. Missing Values Analysis
3. Univariate Analysis
4. Bivariate Analysis
5. Multivariate Analysis
6. Exit
```
Enter the number corresponding to the analysis you'd like to perform. You can run multiple analyses in a single session.

### 🔍 Example Exploratory Data Analysis Visualizations

Here are some example visualizations you might see when using the EDA script:

---
> ![Feature Distribution](images/univariate.png)  
> *Histogram showing feature value distribution.*

> ![Correlation Heatmap](images/heatmap.png)  
> *Correlation matrix of numerical features.*

---

## 🚀 Training, Deployment & Prediction

After exploring your data with the EDA tool, you can proceed to train, deploy, and test your machine learning model using the following scripts.

### 🧠 1. Training the Model

Run the training pipeline using:

```bash
python run_training.py --config 'path/to/config.json'
```
- The config.json file contains pipeline parameters 
- You can modify the existing JSON file to match your dataset and training requirements.

During the training phase (`run_training.py`), the logs will display the **expected input features** that the model was trained on.

### 🌐 2. Deploying the Mode

You can deploy the model using:

```bash
python run_deployment.py --config 'path/to/config.json'
```

- This uses your trained model and pushes it to a deployment server.
- After deployment, you will receive a prediction URL.

### 🔍 3. Running Inference

To perform predictions with your deployed model:

```bash
python run_prediction.py
```

You will be prompted to:

- Enter the prediction URL (received from the deployment step)

- Provide the path to a sample_data.json file containing the input features

Modify `sample_test.json` according to the expected features obtained during training phase.

### 🧩 Inference Pipeline (Optional)

In addition to the deployed prediction endpoint, an inference pipeline is included to support data from:

- External APIs

- Databases

- Streaming services

- Files or batch inputs

You can use this to perform inference beyond CLI, such as integrating with a data service or dashboard.

To customize how and from where the data is fetched for inference, modify the `dynamic_importer.py` module.