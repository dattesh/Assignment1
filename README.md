# Assignment1
# MLOps Assignment 1 – House Price Prediction

This project implements a complete ML workflow for predicting house prices using the **Boston Housing dataset**.  
The assignment is structured into multiple branches as required:

- **main** → Contains this README and merged code.  
- **dtree** → Contains DecisionTreeRegressor implementation (`train.py`).  
- **kernelridge** → Contains KernelRidge implementation (`train2.py`) and CI pipeline.

---

## 🔧 Setup Instructions

### 1.Create and Activate Conda environment

conda create --name miniconda_env
conda activate miniconda_env

###2. Install dependencies
pip install -r requirements.txt

###3.Run a model
python train.py

###4. kernelridge
python train2.py
