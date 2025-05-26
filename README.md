# Regression Model Comparison on a Diabetes Dataset
Submitted by: Arianne A. Reyes for Physics 215 Project

Date: May 26, 2025 Monday

## Project Summary ##

## I. Project Description and Motivation
This project aims to conduct a supervised machine learning experince by simulating regression models on the **diabetese dataset** which was taken from scikit-learn. **Diabetes** is a medical condition where a person's blood sugar is too high as a result of insulin complications. The IDF Diabetes Atlas (2025) reports that 1 in 9 adults in age ranges between 20-79 years old are living with diabetes or roughly 590 million people worldwide! Having regression models can help **predict future outcomes** by using existing medical data (BMI, insulin level, etc). These models can aid hospitals and clinics in prioritizing high-risk patients and anticipate further complications.

## II. Project Objectives
A. **Preprocess real-world medical data** to predict **disease progression** based on patient features.

B. **Train and compare multiple regression models**:
  1. Linear Regression
  2. Ridge Regression
  3. Lasso Regression
  4. Elastic Net Regression

C. **Visualize model performance** using 5-Fold Cross Validation.

D. **Evaluate the best model** on test data.

E. **Test model robustness** by *corrupting a data point* on purpose (or creating an outlier) and observing the effect on the accuracy of regression models.

## III. Dataset and Variables
Source: `sklearn.datasets.load_diabetes()`

Features (X): 10 standardized variables (age, BMI, blood pressure, etc.)

Target (y): Quantitative measure of disease progression one year after baseline.

## IV. Project Workflow

├── main.ipynb ← Main code to run the project via Jupyter Notebook

├── data/ 

    ├── raw/ ← Unmodified downloaded data

    ├── processed/ ← Cleaned and split training/testing data
  
    └── final/ ← Final evaluation predictions and corrupted data


├── src/

    ├── make_dataset.py ← Downloads and saves raw diabetes dataset

    ├── preprocess.py ← Preprocessing and 80/20 train-test split

    ├── regression_models.py ← Trains the 4 regression models (linear, ridge, lasso, and elastic net) with 5-fold CV

    └── evaluation.py ← Final model testing + corrupt data evaluation

## V. How to Run This?
1. Clone this GitHub repository: git clone https://github.com/Arianne-Reyes/ReyesAA-Physics215Project

2. Using Anaconda command prompt to go to the cloned folder:
   cd "C:\Users\XXX\XXX\Physics215Reyes\ReyesAA-Physics215Project"
   
3. Launch "jupyter notebook"

4. Run "main.ipynb" (run all cells)

This will automically create and save the data in a new "data" folder with three corresponding subfolders: "raw", "processed", and "final"

## VI. Final Project Insights and Take-aways

