# Regression Model Comparison on Diabetes Progression
Submitted by: Arianne A. Reyes for Physics 215 Project

Date: May 26, 2025 Monday

## Project Summary ##

## I. Project Description and Motivation
This project aims to conduct a supervised machine learning experience by simulating regression models on the **diabetes dataset** and evaluate the robustness of each model against corrupted data. **Diabetes** is a medical condition where a person's blood sugar is too high as a result of insulin complications. The IDF Diabetes Atlas (2025) reports that 1 in 9 adults in age ranges between 20-79 years old are living with diabetes or roughly 590 million people worldwide. Having regression models can help **predict future outcomes** by using existing medical data (BMI, insulin level, etc). These models can aid hospitals and clinics in prioritizing high-risk patients and anticipate further complications.

## II. Project Objectives
A. **Preprocess real-world medical data** to predict **disease progression** after **one year** based on patients' baseline measurement or features.

B. **Train and compare the following regression models**:
  1. Linear Regression
  2. Ridge Regression
  3. Lasso Regression
  4. Elastic Net Regression

C. **Visualize model performance** using 5-Fold Cross Validation. This prevents overfitting and gives a reliable estimate for the performance of the regression models.

D. **Evaluate the best model** on test data. The mean squared error (MSE) is used and compared against each model. A lower MSE value means a better performance.

E. **Test model robustness** by *corrupting a data point* on purpose (or creating an outlier) and observing the effect on the accuracy of regression models.

## III. Dataset and Variables
Source: `sklearn.datasets.load_diabetes()`

Input features (x): 10 standardized variables (age, sex, BMI, blood pressure, and six other blood serum measurements.)

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

## VI. Final Project Results and Insights

The bar graph shows an MSE 5-Fold cross validation with the linear regression, ridge, LASSO, elastic net. The four models performed similarly against the original data with an MSE value of ~3100. Meanwhile, after corrupting the data, **elastic net is the most robust with an MSE of 3700** and linear regression is the least robust with an MSE of ~4200. This demonstrates the effects of regularization to reduce the noise in the data.

![image](https://github.com/user-attachments/assets/f29ac47b-11bc-4749-a35e-6cff62ddfdaa)

Comparing the graphs showing the Predicted versus Actual on the Test Set, **elastic net** also shows better performance with the lowest test MSE of 2866.20. Moreover, regularized models generally outperform the linear regression.

![image](https://github.com/user-attachments/assets/b64c95a3-909c-41e7-96df-5003a896c593)

**Takeaway**. The following linear regression models effectively illustrates diabetes progression one year after the baseline using medical records on their BMI, blood pressure, age, and blood serum measurements. This information can aid doctors in identifying individuals for early intervention and monitoring. The scatter plots in the actual vs predicted graphs visualizes how a regression model closely tracks the outcome for a real patient. In terms of noisy or corrupted data that can be a result of missing/incomplete medical records, elastic net is the most suitable regression model in creating reliable predictions.


