# EPFL Machine Learning Course - Project 1

**Team Members**: 

- [Ahmed Abdelmalek](https://people.epfl.ch/ahmed.abdelmalek)  
- [Nestor Lomba](https://people.epfl.ch/nestor.lombalomba)  
- [Khaled Miraoui](https://people.epfl.ch/mohamed.miraoui)  

## Project Overview

This project is part of the EPFL Machine Learning Course (Fall 2024). Our goal is to predict the risk of developing coronary heart disease (MICHD) based on data from the Behavioral Risk Factor Surveillance System (BRFSS). Using a real-world medical dataset containing health-related data of over 300,000 individuals, we implemented multiple machine learning techniques to build and optimize predictive models.

## Project Objectives

- Conduct **Data Preprocessing and Cleaning** to handle missing values and scale the data effectively.
- Implement and test various machine learning models:
  - Linear regression with gradient descent and stochastic gradient descent
  - Least squares regression
  - Ridge regression
  - Logistic regression
  - Regularized logistic regression
- Perform **Feature Selection** to enhance model performance, focusing on high-correlation features with the target variable.
- **Optimize Models** to achieve the best F1 score and accuracy on the test dataset.
- Generate predictions for submission to the [AIcrowd competition platform](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

## Dataset

We use three main datasets provided in CSV format:
- `x_train.csv`: Training feature set
- `y_train.csv`: Binary labels indicating the presence or absence of coronary heart disease
- `x_test.csv`: Testing feature set without labels for prediction submission

These datasets contain health-related demographic, behavioral, and lifestyle features relevant to coronary heart disease risk.

## Methodology

### Data Preprocessing

1. **Data Cleaning**: Features with a high proportion of NaN values were removed to maintain data quality. Remaining NaN values were filled with feature means.
2. **Feature Engineering**: We expanded certain categorical features based on provided descriptions to enhance model interpretability and accuracy.
3. **Standardization**: All features were standardized to zero mean and unit variance.
4. **Feature Selection**: Using Pearson correlation, we retained features with significant correlation to the target variable.

### Model Implementations

Each model is implemented from scratch in `implementations.py`, using only NumPy for matrix operations as per project guidelines. The following machine learning methods were tested:

- **Least Squares Regression**: Finds the optimal weights to minimize the mean squared error (MSE) using the normal equation.
- **Linear Regression (Gradient Descent & Stochastic Gradient Descent)**: Uses iterative methods to minimize the MSE.
- **Ridge Regression**: Adds L2 regularization to the least squares regression to reduce overfitting.
- **Logistic Regression**: A binary classifier for predicting coronary heart disease risk.
- **Regularized Logistic Regression**: Adds L2 regularization to logistic regression, enhancing generalization and preventing overfitting.

### Model Training and Evaluation

In the `run.ipynb` notebook, we performed the following:

1. **Model Training**: We trained each model on a balanced subset of the data to ensure a consistent positive-to-negative class ratio.
2. **Hyperparameter Optimization**: Experimented with various values for learning rate, regularization strength, and feature selection thresholds to maximize F1 scores.
3. **Model Evaluation**: We evaluated each model on F1 score and accuracy, using validation data for unbiased assessment. Our best submission achieved an F1 score of **0.447** on AIcrowd.

## Code Structure

- `implementations.py`: Contains the core machine learning functions:
  - `least_squares(y, tx)`
  - `logistic_regression(y, tx, initial_w, max_iters, gamma)`
  - `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)`
  - `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)`
  - `ridge_regression(y, tx, lambda_)`
  - `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)`
- `run.ipynb`: Main notebook that includes data loading, preprocessing, model training, and performance evaluation.
- `predictions.csv`: Contains the predictions from our best model for submission on AIcrowd.

## Getting Started

1. **Clone the repository**: 
   ```bash
   git clone https://github.com/username/project1.git
2. Install dependencies (NumPy and Matplotlib are required).
3. Run the Notebook: Open run.ipynb in Jupyter Notebook and execute the cells sequentially to reproduce our results and visualize model comparisons.

## Results

- Our final submission, `predictions.csv`, achieved an F1 score of **0.447** on the AIcrowd leaderboard.
- Detailed plots comparing model performances and highlighting the impact of different feature selection thresholds are available in the notebook.

## Platform Submission

Predictions generated from our best model were submitted to the AIcrowd platform for evaluation. The submission file `predictions.csv` contains the predicted labels for the test dataset, formatted according to AIcrowd requirements. We experimented with different models and configurations to reach our highest score, continuously refining the feature selection and regularization methods based on leaderboard feedback.

## Conclusion

This project demonstrates the importance of data preprocessing, feature selection, and hyperparameter tuning in machine learning pipelines. By implementing models from scratch, we gained a deeper understanding of each algorithm's mechanics, allowing us to effectively tackle real-world data.

---
**Disclaimer**: This project is for educational purposes and part of the EPFL Machine Learning course.
