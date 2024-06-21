# T-Shirts Demand Classifier

## Overview

This project aims to develop a classification model to predict the demand for T-shirts. The project involves data preprocessing, model training, and evaluation using various machine learning techniques. The objective is to build a reliable classifier that can accurately predict T-shirt demand based on the given features.

## Project Structure

- **ML-T-Shirts.ipynb**: This Jupyter Notebook contains the entire workflow for data analysis, preprocessing, model training, and evaluation. It covers steps such as loading data, visualizing data, preprocessing, training various models, and evaluating their performance.
- **requirements.txt**: This file lists all the necessary Python libraries required to run the project. You can install these dependencies using the command `pip install -r requirements.txt`.
- **t-shirts.csv**: This is the dataset used for training and testing the models. It contains various features that describe the T-shirts and their respective demand categories.

## Data Analysis and Preprocessing

1. **Data Loading and Exploration**: The dataset `t-shirts.csv` is loaded and explored for basic statistics, missing values, and initial visualizations to understand the data distribution and relationships between variables.
2. **Data Preprocessing**:
    - **Standardization**: Data is standardized to have zero mean and unit variance.
    - **Normalization**: Data is scaled to a range of [0, 1].
    - **Discretization**: Continuous features are discretized into bins.
    - **Feature Selection**: Important features are selected based on their relevance to the target variable.
    - **Principal Component Analysis (PCA)**: Dimensionality reduction technique applied to reduce the number of features.

## Model Training and Evaluation

Multiple machine learning algorithms are used to train the model:

- **Logistic Regression**: A simple and effective linear model for binary classification.
- **Random Forest**: An ensemble learning method that builds multiple decision trees and merges them for a more accurate and stable prediction.
- **Support Vector Machine (SVM)**: A powerful model for classification tasks, effective in high-dimensional spaces.

## Dependencies

This project depends on several essential Python libraries. To install all the necessary dependencies, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    ```

2. **Navigate to the project directory**:

    ```bash
    cd TShirtsDemandClassifier
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, follow these steps:

1. **Open the Jupyter Notebook**:

    ```bash
    jupyter notebook ML-T-Shirts.ipynb
    ```

2. **Execute the notebook**:

    Run all the cells in the notebook to perform data analysis, preprocessing, model training, and evaluation. This will guide you through each step of the workflow.

## Learning Outcomes

During this project, I enhanced my expertise in several key areas:

- **Data Preprocessing**: Acquired hands-on experience with various techniques such as standardization, normalization, and feature selection to prepare data for machine learning models.
- **Model Training**: Gained a deeper understanding of multiple classification algorithms and their practical applications.
- **Model Evaluation**: Learned to evaluate model performance using cross-validation and relevant metrics to ensure robustness and accuracy.
- **Proficiency in Python and Libraries**: Improved my skills in Python and crucial libraries like Pandas, NumPy, Scikit-learn, and Matplotlib, which are fundamental for data analysis and machine learning.

## Conclusion

This project provided valuable insights into the end-to-end machine learning workflow, from data preprocessing to model evaluation. It underscored the importance of proper data handling and meticulous model tuning to build effective and reliable classifiers. Through this experience, I developed a comprehensive understanding of how to approach and solve classification problems in a systematic and efficient manner.
