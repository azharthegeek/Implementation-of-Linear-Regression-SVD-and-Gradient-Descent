# Linear Regression: SVD and Gradient Descent Implementation

## Project Overview
Implementation of Linear Regression using multiple mathematical approaches including Ordinary Least Squares (OLS), Singular Value Decomposition (SVD), and Gradient Descent optimization on the Life Expectancy dataset.

**Authors:** Muhammad Azhar (24K-7606), Hamza Mughal (25K-7623)  
**Course:** Mathematical Foundation for Data AI

## Dataset
- **Dataset:** Life Expectancy Data.csv
- **Target Variable:** Life expectancy (years)
- **Features:** Health, economic, and social indicators including schooling, GDP, mortality rates, etc.

## Prerequisites

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

### 1. **Setup**
Ensure all required libraries are installed and the dataset `Life Expectancy Data.csv` is in the same directory as the notebook.

### 2. **Run the Notebook**
```bash
jupyter notebook project.ipynb
```
Or open in VS Code and run cells sequentially.

### 3. **Execute Cells in Order**
The notebook is organized into sections:
- **Data Loading & Exploration** - Load and inspect the dataset
- **Preprocessing** - Handle missing values, encode categorical variables, scale features
- **Task 2: OLS Regression** - Ordinary Least Squares implementation
- **Task 3: SVD Solution** - Singular Value Decomposition approach
- **Task 4: Gradient Descent** - Optimization using gradient descent
- **Task 5: PCA & Dimensionality Reduction** - Principal Component Analysis

## Project Structure

```
.
├── project.ipynb              # Main Jupyter notebook
├── Life Expectancy Data.csv   # Dataset (required)
├── README.md                  # This file
└── (Generated files after running):
    ├── X_train_scaled.npy
    ├── X_test_scaled.npy
    ├── y_train.npy
    ├── y_test.npy
    └── feature_names.txt
```

## Key Features

### Data Preprocessing
- Missing value imputation (median for numeric, mode for categorical)
- Label encoding for categorical variables
- 80-20 train-test split
- Feature standardization (zero mean, unit variance)

### Implementation Approaches
1. **OLS Regression** - Closed-form solution using normal equations
2. **SVD-Based Solution** - Numerically stable matrix decomposition
3. **Gradient Descent** - Iterative optimization algorithm
4. **PCA** - Dimensionality reduction and feature analysis

### Analysis & Visualization
- Correlation heatmaps
- Distribution plots
- Scatter plots for feature relationships
- Model performance comparisons

## Output

After running the notebook, you'll get:
- Preprocessed data files (`.npy` format)
- Comprehensive visualizations
- Model performance metrics (R², MSE, RMSE)
- Comparative analysis between different approaches

## Notes

- Run all cells sequentially from top to bottom
- The notebook includes detailed markdown explanations for each step
- Generated `.npy` files can be used for future analysis without reprocessing
- Ensure sufficient memory for large matrix operations

## Results

The notebook compares multiple regression approaches:
- Mathematical derivations and implementations
- Performance metrics for each method
- Visualization of predictions vs actual values
- Analysis of feature importance


