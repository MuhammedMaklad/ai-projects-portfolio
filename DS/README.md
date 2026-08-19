# Data Science Projects

6 projects covering Python fundamentals, NumPy, Pandas, data visualization, ML algorithms, and PCA/clustering.

## Projects

| Project | Notebooks | Description |
|---------|-----------|-------------|
| [python-crash-course](./python-crash-course/) | 2 | Python fundamentals crash course with exercises |
| [numpy-fundamentals](./numpy-fundamentals/) | 4 | NumPy arrays, indexing, operations, exercises |
| [pandas-fundamentals](./pandas-fundamentals/) | 10 | Pandas Series, DataFrames, missing data, groupby, merging, I/O, exercises |
| [visualization](./visualization/) | 24 | Matplotlib, Seaborn, Pandas built-in, Plotly, Cufflinks, geographical plotting, capstone projects |
| [ml-algorithms](./ml-algorithms/) | 13 | KNN, Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM with projects |
| [pca-clustering](./pca-clustering/) | 5 | K-Means clustering, PCA for visualization and speedup, clustering project |

## Detailed Contents

### python-crash-course
- `01-Python Crash Course.ipynb` - Python basics
- `02-Python Crash Course Exercises.ipynb` - Practice exercises

### numpy-fundamentals
- `01-NumPy Arrays.ipynb` - Array creation and basics
- `02-Numpy Indexing and Selection.ipynb` - Indexing, slicing, boolean masking
- `03-Numpy Operations.ipynb` - Arithmetic, broadcasting, universal functions
- `04-Numpy Exercises.ipynb` - Practice problems

### pandas-fundamentals
- `01-Introduction to Pandas.ipynb` - Pandas overview
- `02-Series.ipynb` - Series creation and manipulation
- `03-DataFrames.ipynb` - DataFrame basics
- `04-Missing Data.ipynb` - Handling NaN values
- `05-Groupby.ipynb` - GroupBy operations
- `06-Merging, Joining, and Concatenating.ipynb` - Combining DataFrames
- `07-Operations.ipynb` - Common DataFrame operations
- `08-Data Input and Output.ipynb` - Reading/writing CSV, Excel, SQL, JSON
- `01-SF Salaries Exercise.ipynb` - Salary data analysis exercise
- `03-Ecommerce Purchases Exercise.ipynb` - E-commerce data exercise

### visualization (24 notebooks)
**Matplotlib (4)**
- `01-Matplotlib Concepts Lecture.ipynb`
- `02-Matplotlib Exercises.ipynb`
- `03-Matplotlib Exercises - Solutions.ipynb`
- `04-Advanced Matplotlib Concepts.ipynb`

**Seaborn (8)**
- `01-Distribution Plots.ipynb`
- `02-Categorical Plots.ipynb`
- `03-Matrix Plots.ipynb`
- `04-Grids.ipynb`
- `05-Regression Plots.ipynb`
- `06-Style and Color.ipynb`
- `07-Seaborn Exercises.ipynb`
- `08-Seaborn Exercises - Solutions.ipynb`

**Pandas Built-in (3)**
- `01-Pandas Built-in Data Visualization.ipynb`
- `02-Pandas Data Visualization Exercise.ipynb`
- `03-Pandas Data Visualization Exercise - Solutions.ipynb`

**Plotly & Cufflinks (1)**
- `01-Plotly and Cufflinks.ipynb`

**Geographical Plotting (3)**
- `01-Choropleth Maps.ipynb`
- `02-Choropleth Maps Exercise.ipynb`
- `03-Choropleth Maps Exercise - Solutions.ipynb`

**Capstone Projects (5)**
- `01-911 Calls Data Capstone Project.ipynb`
- `03-Finance Project.ipynb`
- `main.ipynb`
- `Untitled.ipynb`
- `visualization.ipynb`

### ml-algorithms (13 notebooks, 6 projects with exercises)
**KNN (2)**
- `01-K Nearest Neighbors with Python.ipynb`
- `02-K Nearest Neighbors Project.ipynb`

**Linear Regression (2)**
- `01-Linear Regression with Python.ipynb`
- `02-Linear Regression Project.ipynb`

**Logistic Regression (2)**
- `01-Logistic Regression with Python.ipynb`
- `02-Logistic Regression Project.ipynb`

**Decision Trees & Random Forest (3)**
- `01-Decision Trees and Random Forests in Python.ipynb`
- `DT&RandomForst.ipynb`
- `02-Decision Trees and Random Forest Project.ipynb`

**SVM (2)**
- `01-Support Vector Machines with Python.ipynb`
- `02-Support Vector Machines Project.ipynb`

**Additional**
- `main.ipynb`
- `main (3).ipynb`

### pca-clustering (5 notebooks)
**K-Means (3)**
- `01-K Means Clustering with Python.ipynb`
- `KMeans_Clustering.ipynb`
- `02-K Means Clustering Project.ipynb`

**PCA (2)**
- `01-Principal Component Analysis.ipynb`
- `PCA_for visualization_&_speed_Up_ML_models (1).ipynb`

## Getting Started

```bash
# Navigate to a project
cd pandas-fundamentals

# Open in Jupyter (auto-syncs with .py)
jupyter lab 01-Introduction to Pandas.ipynb

# Or edit .py in your IDE
code 01-Introduction to Pandas.py
```

All notebooks use Jupytext paired format (`.py` + `.ipynb`).

## Learning Path Suggestion

1. **Foundation**: `python-crash-course/` → `numpy-fundamentals/` → `pandas-fundamentals/`
2. **Visualization**: `visualization/` - Matplotlib → Seaborn → Plotly → Geographical → Capstones
3. **ML Algorithms**: `ml-algorithms/` - KNN → Linear/Logistic Regression → Trees/Forest → SVM
4. **Dimensionality Reduction**: `pca-clustering/` - K-Means → PCA