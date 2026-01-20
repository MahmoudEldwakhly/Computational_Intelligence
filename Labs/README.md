# CSE473 – Computational Intelligence Labs

This directory contains a collection of laboratory assignments for the **CSE473: Computational Intelligence** course.  
The labs focus on building strong foundations in numerical computation, data analysis, optimization, machine learning, and neural networks through hands-on Python implementations.

---

## Lab 01 — Solving Linear Systems
**Topic:** Linear algebra and numerical computation

**Description:**
- Solves systems of linear equations of the form **A x = b**.
- Handles both square systems (m = n) and overdetermined systems (m > n).
- Uses:
  - Exact solutions via matrix inversion when possible.
  - Least-squares solutions for overdetermined systems.

**Key concepts:**
- Matrix formulation of linear systems
- Numerical stability
- Least-squares approximation

**Tools:** NumPy

---

## Lab 02 — Dataset Analysis with Pandas (Iris Dataset)
**Topic:** Data analysis and preprocessing

**Description:**
- Loads the Iris dataset into a Pandas DataFrame.
- Explores dataset structure and summary statistics.
- Detects and handles missing values.
- Computes descriptive statistics and correlations.
- Groups samples by species for comparative analysis.
- Visualizes feature distributions using histograms.

**Key concepts:**
- Data loading and inspection
- Missing data handling
- Statistical analysis
- Exploratory data analysis (EDA)

**Tools:** Pandas, Matplotlib

---

## Lab 03 — Function Visualization and Extremum Detection
**Topic:** Mathematical modeling and visualization

**Description:**
- Defines multiple 2D mathematical functions.
- Generates surface and contour plots.
- Detects local maxima and minima using numerical gradients.
- Visualizes extrema on both 3D surfaces and 2D contour plots.

**Key concepts:**
- Numerical differentiation
- Gradient-based analysis
- Visualization of optimization landscapes

**Tools:** NumPy, Matplotlib (2D & 3D)

---

## Lab 04 — Solving Nonlinear Systems with Optimization
**Topic:** Nonlinear optimization

**Description:**
- Solves a system of nonlinear equations by reformulating it as a minimization problem.
- Uses least-squares optimization to find solutions.
- Analyzes convergence behavior and optimization settings.

**Key concepts:**
- Nonlinear equation systems
- Objective function formulation
- Optimization-based solving

**Tools:** NumPy, SciPy

---

## Lab 05 — Double Moon Dataset (Binary Classification)
**Topic:** Supervised learning and decision boundaries

**Description:**
- Generates a synthetic double-moon dataset.
- Splits data into training, validation, and test sets.
- Trains and evaluates:
  - Linear classifier (Logistic Regression)
  - Multi-Layer Neural Network (MLNN)
- Visualizes:
  - Decision boundaries
  - Loss curves
  - Confusion matrices
  - ROC and Precision–Recall curves

**Key concepts:**
- Linear vs non-linear classification
- Overfitting and generalization
- Performance visualization

**Tools:** Scikit-learn, Matplotlib

---

## Lab 06 — Multi-Class Double Moon Classification
**Topic:** Multi-class classification

**Description:**
- Extends the double-moon dataset to four classes.
- Trains and compares:
  - Multi-Class Support Vector Machine (MCSVM)
  - Neural network classifier (TensorFlow)
- Includes feature scaling, training visualization, and decision boundary plots.

**Key concepts:**
- Multi-class learning
- Kernel-based methods
- Neural networks for classification

**Tools:** Scikit-learn, TensorFlow, Matplotlib

---

## Skills Developed from Labs
- Numerical linear algebra
- Data cleaning and preprocessing
- Exploratory data analysis
- Optimization techniques
- Classification (linear, SVM, neural networks)
- Visualization and interpretation of model behavior
