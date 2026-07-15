# Product Segmentation Dashboard

## Overview

This Streamlit application performs product segmentation using multiple clustering algorithms and business-oriented features extracted from sales transactions.

The application allows users to:

- Upload a sales dataset (CSV or Excel)
- Generate product-level features automatically
- Apply different clustering algorithms
- Compare clustering methods
- Tune K-Means and DBSCAN parameters
- Visualize products using PCA
- Analyze cluster profiles
- Export segmentation results

---

## Features

### Feature Engineering

The application computes:

- Number of Sales Transactions
- Total Quantity Sold
- Total Revenue
- Average Sale Value
- Days Since Last Sale
- Product Lifetime
- Average Days Between Sales
- Number of Distinct Customers
- Average VWAP
- VWAP Volatility
- Return Rate
- RFM Score

---

## Supported Clustering Methods

- K-Means
- Hierarchical Clustering
- Gaussian Mixture Model (GMM)
- DBSCAN
- HDBSCAN

---

## Model Evaluation

The application includes:

- Silhouette Score
- Elbow Method
- Automatic K-Means tuning
- Automatic DBSCAN tuning
- PCA visualization
- Cluster comparison using Adjusted Rand Index (ARI)

---

## Input Data

The dataset must contain the following columns:

- Oid
- Code
- Label1
- Date
- Quantity
- Amount
- Type
- VWAP

Optional:

- Code-2 (Customer ID)

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/product-segmentation.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

---

## Output

The dashboard provides:

- Product clusters
- PCA visualization
- Cluster profiles
- Business labels
- Downloadable CSV results

---

## Author

Baamara Yahia 
