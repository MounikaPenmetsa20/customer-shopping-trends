# -------------------------------
# Customer Shopping Trends Project
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# ===== Step 1: Load data =====
DATA_PATH = r"C:\Users\Dell\Documents\Custom Office Templates\shopping_trends.csv"  # <-- your dataset path here

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

# ===== Step 2: Basic cleaning =====
df.drop_duplicates(inplace=True)

# Make a Total column directly from Purchase Amount (USD)
df['Total'] = pd.to_numeric(df['Purchase Amount (USD)'], errors='coerce')

# Age grouping
if 'Age' in df.columns:
    df['age_group'] = pd.cut(df['Age'],
                             bins=[0,18,25,35,50,200],
                             labels=['<18','18-24','25-34','35-49','50+'],
                             right=False)

# ===== Step 3: Simple insights =====
print("\nTop products by total spent:")
if 'Item Purchased' in df.columns:
    top_products = (df.groupby('Item Purchased')['Total']
                      .sum()
                      .sort_values(ascending=False)
                      .head(10))
    print(top_products)

print("\nCategory summary (sum, count, mean):")
if 'Category' in df.columns:
    cat_summary = (df.groupby('Category')['Total']
                     .agg(['sum','count','mean'])
                     .sort_values('sum',ascending=False)
                     .head(10))
    print(cat_summary)

print("\nPurchase frequency per customer:")
if 'Customer ID' in df.columns:
    freq = (df.groupby('Customer ID')
              .agg(total_spent=('Total','sum'),
                   avg_purchase=('Total','mean'),
                   purchases=('Total','count'))
              .sort_values('purchases',ascending=False)
              .head(10))
    print(freq)

# ===== Step 4: Segmentation (frequency & monetary) =====
if 'Customer ID' in df.columns:
    rfm = df.groupby('Customer ID').agg(
        frequency=('Total','count'),
        monetary=('Total','sum')
    ).reset_index()

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[['frequency','monetary']].fillna(0))

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['segment'] = kmeans.fit_predict(X)

    os.makedirs("outputs", exist_ok=True)
    rfm.to_csv("outputs/rfm_segments.csv", index=False)
    print("\nSaved RFM segments to outputs/rfm_segments.csv")
    print(rfm.head())

# ===== Step 5: Plot top products =====
if 'Item Purchased' in df.columns:
    plt.figure(figsize=(8,5))
    top_products.sort_values(ascending=True).plot(kind='barh')
    plt.title("Top Products by Total Spent")
    plt.xlabel("Total Spent (USD)")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/top_products.png")
    print("Saved plot to outputs/top_products.png")
