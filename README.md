# customer-shopping-trends
# Customer Shopping Trends

**Python script to analyze customer shopping patterns & segment customers.**

This project takes a customer shopping dataset (CSV) and:

- Cleans and prepares the data (remove duplicates, parse dates, calculate totals).
- Extracts insights like:
  - Peak shopping hours
  - Top-selling products
  - Age group and category preferences
- Computes customer purchase frequency and spending.
- Builds RFM (Recency, Frequency, Monetary) features and clusters customers into segments using KMeans.

The outputs can be used by marketing or sales teams to target customers, plan promotions, or understand shopping behavior.

---

## 🚀 How to Run

1. **Download a dataset**  
   Get a CSV from [Kaggle](https://www.kaggle.com/datasets) (for example: *Online Retail II* or *Customer Personality Analysis*).

2. **Save the CSV**  
   Put the file anywhere on your machine, e.g. `"C:\Users\Dell\Documents\Custom Office Templates\shopping_trends.csv"`.

3. **Download this script**  
   [customer_shopping_trends.py](./customer_shopping_trends.py)

4. **Install dependencies**  
   ```bash
   pip install pandas numpy matplotlib scikit-learn
