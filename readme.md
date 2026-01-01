# Customer Segmentation & Churn Risk Engine

## Project Overview
This tool automates the analysis of customer behavior using **Unsupervised Machine Learning**. It processes transaction data to group customers into distinct "personas" and automatically flags high-value users who are at risk of churning (or customer attrition = losing customers).

Instead of manual spreadsheet analysis, this engine uses **K-Means Clustering** to perform **RFM Analysis** (Recency, Frequency, Monetary) at scale.

## Key Features
* **Automated Data Generation:** Simulates a realistic e-commerce dataset (no external CSVs required).
* **Machine Learning Clustering:** Uses Scikit-Learn's K-Means to find hidden patterns in customer behavior.
* **Risk Detection Logic:** Applies business rules to identify "VIP" vs. "High Risk" customers.
* **Actionable Export:** Saves the final analyzed list to `final_customer_segments.csv` for marketing teams.

## Technologies Used
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, StandardScaler)

## Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Astathiel/Customer_Segmentation_Engine.git](https://github.com/Astathiel/Customer_Segmentation_Engine.git)
    cd Customer_Segmentation_Engine
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run
Execute the main script from your terminal:
```bash
python analysis_engine.py