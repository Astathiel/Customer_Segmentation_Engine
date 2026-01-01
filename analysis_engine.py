import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta


def generate_dummy_data_file(n_customers=200):
    np.random.seed(42)
    customer_ids = np.arange(1000, 1000 + n_customers)
    
    recency = np.random.randint(1, 365, n_customers)
    frequency = np.random.randint(1, 50, n_customers)
    montery = np.random.randint(20, 5000, n_customers)
    
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Recency_Days_Ago': recency,
        'Frequency_Purchases': frequency,
        'Monetary_Total_Spent': montery
    })
    
    data.to_csv('dummy_data.csv', index=False)
    print("Dummy data file 'dummy_data.csv' generated.")
    return data

def run_segmentation():
    df = generate_dummy_data_file() # <--- Replace with your own data or comment out/remove
    # df = pd.read_csv('dummy_rfm_data.csv') # < --- Replace with your own csv file path
    
    features = df[['Recency_Days_Ago', 'Frequency_Purchases', 'Monetary_Total_Spent']]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_summary = df.groupby('Cluster').agg({
        'Recency_Days_Ago': 'mean',
        'Frequency_Purchases': 'mean',
        'Monetary_Total_Spent': ['mean', 'count']
    }).round(2)
    
    print("---- Customer Segmentation Summary ---")
    print(cluster_summary)
    
    def label_customer(row):
        if row['Monetary_Total_Spent'] > 2000 and row['Recency_Days_Ago'] < 30:
            return 'VIP - Retain'
        elif row['Recency_Days_Ago'] > 200:
            return 'High Risk - Churning'
        else:
            return 'Standard Customer - Engage'
    
    df['Status_Tag'] = df.apply(label_customer, axis=1)
    
    high_value_risk = df[(df['Status_Tag'] == 'VIP - Retain')]
    df.to_csv('final_customer_segments.csv', index=False)
    
    print("\nProcessing Complete. Final customer segments saved to 'final_customer_segments.csv'.")
    print(f"identified {len(high_value_risk)} high-value customers at risk of churning.")
    
if __name__ == "__main__":
    run_segmentation()