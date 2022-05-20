#### RFM analysis

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np

os.getcwd()
os.chdir("/home/aumaron/Desktop/datasets")

previous_year = pd.read_excel("Online_Retail_II.xlsx", engine="openpyxl", sheet_name="Year 2009-2010")
current_year = pd.read_excel("Online_Retail_II.xlsx", engine="openpyxl", sheet_name="Year 2010-2011")
previous_year.columns
current_year.columns

frames = [previous_year, current_year]
data = pd.concat(frames)
data.dropna(subset=["Customer ID", "Invoice", "InvoiceDate"], how="any", inplace=True)
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
data["Total Price"] = data["Quantity"] * data["Price"]
# Save the date which will be used to calculate how recent a given user's purchase was
reference_date = data["InvoiceDate"].max()
# Build an RFM matrix
# Recency -> Assuming 5 days of recency -> assuming Days aggregation
# Aggregating Daily
rfm_matrix = data.groupby(["Customer ID"]).aggregate(
    {
        "InvoiceDate": lambda date: (reference_date - date.max()).days,
        "Invoice": "nunique",
        "Total Price": "sum"
    }
).reset_index()

rfm_matrix.rename(
    columns={
        "InvoiceDate": "Recency",
        "Invoice": "Frequency",
        "Total Price": "MonetaryValue"
    },
    inplace=True
)

rfm_matrix.to_csv("rfm_actual.csv")

# Normalising the RFM matrix
numerical_columns = ["Recency", "Frequency", "MonetaryValue"]
for i in numerical_columns:
    scale = StandardScaler().fit(rfm_matrix[[i]])
    rfm_matrix[i] = scale.transform(rfm_matrix[[i]])

rfm_matrix.to_csv("rfm_normalised.csv")


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
cluster.fit_predict(rfm_matrix.iloc[:, 1:len(rfm_matrix.columns)+1].to_numpy())

print(cluster.labels_)
