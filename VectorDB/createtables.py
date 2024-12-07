# setx ASTRA_DB_SECURE_CONNECT_practice "D:\TaskL\ecommerce\practicedb\secure-connect-practice.zip"
# setx ASTRA_DB_TOKEN_practice "AstraCS:ttwdbKMrAlQOuZxKIlXaBAQO:521fb04eab516d8e739d3cc04aedefcdb449d173f649725436af576b3d652fea"


import os
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# Read environment variables
secure_connect_bundle_path = os.getenv('ASTRA_DB_SECURE_CONNECT_practice')
application_token = os.getenv('ASTRA_DB_TOKEN_practice')

# Debug prints to check environment variables
print(f"Secure Connect Bundle Path: {secure_connect_bundle_path}")
print(f"Application Token: {application_token}")

# Check if the environment variables are loaded correctly
if not secure_connect_bundle_path or not os.path.exists(secure_connect_bundle_path):
    raise FileNotFoundError(f"Secure connect bundle not found at path: {secure_connect_bundle_path}")

if not application_token:
    raise ValueError("Application token not found in environment variables")

# Connect to the Cassandra database using the secure connect bundle
session = Cluster(
    cloud={"secure_connect_bundle": secure_connect_bundle_path},
    auth_provider=PlainTextAuthProvider("token", application_token),
).connect()


# Use the keyspace
session.set_keyspace('foodtech')

# Create a table
session.execute("CREATE TABLE IF NOT EXISTS ProductImageVectors (ProductId int PRIMARY KEY, ProductDesc text, price int);")

# Insert data
session.execute("INSERT INTO ProductImageVectors (ProductId, ProductDesc, price) VALUES (1, 'Proline Men Cream-Coloured Polo T-Shirt', 3000);")

# Select data
results = session.execute("SELECT * FROM ProductImageVectors;")
for row in results:
    print(row)

df = pd.read_sql_query(results, session)
print(df)



