import random
from faker import Faker
import os
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import openai
import numpy as np
from datetime import datetime, timedelta

# Entities: AI will detect entities like supplier names, products, symptoms, locations, and reports.

# suppliers: Extract companies, the products they supply, and any recall details.
# products: Identify product names, types (e.g., food item), recall status, and generate a product embedding.
# stores: Extract names of stores and locations associated with outbreaks.
# symptoms: Identify symptoms, severity levels, and generate symptom embeddings.
# outbreak_report: Extract outbreak details such as start and end dates, reported cases, hospitalizations, deaths, and location.

# Initialize Faker for generating sample data
fake = Faker()

# Connect to Cassandra

 # Read environment variables
secure_connect_bundle_path = os.getenv('ASTRA_DB_SECURE_CONNECT_practice')
application_token = os. getenv('ASTRA_DB_TOKEN_practice')

session = Cluster(
    cloud={"secure_connect_bundle": secure_connect_bundle_path},
    auth_provider=PlainTextAuthProvider("token", application_token),
).connect()

keyspace = 'foodtech'
# Use the keyspace
session.set_keyspace(keyspace)



# Insert data into suppliers table
for _ in range(20):
    name = fake.company()
    product = random.choice(['Beef Patties', 'Slivered Onions', 'Lettuce'])
    recall_status = random.choice(['Recalled', 'Not Recalled'])
    recall_date = fake.date_between(start_date='-1y', end_date='today') if recall_status == 'Recalled' else None
    session.execute(
        """
        INSERT INTO suppliers (name, product, recall_status, recall_date)
        VALUES (%s, %s, %s, %s)
        """,
        (name, product, recall_status, recall_date)
    )

# Insert data into products table
for _ in range(20):
    name = fake.word().capitalize()
    type_ = random.choice(['Burger', 'Salad', 'Drink'])
    recall_status = random.choice(['Recalled', 'Not Recalled'])
    embedding = [random.uniform(-1, 1) for _ in range(10)]  # 10-dimensional embedding
    session.execute(
        """
        INSERT INTO products (name, type, recall_status, embedding)
        VALUES (%s, %s, %s, %s)
        """,
        (name, type_, recall_status, embedding)
    )

# Insert data into stores table
for _ in range(20):
    name = fake.company()
    location = fake.city()
    session.execute(
        """
        INSERT INTO stores (name, location)
        VALUES (%s, %s)
        """,
        (name, location)
    )

# Insert data into symptoms table
for _ in range(20):
    name = fake.word().capitalize()
    severity = random.choice(['Mild', 'Moderate', 'Severe'])
    embedding = [random.uniform(-1, 1) for _ in range(10)]  # 10-dimensional embedding
    session.execute(
        """
        INSERT INTO symptoms (name, severity, embedding)
        VALUES (%s, %s, %s)
        """,
        (name, severity, embedding)
    )

# Insert data into outbreak_report table
for _ in range(20):
    name = "Outbreak-" + str(fake.random_int(min=1000, max=9999))
    start_date = fake.date_between(start_date="-1y", end_date="today")
    end_date = start_date + timedelta(days=random.randint(1, 30))
    cases_reported = random.randint(10, 100)
    hospitalizations = random.randint(1, cases_reported // 2)
    deaths = random.randint(0, hospitalizations // 5)
    location = random.choice(['Colorado', 'Kansas', 'Utah', 'Wyoming', 'Idaho'])
    session.execute(
        """
        INSERT INTO outbreak_report (name, start_date, end_date, cases_reported, hospitalizations, deaths, location)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (name, start_date, end_date, cases_reported, hospitalizations, deaths, location)
    )

print("Data inserted successfully.")
