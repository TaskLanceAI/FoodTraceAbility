import os
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import openai
import numpy as np

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

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

keyspace = 'foodtech'
# Use the keyspace
session.set_keyspace(keyspace)


# Sample Data
data = {
    "suppliers": [
        {"name": "Taylor Farms", "product": "Slivered Onions", "recall_status": "Voluntary Recall", "recall_date": "2024-10-10"}
    ],
    "products": [
        {"name": "Quarter Pounder", "type": "Burger"},
        {"name": "Slivered Onions", "type": "Ingredient", "recall_status": "Recalled"},
        {"name": "Beef Patties", "type": "Ingredient", "status": "In Use"},
        {"name": "Diced Onions", "type": "Ingredient", "status": "Not Implicated"}
    ],
    "stores": [
        {"name": "McDonald's", "location": "Colorado"},
        {"name": "McDonald's", "location": "Kansas"},
        {"name": "McDonald's", "location": "Utah"}
        # Add other affected states as needed
    ],
    "symptoms": [
        {"name": "Stomach Cramps"},
        {"name": "Diarrhea"},
        {"name": "Fever"},
        {"name": "Nausea"},
        {"name": "Vomiting"},
        {"name": "Hemolytic Uremic Syndrome", "severity": "Severe"}
    ],
    "outbreak_report": {
        "name": "E. coli O157:H7 Outbreak",
        "start_date": "2024-09-27",
        "end_date": "2024-10-10",
        "cases_reported": 75,
        "hospitalizations": 22,
        "deaths": 1,
        "location": "Colorado"
    }
}

def generate_embedding(text):
    # Generate embeddings for the given text
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def connect_to_astra():
    # Connect to AstraDB
    cloud_config = {'secure_connect_bundle': secure_connect_bundle_path}
    auth_provider = PlainTextAuthProvider("token", application_token)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect('foodtech')
    return session

# Create tables for storing data
def create_tables(session):
    session.execute("""
    CREATE TABLE IF NOT EXISTS suppliers (
        name TEXT PRIMARY KEY,
        product TEXT,
        recall_status TEXT,
        recall_date DATE
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS products (
        name TEXT PRIMARY KEY,
        type TEXT,
        recall_status TEXT,
        embedding LIST<FLOAT>
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        name TEXT,
        location TEXT,
        PRIMARY KEY (name, location)
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS symptoms (
        name TEXT PRIMARY KEY,
        severity TEXT,
        embedding LIST<FLOAT>
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS outbreak_report (
        name TEXT PRIMARY KEY,
        start_date DATE,
        end_date DATE,
        cases_reported INT,
        hospitalizations INT,
        deaths INT,
        location TEXT
    );
    """)
    print("Tables created.")

    def insert_data(session, data):
        # Insert suppliers
        for supplier in data["suppliers"]:
            session.execute("""
            INSERT INTO suppliers (name, product, recall_status, recall_date)
            VALUES (%s, %s, %s, %s)
            """, (supplier["name"], supplier["product"], supplier["recall_status"], supplier["recall_date"]))

        # Insert products with embeddings
        for product in data["products"]:
            embedding = generate_embedding(product["name"])
            session.execute("""
            INSERT INTO products (name, type, recall_status, embedding)
            VALUES (%s, %s, %s, %s)
            """, (product["name"], product["type"], product.get("recall_status", "None"), embedding.tolist()))

        # Insert stores
        for store in data["stores"]:
            session.execute("""
            INSERT INTO stores (name, location)
            VALUES (%s, %s)
            """, (store["name"], store["location"]))

        # Insert symptoms with embeddings
        for symptom in data["symptoms"]:
            embedding = generate_embedding(symptom["name"])
            session.execute("""
            INSERT INTO symptoms (name, severity, embedding)
            VALUES (%s, %s, %s)
            """, (symptom["name"], symptom.get("severity", "None"), embedding.tolist()))

    # Insert outbreak report
    outbreak = data["outbreak_report"]
    session.execute("""
    INSERT INTO outbreak_report (name, start_date, end_date, cases_reported, hospitalizations, deaths, location)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (outbreak["name"], outbreak["start_date"], outbreak["end_date"], outbreak["cases_reported"],
          outbreak["hospitalizations"], outbreak["deaths"], outbreak["location"]))

    print("Data inserted.")


    from sklearn.metrics.pairwise import cosine_similarity

def find_similar_products(session, query_text):
    query_embedding = generate_embedding(query_text)
    rows = session.execute("SELECT name, embedding FROM products")
    
    products = []
    similarities = []

    for row in rows:
        product_embedding = np.array(row.embedding)
        similarity = cosine_similarity([query_embedding], [product_embedding])[0][0] # type: ignore
        products.append(row.name)
        similarities.append(similarity)

    # Sort by similarity
    similar_products = sorted(zip(products, similarities), key=lambda x: x[1], reverse=True)
    return similar_products[:3]  # Return top 3 matches

# Example search for products related to "Quarter Pounder"
session = connect_to_astra()
create_tables(session)
insert_data(session, data)
import os
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import openai
import numpy as np

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

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

keyspace = 'foodtech'
# Use the keyspace
session.set_keyspace(keyspace)


# Sample Data
data = {
    "suppliers": [
        {"name": "Taylor Farms", "product": "Slivered Onions", "recall_status": "Voluntary Recall", "recall_date": "2024-10-10"}
    ],
    "products": [
        {"name": "Quarter Pounder", "type": "Burger"},
        {"name": "Slivered Onions", "type": "Ingredient", "recall_status": "Recalled"},
        {"name": "Beef Patties", "type": "Ingredient", "status": "In Use"},
        {"name": "Diced Onions", "type": "Ingredient", "status": "Not Implicated"}
    ],
    "stores": [
        {"name": "McDonald's", "location": "Colorado"},
        {"name": "McDonald's", "location": "Kansas"},
        {"name": "McDonald's", "location": "Utah"}
        # Add other affected states as needed
    ],
    "symptoms": [
        {"name": "Stomach Cramps"},
        {"name": "Diarrhea"},
        {"name": "Fever"},
        {"name": "Nausea"},
        {"name": "Vomiting"},
        {"name": "Hemolytic Uremic Syndrome", "severity": "Severe"}
    ],
    "outbreak_report": {
        "name": "E. coli O157:H7 Outbreak",
        "start_date": "2024-09-27",
        "end_date": "2024-10-10",
        "cases_reported": 75,
        "hospitalizations": 22,
        "deaths": 1,
        "location": "Colorado"
    }
}

def generate_embedding(text):
    # Generate embeddings for the given text
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def connect_to_astra():
    # Connect to AstraDB
    cloud_config = {'secure_connect_bundle': secure_connect_bundle_path}
    auth_provider = PlainTextAuthProvider("token", application_token)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect('foodtech')
    return session

# Create tables for storing data
def create_tables(session):
    session.execute("""
    CREATE TABLE IF NOT EXISTS suppliers (
        name TEXT PRIMARY KEY,
        product TEXT,
        recall_status TEXT,
        recall_date DATE
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS products (
        name TEXT PRIMARY KEY,
        type TEXT,
        recall_status TEXT,
        embedding LIST<FLOAT>
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        name TEXT,
        location TEXT,
        PRIMARY KEY (name, location)
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS symptoms (
        name TEXT PRIMARY KEY,
        severity TEXT,
        embedding LIST<FLOAT>
    );
    """)
    session.execute("""
    CREATE TABLE IF NOT EXISTS outbreak_report (
        name TEXT PRIMARY KEY,
        start_date DATE,
        end_date DATE,
        cases_reported INT,
        hospitalizations INT,
        deaths INT,
        location TEXT
    );
    """)
    print("Tables created.")

def insert_data(session, data):
    # Insert suppliers
    for supplier in data["suppliers"]:
        session.execute("""
        INSERT INTO suppliers (name, product, recall_status, recall_date)
        VALUES (%s, %s, %s, %s)
        """, (supplier["name"], supplier["product"], supplier["recall_status"], supplier["recall_date"]))

    # Insert products with embeddings
    for product in data["products"]:
        embedding = generate_embedding(product["name"])
        session.execute("""
        INSERT INTO products (name, type, recall_status, embedding)
        VALUES (%s, %s, %s, %s)
        """, (product["name"], product["type"], product.get("recall_status", "None"), embedding.tolist()))

    # Insert stores
    for store in data["stores"]:
        session.execute("""
        INSERT INTO stores (name, location)
        VALUES (%s, %s)
        """, (store["name"], store["location"]))

    # Insert symptoms with embeddings
    for symptom in data["symptoms"]:
        embedding = generate_embedding(symptom["name"])
        session.execute("""
        INSERT INTO symptoms (name, severity, embedding)
        VALUES (%s, %s, %s)
        """, (symptom["name"], symptom.get("severity", "None"), embedding.tolist()))

    # Insert outbreak report
    outbreak = data["outbreak_report"]
    session.execute("""
    INSERT INTO outbreak_report (name, start_date, end_date, cases_reported, hospitalizations, deaths, location)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (outbreak["name"], outbreak["start_date"], outbreak["end_date"], outbreak["cases_reported"],
          outbreak["hospitalizations"], outbreak["deaths"], outbreak["location"]))

    print("Data inserted.")

from sklearn.metrics.pairwise import cosine_similarity

def find_similar_products(session, query_text):
    query_embedding = generate_embedding(query_text)
    rows = session.execute("SELECT name, embedding FROM products")
    
    products = []
    similarities = []

    for row in rows:
        product_embedding = np.array(row.embedding)
        similarity = cosine_similarity([query_embedding], [product_embedding])[0][0] # type: ignore
        products.append(row.name)
        similarities.append(similarity)

    # Sort by similarity
    similar_products = sorted(zip(products, similarities), key=lambda x: x[1], reverse=True)
    return similar_products[:3]  # Return top 3 matches

# Example search for products related to "Quarter Pounder"
session = connect_to_astra()
create_tables(session)
insert_data(session, data)
similar_products = find_similar_products(session, "Quarter Pounder")
print("Similar Products:", similar_products)
session.shutdown()
similar_products = find_similar_products(session, "Quarter Pounder")
print("Similar Products:", similar_products)
session.shutdown()
