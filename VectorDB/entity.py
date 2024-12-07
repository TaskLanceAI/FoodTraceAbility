from transformers import AutoTokenizer, AutoModel
import torch
import openai  # Assuming OpenAI API is still used for context extraction
import os

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Load the Hugging Face embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :].squeeze().tolist()  # Use the first token ([CLS]) embedding
    return embeddings

# Connect to AstraDB
secure_connect_bundle_path = os.getenv('ASTRA_DB_SECURE_CONNECT_practice')
application_token = os.getenv('ASTRA_DB_TOKEN_practice')

session = Cluster(
    cloud={"secure_connect_bundle": secure_connect_bundle_path},
    auth_provider=PlainTextAuthProvider("token", application_token),
).connect()

session.set_keyspace('foodtech')

# Extract entities and context using OpenAI (or any other LLM)



# Print the current working directory
print("Current Working Directory:", os.getcwd())


text_file_path = r"Foodborne.txt"  # Define the path
with open("Foodborne.txt", 'r') as file:
    outbreak_text = file.read()

# Example extraction function (replace with actual extraction logic)
extracted_suppliers = [("Taylor Farms", "Slivered Onions", "Recalled", "2024-10-22")]
extracted_products = [("Quarter Pounder", "Burger", "Recalled")]
extracted_symptoms = [("Nausea", "Moderate")]
extracted_outbreaks = [("Outbreak-2024", "2024-10-01", "2024-10-15", 50, 10, 1, "Kansas")]

# Insert into suppliers
for supplier, product, recall_status, recall_date in extracted_suppliers:
    session.execute("""
        INSERT INTO suppliers (name, product, recall_status, recall_date)
        VALUES (%s, %s, %s, %s)
    """, (supplier, product, recall_status, recall_date))

# Insert into products with embeddings
for product, type_, recall_status in extracted_products:
    embedding = get_embedding(product)
    session.execute("""
        INSERT INTO products (name, type, recall_status, embedding)
        VALUES (%s, %s, %s, %s)
    """, (product, type_, recall_status, embedding))

# Insert into symptoms with embeddings
for symptom, severity in extracted_symptoms:
    embedding = get_embedding(symptom)
    session.execute("""
        INSERT INTO symptoms (name, severity, embedding)
        VALUES (%s, %s, %s)
    """, (symptom, severity, embedding))

# Insert into outbreak_report
for name, start_date, end_date, cases_reported, hospitalizations, deaths, location in extracted_outbreaks:
    session.execute("""
        INSERT INTO outbreak_report (name, start_date, end_date, cases_reported, hospitalizations, deaths, location)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (name, start_date, end_date, cases_reported, hospitalizations, deaths, location))

print("Data inserted successfully with embeddings.")
