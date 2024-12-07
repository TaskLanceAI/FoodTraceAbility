import openai
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Import the AstraDB configuration module
from astradbconn import connect_to_astra

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 1. Data Collection from IoT Sensors
iot_data = [
    {
        "batch_id": "batch_001",
        "temperature": 12.5,
        "humidity": 65,
        "shipment_date": "2024-10-15",
        "ingredient_list": "Onions, Water, Salt"
    },
    {
        "batch_id": "batch_002",
        "temperature": 14.0,
        "humidity": 60,
        "shipment_date": "2024-10-16",
        "ingredient_list": "Onions, Garlic, Vinegar"
    },
    # Add more IoT data entries as needed
]

# 2. Vectorization: Convert text-based data to embeddings
def generate_embedding(text):
    # Use OpenAI's new API to generate embeddings for text-based data
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text]  # Updated to pass input as a list
    )
    # Extract the embedding
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

# Generate embeddings for ingredient lists
for entry in iot_data:
    entry['embedding'] = generate_embedding(entry['ingredient_list'])

# 3. Store Embeddings in AstraDB
session = connect_to_astra()

# Create the onion_batches table if it doesn't exist
def create_onion_batches_table(session):
    # Create the table if it does not exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS onion_batches (
        batch_id TEXT PRIMARY KEY,
        embedding LIST<FLOAT>
    )
    """
    session.execute(create_table_query)
    print("Table 'onion_batches' is ready.")

# Create the table
create_onion_batches_table(session)

def insert_embedding(session, batch_id, embedding):
    # Convert embedding to a list for storage
    embedding_list = embedding.tolist()
    session.execute(
        """
        INSERT INTO onion_batches (batch_id, embedding)
        VALUES (%s, %s)
        """,
        (batch_id, embedding_list)
    )

# Insert each IoT data entry into AstraDB
for entry in iot_data:
    insert_embedding(session, entry['batch_id'], entry['embedding'])

# 4. Similarity Search in AstraDB
def find_similar_batches(session, query_embedding, top_n=3):
    rows = session.execute("SELECT batch_id, embedding FROM onion_batches")
    batch_ids = []
    embeddings = []

    for row in rows:
        batch_ids.append(row.batch_id)
        embeddings.append(np.array(row.embedding))

    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    # Sort by similarity in descending order
    similar_indices = np.argsort(similarities)[::-1][:top_n]

    # Return the top N most similar batches
    return [(batch_ids[i], similarities[i]) for i in similar_indices]

# Example query to find similar batches
query_embedding = generate_embedding("Onions, Water")
similar_batches = find_similar_batches(session, query_embedding)
print("Similar Batches:", similar_batches)

# Clean up and close the connections
session.shutdown()
