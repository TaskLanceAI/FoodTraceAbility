import openai
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

# Import the AstraDB configuration module
from astradbconn import connect_to_astra
from neo4jconn import connect_to_neo4j

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 1. Data Collection from IoT Sensors
iot_data = [
    {
        "batch_id": "batch_004",
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

# 5. Graph Database Construction using Neo4j
driver = connect_to_neo4j()

def create_batch_node(tx, batch_id, temperature, humidity, shipment_date):
    tx.run(
        """
        CREATE (b:Batch {batch_id: $batch_id, temperature: $temperature, humidity: $humidity, shipment_date: $shipment_date})
        """,
        batch_id=batch_id, temperature=temperature, humidity=humidity, shipment_date=shipment_date
    )

def create_similarity_relationship(tx, batch_id_1, batch_id_2, similarity_score):
    tx.run(
        """
        MATCH (b1:Batch {batch_id: $batch_id_1}), (b2:Batch {batch_id: $batch_id_2})
        CREATE (b1)-[:SIMILAR {score: $similarity_score}]->(b2)
        """,
        batch_id_1=batch_id_1, batch_id_2=batch_id_2, similarity_score=similarity_score
    )

# Create nodes for each batch in Neo4j
with driver.session() as session_neo4j:
    for entry in iot_data:
        session_neo4j.write_transaction(
            create_batch_node,
            entry['batch_id'],
            entry['temperature'],
            entry['humidity'],
            entry['shipment_date']
        )

    # Create similarity relationships based on the similarity search results
    for batch_id, similarity_score in similar_batches:
        session_neo4j.write_transaction(
            create_similarity_relationship,
            'batch_004',  # Example base batch for relationships
            batch_id,
            similarity_score
        )

print("Graph Database Updated with Nodes and Relationships.")

# Close connections
driver.close()
session.shutdown()
