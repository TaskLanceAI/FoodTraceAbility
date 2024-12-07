import warnings
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Initialize the embedding model with the desired Hugging Face model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# List of product descriptions
product_descriptions = [
    "Under colors of Benetton Men White Boxer Trunks",
    "Turtle Men Check Red Shirt",
    "United Colors of Benetton Men White Check Shirt",
    "United Colors of Benetton Men Check White Shirts",
    "Wrangler Men Broad Blue Shirt"
]

# Path to your Secure Connect Bundle
secure_connect_bundle_path = 'D:/TaskL/ecommerce/fashion/FashionRecommenderApp/astra-db-recommendations-starter/SCB/secure-connect-fashion.zip'

# Your application token
application_token = 'AstraCS:ZlyMyCobxWSafNzEERRFGkJt:b8190d55067628fcadb3e4224afbfb92327cb33590d8ddb955f0718d86b4d1bf'

# Setup authentication provider
auth_provider = PlainTextAuthProvider('token', application_token)

# Connect to the Cassandra database using the secure connect bundle
cluster = Cluster(
    cloud={"secure_connect_bundle": secure_connect_bundle_path},
    auth_provider=auth_provider
)
session = cluster.connect()

# Define keyspace and vector dimension
keyspace = "catalog"
v_dimension = 5

# Set the keyspace
session.set_keyspace(keyspace)

# Create the table if it doesn't exist
session.execute((
    "CREATE TABLE IF NOT EXISTS {keyspace}.ProductImageVectors (ProductId INT PRIMARY KEY, ProductDesc TEXT, ImageURL TEXT, ProductImageVector VECTOR<FLOAT,{v_dimension}>);"
).format(keyspace=keyspace, v_dimension=v_dimension))

# Create the index if it doesn't exist
session.execute((
    "CREATE CUSTOM INDEX IF NOT EXISTS idx_ProductImageVectors "
    "ON {keyspace}.ProductImageVectors "
    "(ProductImageVector) USING 'StorageAttachedIndex' WITH OPTIONS = "
    "{{'similarity_function' : 'cosine'}};"
).format(keyspace=keyspace))

# Iterate over each product description and insert the embeddings into AstraDB
for i, product_desc in enumerate(product_descriptions, start=1):
    # Get embeddings for the current product description
    embeddings = embed_model.get_text_embedding(product_desc)
    
    # Truncate or pad embeddings to match the vector dimension
    if len(embeddings) > v_dimension:
        embeddings = embeddings[:v_dimension]
    elif len(embeddings) < v_dimension:
        embeddings.extend([0.0] * (v_dimension - len(embeddings)))

    # Insert into AstraDB
    session.execute(
        f"INSERT INTO {keyspace}.ProductImageVectors(ProductId, ProductDesc, ImageURL, ProductImageVector) VALUES (%s, %s, %s, %s)",
        (i, product_desc, f"ProductImage_{i}.jpg", embeddings)
    )
    print(f"Inserted: {product_desc}")

print("All product descriptions have been inserted into AstraDB with embeddings.")
