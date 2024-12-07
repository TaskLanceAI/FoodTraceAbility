import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def connect_to_astra():
    # AstraDB Configuration
    secure_connect_bundle_path = os.getenv('ASTRA_DB_SECURE_CONNECT_practice')
    application_token = os.getenv('ASTRA_DB_TOKEN_practice')

    # Debug prints to check environment variables (optional)
    print(f"Secure Connect Bundle Path: {secure_connect_bundle_path}")
    print(f"Application Token: {application_token}")

    if not secure_connect_bundle_path or not application_token:
        raise ValueError("AstraDB configuration is missing. Please set the environment variables for secure connect bundle and application token.")

    # Connect to the Cassandra database using the secure connect bundle
    cloud_config = {
        'secure_connect_bundle': secure_connect_bundle_path
    }
    auth_provider = PlainTextAuthProvider("token", application_token)

    # Create the cluster and connect
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    keyspace = 'foodtech'

    # Optionally, set the keyspace if needed
    # session.set_keyspace('your_keyspace_name')
    session = cluster.connect(keyspace)

    return session

# Example usage
try:
    session = connect_to_astra()
    print("Successfully connected to AstraDB!")
    # You can now perform operations with the session
    # Example: session.execute("SELECT * FROM your_table")
    session.execute("SELECT * FROM suppliers")
except Exception as e:
    print(f"Failed to connect to AstraDB: {e}")
