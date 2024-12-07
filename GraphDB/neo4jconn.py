import os
from neo4j import GraphDatabase

class Neo4jConfig:

    def __init__(self, uri=None, user=None, password=None):
        # Use environment variables if parameters are not provided
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        # Check for configuration and raise an error if missing
        if not self.uri or not self.user or not self.password:
            raise ValueError("Neo4j configuration is missing. Please set the URI, user, and password.")

        # Establish the connection
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Attempt a simple query to check the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Successfully connected to Neo4j!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        # Close the connection to the database
        if self.driver:
            self.driver.close()

# Example usage
if __name__ == "__main__":
    # Set environment variables directly in the code (optional)
    # Replace with actual values if needed
    os.environ['NEO4J_URI'] = "neo4j+s://3faf53d2.databases.neo4j.io"
    os.environ['NEO4J_USER'] = "neo4j"
    os.environ['NEO4J_PASSWORD'] = "QwKxG9senUw0F6xjEgzlQYSGhBIn9ZN-qXcKYLxXago"

    # Create an instance of the configuration class
    neo4j_config = Neo4jConfig()

    # Optionally close the connection if needed
    neo4j_config.close()
