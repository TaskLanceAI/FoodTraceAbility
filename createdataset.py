import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define parameters for the synthetic dataset
num_stores = 1000  # Total number of stores

# Define the fast food chains and their specific products with ingredients
fast_food_chains = {
    "McDonald's": {
        "Quarter Pounder": [
            {"ingredient": "Beef Patty", "quantity": "4 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Tomato", "quantity": "1 slice"},
            {"ingredient": "Pickles", "quantity": "3 slices"},
            {"ingredient": "Onions", "quantity": "0.25 oz"},
            {"ingredient": "Ketchup", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ],
        "Big Mac": [
            {"ingredient": "Beef Patty", "quantity": "2 x 1.6 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "3-part bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Special Sauce", "quantity": "0.5 oz"},
            {"ingredient": "Pickles", "quantity": "2 slices"},
            {"ingredient": "Onions", "quantity": "0.2 oz"}
        ]
    },
    "Burger King": {
        "Whopper": [
            {"ingredient": "Beef Patty", "quantity": "4 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Tomato", "quantity": "2 slices"},
            {"ingredient": "Pickles", "quantity": "4 slices"},
            {"ingredient": "Onions", "quantity": "0.25 oz"},
            {"ingredient": "Ketchup", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ],
        "Chicken Sandwich": [
            {"ingredient": "Chicken Patty", "quantity": "4 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ]
    },
    "Taco Bell": {
        "Crunchwrap Supreme": [
            {"ingredient": "Ground Beef", "quantity": "2 oz"},
            {"ingredient": "Tostada Shell", "quantity": "1 shell"},
            {"ingredient": "Tortilla", "quantity": "1 large"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Tomato", "quantity": "0.5 oz"},
            {"ingredient": "Cheese", "quantity": "0.5 oz"},
            {"ingredient": "Sour Cream", "quantity": "0.5 oz"}
        ],
        "Burrito": [
            {"ingredient": "Ground Beef", "quantity": "2 oz"},
            {"ingredient": "Tortilla", "quantity": "1 large"},
            {"ingredient": "Rice", "quantity": "1 oz"},
            {"ingredient": "Beans", "quantity": "1 oz"},
            {"ingredient": "Cheese", "quantity": "0.5 oz"},
            {"ingredient": "Sour Cream", "quantity": "0.5 oz"}
        ]
    },
    "KFC": {
        "Fried Chicken": [
            {"ingredient": "Chicken Breast", "quantity": "1 piece"},
            {"ingredient": "Bread Crumbs", "quantity": "0.5 oz"},
            {"ingredient": "Seasoning", "quantity": "0.2 oz"},
            {"ingredient": "Oil", "quantity": "0.5 oz"}
        ],
        "Zinger Burger": [
            {"ingredient": "Chicken Fillet", "quantity": "1 fillet"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ]
    }
}

# Generate lists of chains, products, and ingredients
chains = []
products = []
ingredients = []
for _ in range(num_stores):
    chain = random.choice(list(fast_food_chains.keys()))
    product = random.choice(list(fast_food_chains[chain].keys()))
    chains.append(chain)
    products.append(product)
    ingredients.append(json.dumps(fast_food_chains[chain][product]))  # Convert ingredients list to JSON format

# Define affected states for recall and contamination
affected_states = ['CO', 'KS', 'WY', 'IA', 'ID', 'MO', 'MT', 'NE', 'NM', 'NV', 'OK', 'UT']
states = ['CO', 'KS', 'WY', 'IA', 'ID', 'MO', 'MT', 'NE', 'NM', 'NV', 'OK', 'UT', 'CA', 'TX', 'NY', 'FL']

# Generate random data for each feature
store_ids = [f"Store_{i+1}" for i in range(num_stores)]
locations = [random.choice(states) for _ in range(num_stores)]

# Types of onions used (slivered = contaminated, diced = not contaminated)
onion_types = ['slivered', 'diced']
onion_products = [random.choice(onion_types) for _ in range(num_stores)]

# Randomly assign symptoms reported based on likelihood (0 if no symptoms, 1 if symptoms)
symptoms_reported = [random.choices([0, 1], weights=[0.8, 0.2])[0] for _ in range(num_stores)]

# Generate random supply dates for each store (recent dates)
start_date = datetime(2024, 9, 1)
supply_dates = [start_date + timedelta(days=random.randint(0, 60)) for _ in range(num_stores)]

# Affected status (1 for affected, 0 for unaffected)
# Stores in certain states with slivered onions have a higher chance of being affected
affected = [
    1 if (locations[i] in affected_states and onion_products[i] == 'slivered' and symptoms_reported[i] == 1) else 0
    for i in range(num_stores)
]

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define parameters for the synthetic dataset
num_stores = 1000  # Total number of stores

# Define the fast food chains and their specific products with ingredients
fast_food_chains = {
    "McDonald's": {
        "Quarter Pounder": [
            {"ingredient": "Beef Patty", "quantity": "4 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Tomato", "quantity": "1 slice"},
            {"ingredient": "Pickles", "quantity": "3 slices"},
            {"ingredient": "Onions", "quantity": "0.25 oz"},
            {"ingredient": "Ketchup", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ],
        "Big Mac": [
            {"ingredient": "Beef Patty", "quantity": "2 x 1.6 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "3-part bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Special Sauce", "quantity": "0.5 oz"},
            {"ingredient": "Pickles", "quantity": "2 slices"},
            {"ingredient": "Onions", "quantity": "0.2 oz"}
        ]
    },
    "Burger King": {
        "Whopper": [
            {"ingredient": "Beef Patty", "quantity": "4 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Tomato", "quantity": "2 slices"},
            {"ingredient": "Pickles", "quantity": "4 slices"},
            {"ingredient": "Onions", "quantity": "0.25 oz"},
            {"ingredient": "Ketchup", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ],
        "Chicken Sandwich": [
            {"ingredient": "Chicken Patty", "quantity": "4 oz"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ]
    },
    "Taco Bell": {
        "Crunchwrap Supreme": [
            {"ingredient": "Ground Beef", "quantity": "2 oz"},
            {"ingredient": "Tostada Shell", "quantity": "1 shell"},
            {"ingredient": "Tortilla", "quantity": "1 large"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Tomato", "quantity": "0.5 oz"},
            {"ingredient": "Cheese", "quantity": "0.5 oz"},
            {"ingredient": "Sour Cream", "quantity": "0.5 oz"}
        ],
        "Burrito": [
            {"ingredient": "Ground Beef", "quantity": "2 oz"},
            {"ingredient": "Tortilla", "quantity": "1 large"},
            {"ingredient": "Rice", "quantity": "1 oz"},
            {"ingredient": "Beans", "quantity": "1 oz"},
            {"ingredient": "Cheese", "quantity": "0.5 oz"},
            {"ingredient": "Sour Cream", "quantity": "0.5 oz"}
        ]
    },
    "KFC": {
        "Fried Chicken": [
            {"ingredient": "Chicken Breast", "quantity": "1 piece"},
            {"ingredient": "Bread Crumbs", "quantity": "0.5 oz"},
            {"ingredient": "Seasoning", "quantity": "0.2 oz"},
            {"ingredient": "Oil", "quantity": "0.5 oz"}
        ],
        "Zinger Burger": [
            {"ingredient": "Chicken Fillet", "quantity": "1 fillet"},
            {"ingredient": "Sesame Seed Bun", "quantity": "1 bun"},
            {"ingredient": "Lettuce", "quantity": "0.5 oz"},
            {"ingredient": "Mayonnaise", "quantity": "0.5 oz"}
        ]
    }
}

# Generate lists of chains, products, and ingredients
chains = []
products = []
ingredients = []
for _ in range(num_stores):
    chain = random.choice(list(fast_food_chains.keys()))
    product = random.choice(list(fast_food_chains[chain].keys()))
    chains.append(chain)
    products.append(product)
    ingredients.append(json.dumps(fast_food_chains[chain][product]))  # Convert ingredients list to JSON format

# Define affected states for recall and contamination
affected_states = ['CO', 'KS', 'WY', 'IA', 'ID', 'MO', 'MT', 'NE', 'NM', 'NV', 'OK', 'UT']
states = ['CO', 'KS', 'WY', 'IA', 'ID', 'MO', 'MT', 'NE', 'NM', 'NV', 'OK', 'UT', 'CA', 'TX', 'NY', 'FL']

# Generate random data for each feature
store_ids = [f"Store_{i+1}" for i in range(num_stores)]
locations = [random.choice(states) for _ in range(num_stores)]

# Types of onions used (slivered = contaminated, diced = not contaminated)
onion_types = ['slivered', 'diced']
onion_products = [random.choice(onion_types) for _ in range(num_stores)]

# Randomly assign symptoms reported based on likelihood (0 if no symptoms, 1 if symptoms)
symptoms_reported = [random.choices([0, 1], weights=[0.8, 0.2])[0] for _ in range(num_stores)]

# Generate random supply dates for each store (recent dates)
start_date = datetime(2024, 9, 1)
supply_dates = [start_date + timedelta(days=random.randint(0, 60)) for _ in range(num_stores)]

# Affected status (1 for affected, 0 for unaffected)
# Stores in certain states with slivered onions have a higher chance of being affected
affected = [
    1 if (locations[i] in affected_states and onion_products[i] == 'slivered' and symptoms_reported[i] == 1) else 0
    for i in range(num_stores)
]

# Combine all data into a dictionary
data = {
    'Store_ID': store_ids,
    'State': locations,
    'Fast_Food_Chain': chains,
    'Product': products,
    'Ingredients': ingredients,
    'Onion_Type': onion_products,
    'Symptoms_Reported': symptoms_reported,
    'Supply_Date': supply_dates,
    'Affected': affected
}

# Create a DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
le_state = LabelEncoder()
df['State'] = le_state.fit_transform(df['State'])

le_chain = LabelEncoder()
df['Fast_Food_Chain'] = le_chain.fit_transform(df['Fast_Food_Chain'])

le_product = LabelEncoder()
df['Product'] = le_product.fit_transform(df['Product'])

le_onion = LabelEncoder()
df['Onion_Type'] = le_onion.fit_transform(df['Onion_Type'])

# Prepare features and target variable
X = df[['State', 'Fast_Food_Chain', 'Product', 'Onion_Type', 'Symptoms_Reported']]
y = df['Affected']

# Apply k-NN algorithm
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Predict the risk of being affected for each store
df['Risk_Prediction'] = knn.predict(X)

# Show the first few rows of the dataset with risk predictions
print(df.head())

# Save the dataset to a CSV file
df.to_csv('store_onion_contamination_data_with_ingredients_and_risk.csv', index=False)
print("Dataset saved to 'store_onion_contamination_data_with_ingredients_and_risk.csv'")

# Create a DataFrame
df = pd.DataFrame(data)

# Show the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('store_onion_contamination_data_with_ingredients.csv', index=False)
print("Dataset saved to 'store_onion_contamination_data_with_ingredients.csv'")
