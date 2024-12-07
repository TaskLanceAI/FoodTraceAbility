import pandas as pd
import random
from faker import Faker

# Initialize Faker for generating sample data
fake = Faker()

# Define the number of rows you want to generate
num_rows = 1000

# Define sample data for each column based on the description
brands = ['McDonald\'s', 'Burger King', 'Taco Bell']
products = ['Quarter Pounder', 'Beef Patties', 'French Fries']
suppliers = ['Taylor Farms', 'Sysco', 'US Foods']
manufacturers = ['McDonald\'s Corp', 'Burger King Corp', 'Yum! Brands']
packaged_by = ['ABC Packaging', 'XYZ Packers', 'PQR Packaging']

# Generate data for each row
data = []
for _ in range(num_rows):
    row = {
        'OrderID': fake.unique.random_int(min=1000, max=9999),
        'Brand': random.choice(brands),
        'Product': random.choice(products),
        'CustomerID': fake.random_int(min=10000, max=99999),  # Removed uniqueness
        'Supplier': random.choice(suppliers),
        'Manufactured By': random.choice(manufacturers),
        'Packaged By': random.choice(packaged_by),
        'Sale Price': round(random.uniform(1.99, 10.99), 2),
        'Quantity': random.randint(1, 10),
        'Store_ID': fake.random_int(min=100, max=999),  # Removed uniqueness
        'Street Number': fake.building_number(),
        'Street Name': fake.street_name(),
        'Manufacture Date': fake.date_between(start_date="-1y", end_date="today"),
        'Expiry Date': fake.date_between(start_date="today", end_date="+1y"),
        'Calories': random.randint(200, 800),
        '# of Onions': random.randint(0, 5),
        '# of Tomatoes': random.randint(0, 5),
        '# of Breads': random.randint(1, 2),
        'Sugar': round(random.uniform(0.5, 10.0), 2),
        'Salt': round(random.uniform(0.5, 10.0), 2),
        'Is Non-Veg': random.choice(['Yes', 'No']),
        'Latitude': round(random.uniform(24.396308, 49.384358), 6),  # Approx range for US latitudes
        'Longitude': round(random.uniform(-125.0, -66.93457), 6)    # Approx range for US longitudes
    }
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to an Excel file
df.to_excel("FoodborneInvestigations_1000_rows_with_coordinates.xlsx", index=False)
print("Excel file with 1000 rows created: 'FoodborneInvestigations_1000_rows_with_coordinates.xlsx'")
