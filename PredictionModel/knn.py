import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('onion_contamination_data.csv')

# Encode categorical features (e.g., state, onion type, product type)
label_encoder_state = LabelEncoder()
label_encoder_onion_type = LabelEncoder()
label_encoder_product = LabelEncoder()

df['State'] = label_encoder_state.fit_transform(df['State'])
df['Onion_Type'] = label_encoder_onion_type.fit_transform(df['Onion_Type'])
df['Product'] = label_encoder_product.fit_transform(df['Product'])



# Convert supply date to a numerical format for k-NN compatibility
df['Supply_Date'] = pd.to_datetime(df['Supply_Date']).apply(lambda x: x.toordinal())



# Define features and target variable
X = df[['State', 'Onion_Type', 'Product', 'Symptoms_Reported', 'Supply_Date']]
y = df['Affected']  # Assume 1 = affected, 0 = unaffected

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to normalize their range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the k-NN model
k = 5  # Set the number of neighbors, which can be tuned
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict the likelihood of impact for test data
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# For predicting the likelihood of impact for new stores:
# Sample new data point with the same structure as X
new_store = pd.DataFrame([[label_encoder_state.transform(['CO'])[0],   # Example state
                           label_encoder_onion_type.transform(['slivered'])[0],  # Onion type
                           label_encoder_product.transform(['Quarter Pounder'])[0],  # Product type
                           1,  # Symptoms reported
                           pd.to_datetime('2024-10-15').toordinal()]],  # Supply date in ordinal format
                         columns=['State', 'Onion_Type', 'Product', 'Symptoms_Reported', 'Supply_Date'])

# Scale the new store data to match training data
new_store_scaled = scaler.transform(new_store)

# Predict the likelihood of impact
impact_prediction = knn.predict(new_store_scaled)
print("Predicted Impact (1 = impacted, 0 = not impacted):", impact_prediction[0])
