import ipaddress as ipa
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
df = pd.read_csv(Darknet.csv)

# Remove unwanted columns
df = df.drop([Flow ID, Timestamp, Label2], axis=1)

# Drop rows with any missing data
df = df.dropna()

# Convert IP addresses to integer representations
df['Src IP'] = df['Src IP'].apply(lambda x int(ipa.ip_address(x)))
df['Dst IP'] = df['Dst IP'].apply(lambda x int(ipa.ip_address(x)))

# Encode the target column
encoder = LabelEncoder()
df['Label1'] = encoder.fit_transform(df['Label1'])

# Save the cleaned data
df.to_csv(processed.csv, index=False)

# Identify and address any anomalous entries
anomalous_columns = []
for col in df.columns
    if df[col].dtype == np.float64 or df[col].dtype == np.int64
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val == np.inf or min_val == -np.inf
            anomalous_columns.append(col)

# Filter out rows with anomalous entries
if anomalous_columns
    print(Detected rows with anomalies. Removing...)
    df = df[~df[anomalous_columns].isin([np.inf, -np.inf]).any(axis=1)]

# Split data into features and target
features = df.drop(['Label1'], axis=1)
target = df['Label1']

# Normalize features using Min-Max scaling
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Save the normalized data
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
normalized_df['Label1'] = target
normalized_df.to_csv(scaled3.csv, index=False)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)

# Define the neural network model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(normalized_features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Configure the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=825, validation_data=(X_test, y_test))

# Evaluate the model's performance
loss, accuracy = model.evaluate(X_test, y_test)

# Display and store the evaluation results
print(f'Evaluation Loss {loss.4f}')
print(f'Evaluation Accuracy {accuracy.4f}')

with open('evaluation_results.txt', 'w') as file
    file.write(f'Evaluation Loss {loss.4f}n')
    file.write(f'Evaluation Accuracy {accuracy.4f}n')