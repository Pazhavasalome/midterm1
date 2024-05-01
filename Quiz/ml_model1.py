import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read data
df = pd.read_csv(processed.csv)

# Encode labels if needed
label_encoder = LabelEncoder()
df['Label1'] = label_encoder.fit_transform(df['Label1'])

# Split data into features and labels
features = df.drop(['Label1'], axis=1)
label = df['Label1']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with reduced epochs and batch size
model.fit(X_train, y_train, epochs=20, batch_size=753, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss {loss.4f}')
print(f'Test Accuracy {accuracy.4f}')

# Make predictions (uncomment if needed)
# predictions = model.predict(X_test)