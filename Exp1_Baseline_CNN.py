import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the normalized dataset
df_normalized = pd.read_csv('/Users/qilaawg/Documents/cleaned_dataset_normalized.csv')

# Display dataset overview
print("Dataset Overview:")
print(df_normalized.info())

# Check for missing values
print("\nMissing Values:")
print(df_normalized.isnull().sum())

# Preprocess features
X = df_normalized.drop(columns=[ 'year','time_signature','loudness','acousticness'])  # Drop irrelevant columns
categorical_columns = X.select_dtypes(include=['object']).columns  # Identify categorical columns
if len(categorical_columns) > 0:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)  # Encode categorical features
X = X.astype('float32')  # Ensure all features are numeric

# Preprocess target
y = df_normalized['genre']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # Convert to categorical format for CNN

# Reshape features for CNN
X_reshaped = np.expand_dims(X.values, axis=2)  # Add a third dimension for CNN

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Define a baseline CNN model
baseline_model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),  # Single Conv1D layer
    MaxPooling1D(pool_size=2),  # Basic pooling layer
    Flatten(),  # Flatten the output
    Dense(64, activation='relu'),  # Single dense hidden layer
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer for multi-class classification
])

# Compile the baseline model
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the baseline model
baseline_history = baseline_model.fit(
    X_train, y_train,
    epochs=10,  # Fewer epochs for a quicker training
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the baseline model
baseline_y_pred = baseline_model.predict(X_test)
baseline_y_pred_classes = np.argmax(baseline_y_pred, axis=1)
baseline_y_test_classes = np.argmax(y_test, axis=1)

# Classification Metrics for Baseline Model
baseline_accuracy = accuracy_score(baseline_y_test_classes, baseline_y_pred_classes)
baseline_precision = precision_score(baseline_y_test_classes, baseline_y_pred_classes, average='weighted')
baseline_recall = recall_score(baseline_y_test_classes, baseline_y_pred_classes, average='weighted')
baseline_f1 = f1_score(baseline_y_test_classes, baseline_y_pred_classes, average='weighted')

print("\nBaseline CNN Model Results:")
print(f"Accuracy: {baseline_accuracy:.2%}")
print(f"Precision: {baseline_precision:.2%}")
print(f"Recall: {baseline_recall:.2%}")
print(f"F1-Score: {baseline_f1:.2%}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(baseline_y_test_classes, baseline_y_pred_classes, target_names=encoder.classes_))
