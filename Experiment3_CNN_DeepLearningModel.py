import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from bayes_opt import BayesianOptimization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the normalized dataset
df_normalized = pd.read_csv('/Users/qilaawg/Documents/cleaned_dataset_normalized.csv')

# Preprocess features
X = df_normalized.drop(columns=['year', 'time_signature', 'loudness', 'acousticness'])  # Drop irrelevant columns
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

# Define the function to optimize
def cnn_cross_val(learning_rate, num_filters, dense_units, batch_size):
    """
    Function to train and evaluate a CNN model using cross-validation.
    """
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # Use 3-fold cross-validation
    accuracies = []

    for train_idx, val_idx in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Build the CNN model
        model = Sequential([
            Conv1D(int(num_filters), kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(int(dense_units), activation='relu'),
            Dense(y_categorical.shape[1], activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(X_train_fold, y_train_fold,
                  batch_size=int(batch_size),
                  epochs=10,  # Fewer epochs for faster training
                  verbose=0)

        # Evaluate the model
        val_predictions = model.predict(X_val_fold).argmax(axis=1)
        val_actuals = y_val_fold.argmax(axis=1)
        accuracies.append(accuracy_score(val_actuals, val_predictions))

    # Return the mean accuracy across folds
    return np.mean(accuracies)

# Define the Bayesian Optimization objective function
def bayesian_objective(learning_rate, num_filters, dense_units, batch_size):
    """
    Wrapper function for Bayesian Optimization.
    """
    return cnn_cross_val(learning_rate, num_filters, dense_units, batch_size)

# Define the parameter bounds for Bayesian Optimization
param_bounds = {
    'learning_rate': (0.0001, 0.01),  # Learning rate range
    'num_filters': (32, 128),  # Number of filters in the Conv1D layer
    'dense_units': (32, 256),  # Number of units in the dense layer
    'batch_size': (16, 64),  # Batch size range
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=bayesian_objective,
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)

# Perform Bayesian Optimization
optimizer.maximize(init_points=5, n_iter=10)  # 5 initial random points, 10 optimization steps

# Best parameters found
best_params = optimizer.max['params']
print("\nBest Parameters Found:")
print(best_params)

# Train the final model with the best hyperparameters
final_model = Sequential([
    Conv1D(int(best_params['num_filters']), kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(int(best_params['dense_units']), activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the final model
final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Train the final model
final_model.fit(
    X_train, y_train,
    batch_size=int(best_params['batch_size']),
    epochs=50,  # Train for more epochs
    validation_split=0.2,
    verbose=1
)

# Evaluate the final model
final_y_pred = final_model.predict(X_test)
final_y_pred_classes = np.argmax(final_y_pred, axis=1)
final_y_test_classes = np.argmax(y_test, axis=1)
conf_matrix=confusion_matrix(final_y_test_classes, final_y_pred_classes)


# Classification Metrics for Final Model
final_accuracy = accuracy_score(final_y_test_classes, final_y_pred_classes)
final_precision = precision_score(final_y_test_classes, final_y_pred_classes, average='weighted')
final_recall = recall_score(final_y_test_classes, final_y_pred_classes, average='weighted')
final_f1 = f1_score(final_y_test_classes, final_y_pred_classes, average='weighted')

print("\nCNN Model Results:")
print(f"Accuracy: {final_accuracy:.2%}")
print(f"Precision: {final_precision:.2%}")
print(f"Recall: {final_recall:.2%}")
print(f"F1-Score: {final_f1:.2%}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(final_y_test_classes, final_y_pred_classes, target_names=encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(final_y_test_classes, final_y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("CNN Classifier Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
print("CNN Classifier Confusion Matrix\n",conf_matrix)