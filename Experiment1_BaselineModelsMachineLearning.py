import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf 
import keras
import numpy as np
from keras.models import Sequential                      
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the standardized dataset
df_standardized = pd.read_csv('/Users/qilaawg/Documents/cleaned_dataset_standardized.csv')

# Display basic dataset information
print("Dataset Overview:")
print(df_standardized.info())

print("\nSummary Statistics:")
print(df_standardized.describe())

# Check for missing values
print("\nMissing Values:")
print(df_standardized.isnull().sum())

# Drop irrelevant columns
df = df_standardized.drop(columns=['year','time_signature','loudness','acousticness'])

# Encode the target variable ('genre') into numerical values
label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre'])

# Define the features and target variable
X = df.drop(columns=['genre'])
y = df['genre']

# Feature selection using PCA
pca = PCA(n_components=10) 
X_selected = pca.fit_transform(X)

# Check explained variance ratio
print("Explained variance ratio by selected components:", pca.explained_variance_ratio_)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define a function to print accuracy, precision, recall, and F1-score for a model
def print_metrics(model_name, y_test, y_pred):
    print(f"\n{model_name} Performance Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print_metrics("SVM Classifier", y_test, y_pred_svm)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print_metrics("Random Forest Classifier", y_test, y_pred_rf)



