import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the standardized dataset
df_standardized = pd.read_csv('/Users/qilaawg/Documents/cleaned_dataset_standardized.csv')

# Drop irrelevant columns
df = df_standardized.drop(columns=['year','time_signature','loudness','acousticness'])

# Define the target variable and features
X = df.drop(columns=['genre'])  # Features only
y = df['genre']  # Target variable

# Feature selection using PCA
pca = PCA(n_components=10)  # Select top 10 components
X_selected = pca.fit_transform(X)

# Check explained variance ratio
print("Explained variance ratio by selected components:", pca.explained_variance_ratio_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# Define a function to print accuracy, precision, recall, F1-score, and confusion matrix for a model
def evaluate_model(model_name, y_test, y_pred, labels):
    print(f"\n{model_name} Performance Metrics:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Initialize classifiers
svm_model = SVC()
rf_model = RandomForestClassifier()

# SVM model
svm_param_grid = {
    'C': [100],
    'gamma': [1],
    'kernel': ['rbf']
}

svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)

print("Best cross-validation score for SVM:", svm_grid_search.best_score_)
print("Best parameters for SVM:", svm_grid_search.best_params_)

# Train the best SVM model
best_svm_model = svm_grid_search.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)


# Evaluate SVM
evaluate_model("SVM Classifier", y_test, y_pred_svm, labels=y.unique())
conf_matrix=confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix for SVM Classifier\n",conf_matrix)

# Random Forest
rf_param_grid = {
    'n_estimators': [800],
    'max_depth': [None],
    'min_samples_leaf': [1]
}

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)

print("Best cross-validation score for Random Forest:", rf_grid_search.best_score_)
print("Best parameters for Random Forest:", rf_grid_search.best_params_)

# Train the best Random Forest model
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate Random Forest
evaluate_model("Random Forest Classifier", y_test, y_pred_rf, labels=y.unique())
conf_matrix=confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix for Random Forest Classifier\n",conf_matrix)