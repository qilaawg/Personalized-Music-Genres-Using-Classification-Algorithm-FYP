import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('/Users/qilaawg/spotify data/spotify_data_top_20_genres.csv')

# Display basic dataset information
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handling Missing Value
df = df.dropna()

# Check for missing values after handling
missing_values = df.isnull().sum()
print(missing_values)

### Preprocessing ###

# Outlier removal using IQR method for specific numerical columns
numerical_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 
                      'tempo', 'duration_ms', 'time_signature']

for feature in numerical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# Check the shape of the dataset after outlier removal
print("Shape of dataset after outlier removal:", df.shape)

# Separate features and target
X = df.drop(columns=['genre','artist_name', 'track_name', 'track_id','row_id', 'popularity'])  
y = df['genre']

# Encode the target column if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Combine resampled data into a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['genre'] = label_encoder.inverse_transform(y_resampled)  # Decode the labels back

# Verify the balance
print("Class distribution after SMOTE:")
print(df_resampled['genre'].value_counts())

### Normalization and Standardization ###

# Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
df_normalized = df_resampled.copy()
df_normalized[numerical_features] = scaler.fit_transform(df_normalized[numerical_features])

# Standardization (Standard Scaling)
scaler = StandardScaler()
df_standardized = df_resampled.copy()
df_standardized[numerical_features] = scaler.fit_transform(df_standardized[numerical_features])

# Save cleaned datasets
df_normalized.to_csv('/Users/qilaawg/Documents/cleaned_dataset_normalized.csv', index=False)
df_standardized.to_csv('/Users/qilaawg/Documents/cleaned_dataset_standardized.csv', index=False)

print("Datasets saved successfully!")
### Details of Normalized Dataset ###

# Distribution of genres
plt.figure(figsize=(10, 6))
sns.countplot(y='genre', data=df_normalized, order=df_normalized['genre'].value_counts().index)
plt.title("Distribution of Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

# Yearly trends in the dataset
plt.figure(figsize=(12, 6))
sns.histplot(df_normalized['year'], bins=20, kde=False, color='purple')
plt.title("Yearly Distribution of Songs")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# Correlation heatmap for numerical features
numerical_features = df_normalized.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df_normalized[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot of key features with genre
selected_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
sns.pairplot(df_normalized, vars=selected_features, hue='genre', palette='Set2', height=2.5)
plt.show()

# Boxplot for 'popularity' across genres
plt.figure(figsize=(15, 6))
sns.boxplot(x='genre', y='popularity', data=df_normalized)
plt.title("Popularity Distribution Across Genres")
plt.xlabel("Genre")
plt.ylabel("Popularity")
plt.xticks(rotation=45)
plt.show()

# Top artists in the dataset
top_artists = df_normalized['artist_name'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_artists.index, x=top_artists.values, palette='viridis')
plt.title("Top 10 Artists by Song Count")
plt.xlabel("Number of Songs")
plt.ylabel("Artist")
plt.show()

# Tempo distribution across genres
plt.figure(figsize=(15, 6))
sns.boxplot(x='genre', y='tempo', data=df_normalized)
plt.title("Tempo Distribution Across Genres")
plt.xlabel("Genre")
plt.ylabel("Tempo")
plt.xticks(rotation=45)
plt.show()

### Details of Standardized Dataset ###

# Distribution of genres
plt.figure(figsize=(10, 6))
sns.countplot(y='genre', data=df_standardized, order=df_standardized['genre'].value_counts().index)
plt.title("Distribution of Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

# Yearly trends in the dataset
plt.figure(figsize=(12, 6))
sns.histplot(df_standardized['year'], bins=20, kde=False, color='purple')
plt.title("Yearly Distribution of Songs")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# Correlation heatmap for numerical features
numerical_features = df_standardized.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df_standardized[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot of key features with genre
selected_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
sns.pairplot(df_normalized, vars=selected_features, hue='genre', palette='Set2', height=2.5)
plt.show()

# Boxplot for 'popularity' across genres
plt.figure(figsize=(15, 6))
sns.boxplot(x='genre', y='popularity', data=df_normalized)
plt.title("Popularity Distribution Across Genres")
plt.xlabel("Genre")
plt.ylabel("Popularity")
plt.xticks(rotation=45)
plt.show()

# Top artists in the dataset
top_artists = df_standardized['artist_name'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_artists.index, x=top_artists.values, palette='viridis')
plt.title("Top 10 Artists by Song Count")
plt.xlabel("Number of Songs")
plt.ylabel("Artist")
plt.show()

# Tempo distribution across genres
plt.figure(figsize=(15, 6))
sns.boxplot(x='genre', y='tempo', data=df_standardized)
plt.title("Tempo Distribution Across Genres")
plt.xlabel("Genre")
plt.ylabel("Tempo")
plt.xticks(rotation=45)
plt.show()