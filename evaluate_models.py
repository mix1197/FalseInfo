import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
print("Loading dataset...")
df = pd.read_csv(r'C:\Users\Dominic\OneDrive\Desktop\Manuscript\Philippine Fake News Corpus.csv')
print(f"Dataset loaded with {len(df)} rows")

# Print unique values in Label column
print("\nUnique values in Label column:")
print(df['Label'].unique())
print("\nLabel distribution before cleaning:")
print(df['Label'].value_counts())

# Data cleaning
print("\nCleaning data...")
# Remove rows with missing values
df = df.dropna(subset=['Headline', 'Content'])
print(f"Dataset after cleaning: {len(df)} rows")

# Combine Headline and Content for feature extraction
print("Preprocessing data...")
df['text'] = df['Headline'].fillna('') + ' ' + df['Content'].fillna('')

# Convert labels to numeric
# First, let's see what values we actually have in the Label column
print("\nUnique label values:")
print(df['Label'].unique())

# Map the labels correctly based on your actual data
label_map = {'Credible': 0, 'Not Credible': 1}  # Adjust these values based on your actual labels
df['label'] = df['Label'].map(label_map)

# Print label distribution after mapping
print("\nLabel distribution after mapping:")
print(df['label'].value_counts())

# Remove any rows where label mapping failed (resulted in NaN)
df = df.dropna(subset=['label'])
print(f"\nFinal dataset size: {len(df)} rows")

# Prepare the data
X = df['text']
y = df['label']

# Create TF-IDF vectors
print("\nCreating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=5000)
X_vectors = vectorizer.fit_transform(X)
print(f"Created vectors with {X_vectors.shape[1]} features")

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB()
}

# Results dictionary
results = {}

print("\nStarting model evaluation...")

# Evaluate each model
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # K-fold cross-validation
    print(f"Performing 10-fold cross-validation for {name}...")
    cv_scores = cross_val_score(model, X_vectors, y, cv=10)
    
    # Train model on full training set
    print(f"Training {name} on full training set...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print(f"Making predictions with {name}...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Plot confusion matrix
    print(f"Generating confusion matrix plot for {name}...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'static/images/{name.lower().replace(" ", "_")}_cm.png')
    plt.close()

    # Print current model results
    print(f"\nResults for {name}:")
    print(f"Cross-validation accuracy: {results[name]['cv_mean']:.4f} (+/- {results[name]['cv_std']*2:.4f})")
    print(f"Test accuracy: {results[name]['accuracy']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")
    print(f"F1 Score: {results[name]['f1']:.4f}")

# Save results
print("\nSaving results...")
joblib.dump(results, 'models/evaluation_results.pkl')
print("Evaluation complete! Results saved to models/evaluation_results.pkl")

# Print final comparison
print("\nFinal Model Comparison:")
print("\nCross-validation Accuracies:")
for name, metrics in results.items():
    print(f"{name}: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

print("\nTest Set Metrics:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")