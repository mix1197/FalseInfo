import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # Using LinearSVC instead of SVC
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
print(f"Original dataset size: {len(df)} rows")

# Take a smaller sample for testing
SAMPLE_SIZE = 5000
df = df.sample(n=SAMPLE_SIZE, random_state=42)
print(f"Using sample size of {SAMPLE_SIZE} rows for testing")

# Print label distribution
print("\nLabel distribution:")
print(df['Label'].value_counts())

# Combine Headline and Content
print("\nPreprocessing data...")
df['text'] = df['Headline'].fillna('') + ' ' + df['Content'].fillna('')

# Convert labels
label_map = {'Credible': 0, 'Not Credible': 1}
df['label'] = df['Label'].map(label_map)

# Remove any rows where label mapping failed
df = df.dropna(subset=['label', 'text'])
print(f"Final dataset size: {len(df)} rows")

# Prepare the data
X = df['text']
y = df['label']

# Create TF-IDF vectors with fewer features
print("\nCreating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=1000)  # Reduced features
X_vectors = vectorizer.fit_transform(X)
print(f"Created vectors with {X_vectors.shape[1]} features")

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Initialize models with optimized parameters
models = {
    'SVM': LinearSVC(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB()
}

# Results dictionary
results = {}

print("\nStarting model evaluation...")

# Evaluate each model
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # K-fold cross-validation
    print(f"Performing 5-fold cross-validation for {name}...")  # Reduced from 10-fold
    cv_scores = cross_val_score(model, X_vectors, y, cv=5)
    
    # Train model
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print(f"Making predictions...")
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'static/images/{name.lower().replace(" ", "_")}_cm.png')
    plt.close()

    # Print results
    print(f"\nResults for {name}:")
    print(f"Cross-validation accuracy: {results[name]['cv_mean']:.4f} (+/- {results[name]['cv_std']*2:.4f})")
    print(f"Test accuracy: {results[name]['accuracy']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")
    print(f"F1 Score: {results[name]['f1']:.4f}")

# Save results
joblib.dump(results, 'models/test_results.pkl')
print("\nTest evaluation complete!")