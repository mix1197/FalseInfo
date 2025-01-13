import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load your dataset
df = pd.read_csv(r'C:\Users\Dominic\OneDrive\Desktop\Manuscript\Philippine Fake News Corpus.csv')

print("Dataset loaded successfully!")
print(f"Total rows in dataset: {len(df)}")
print("\nLabel value counts:")
print(df['Label'].value_counts())

# Combine Headline and Content
df['full_text'] = df['Headline'] + ' ' + df['Content']

# Convert labels with the correct values from your dataset
df['label_binary'] = df['Label'].map({'Not Credible': 1, 'Credible': 0})

print("\nAfter conversion, label counts:")
print(df['label_binary'].value_counts())

# Make sure we have both classes
if len(df['label_binary'].unique()) < 2:
    print("\nERROR: Dataset doesn't contain both classes!")
    exit()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['full_text'], 
    df['label_binary'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_binary']
)

print("\nData split complete...")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Create and fit the vectorizer
print("\nTraining vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
print("Training model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Calculate accuracy
X_test_vectorized = vectorizer.transform(X_test)
accuracy = model.score(X_test_vectorized, y_test)
print(f"\nModel accuracy: {accuracy:.2f}")

# Save the model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("Model and vectorizer saved successfully!")