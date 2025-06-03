import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Dataset
data = {
    'text': [
        'I love this product!',
        'This is the worst experience ever.',
        'Absolutely fantastic!',
        'Not good, very disappointing.',
        'Great value for money.',
        'I will never buy this again.'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
os.makedirs('model', exist_ok=True)
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training complete and model saved.")
