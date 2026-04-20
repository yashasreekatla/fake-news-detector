import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (tiny but enough)
data = {
    "text": [
        "Government launches new education policy",
        "Aliens landed in Mumbai yesterday",
        "Stock market hits record high",
        "Man turns invisible after drinking potion"
    ],
    "label": [1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = LogisticRegression()
model.fit(X, df["label"])

# Test input
user_input = input("Enter news: ")
input_data = vectorizer.transform([user_input])

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")