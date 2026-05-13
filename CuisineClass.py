import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create sample dataset
data = {
    "Restaurant": [
        "Cafe Coffee Day",
        "Dominos",
        "Pizza Hut",
        "Barbeque Nation",
        "Tea Villa",
        "Burger King",
        "Subway",
        "Starbucks"
    ],

    "Price": [500, 700, 900, 1500, 600, 400, 450, 800],

    "Rating": [4.1, 4.3, 4.2, 4.7, 4.0, 3.9, 4.1, 4.5],

    "Cuisine": [
        "Cafe",
        "Pizza",
        "Pizza",
        "BBQ",
        "Cafe",
        "Burger",
        "Fast Food",
        "Cafe"
    ]
}

# Convert into dataframe
df = pd.DataFrame(data)

# Encode cuisine labels
le = LabelEncoder()
df["Cuisine_Label"] = le.fit_transform(df["Cuisine"])

# Features and target
X = df[["Price", "Rating"]]
y = df["Cuisine_Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))