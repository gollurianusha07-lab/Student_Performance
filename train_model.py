import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed",
        "sleep_hours"
    ]
]

# Output
y = data["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Model trained successfully!")
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")