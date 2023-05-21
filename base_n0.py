from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pandas as pd

# Fetch the data
data = pd.read_pickle("data.pkl")

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', round(new_r2_score, 3))
print("Insight: Changing the model from LinearRegression in base.py to Ridge with alpha=1.0 and adding StandardScaler in base_n0.py causes the r2_score to go from 0.575 to", round(new_r2_score, 3))