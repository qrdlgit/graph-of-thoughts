from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Fetch the data
data = datasets.fetch_california_housing()

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and Ridge regression with alpha=0.8
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=0.8))
])

# Train the model
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', round(new_r2_score, 3))
print('Insight: Replacing LinearRegression with Ridge regression and using alpha=0.8 in a Pipeline in base_n0_n1.py, as opposed to using LinearRegression with StandardScaler in base_n0.py, causes the r2_score to go from 0.576 to', round(new_r2_score, 3))