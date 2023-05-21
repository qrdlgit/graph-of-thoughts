from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Fetch the data
data = datasets.fetch_california_housing()

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

# Train the model using the pipeline
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', round(new_r2_score, 3))
print('Insight: Using a pipeline for preprocessing and model training causes the r2_score to go from 0.646 to', round(new_r2_score, 3))