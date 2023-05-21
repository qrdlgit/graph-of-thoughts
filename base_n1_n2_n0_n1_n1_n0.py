from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import pandas as pd

# Fetch the data
data = pd.read_pickle("data.pkl")

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures, and ExtraTreesRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', ExtraTreesRegressor(n_estimators=300, random_state=42))
])

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', new_r2_score)
print("Insight: Changing the model from GradientBoostingRegressor with n_estimators=300, learning_rate=0.1, and max_depth=3 in base_n1_n2_n0_n1_n1.py to ExtraTreesRegressor with n_estimators=300 in base_n1_n2_n0_n1_n1_n0.py causes the r2_score to go from 0.8172819769351605 to {:.3f}".format(new_r2_score))