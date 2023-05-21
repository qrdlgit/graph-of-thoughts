from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Fetch the data
data = pd.read_pickle("data.pkl")

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures, and GradientBoostingRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
])

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Define base models for StackingRegressor
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=300, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42))
]

# Create StackingRegressor with RidgeCV as the final estimator
stacking_model = StackingRegressor(estimators=base_models, final_estimator=RidgeCV(), n_jobs=-1)
stacking_model.fit(X_train_transformed, y_train)

# Make predictions
predictions = stacking_model.predict(X_test_transformed)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', new_r2_score)
print("Insight: Changing the model from GradientBoostingRegressor with GridSearchCV in base_n1_n2_n0_n1_n1_n1.py to a StackingRegressor with RandomForestRegressor and GradientBoostingRegressor as base models, and RidgeCV as the final estimator in base_n1_n2_n0_n1_n1_n1_n0.py causes the r2_score to go from 0.8331751744956455 to {:.3f}".format(new_r2_score))