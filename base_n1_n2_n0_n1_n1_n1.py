from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    ('model', GradientBoostingRegressor(random_state=42))
])

# Set up hyperparameter grid for tuning
param_grid = {
    'model__n_estimators': [300, 400],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 4],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Make predictions
predictions = grid_search.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', new_r2_score)
print("Insight: Changing the model from GradientBoostingRegressor with n_estimators=300, learning_rate=0.1, and max_depth=3 in base_n1_n2_n0_n1_n1.py to GradientBoostingRegressor with GridSearchCV for hyperparameter tuning in base_n1_n2_n0_n1_n1_n1.py causes the r2_score to go from 0.8172819769351605 to {:.3f}".format(new_r2_score))