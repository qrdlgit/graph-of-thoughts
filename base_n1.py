from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

# Fetch the data
data = pd.read_pickle("data.pkl")

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

# Instantiate the model
model = Ridge(alpha=1.0)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', new_r2_score)
print("Insight: Changing the model from LinearRegression in base.py to Ridge with alpha=1.0, adding StandardScaler, and applying PolynomialFeatures with degree=2 in base_n1.py causes the r2_score to go from 0.575 to {:.3f}".format(new_r2_score))