from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Fetch the data
data = datasets.fetch_california_housing()

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

# Instantiate the model
model = Ridge(alpha=10.0)

# Train the model
model.fit(X_train_poly, y_train)

# Make predictions
predictions = model.predict(X_test_poly)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', round(new_r2_score, 3))
print('Insight: Using Ridge regression with alpha=10.0 after scaling the features using StandardScaler and adding polynomial features of degree 2 causes the r2_score to go from 0.646 to', round(new_r2_score, 3))