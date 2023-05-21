from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd

# Fetch the data
data = pd.read_pickle("data.pkl")

# Split into features (X) and target (y)
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures, and RidgeCV
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0]))

# Train the model
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)

# Compute and display r^2 score
new_r2_score = r2_score(y_test, predictions)
print('r2_score:', new_r2_score)
print("Insight: Changing the model from Ridge with alpha=1.0 in base_n1.py to RidgeCV with automatic alpha selection in base_n1_n2.py and using a pipeline for preprocessing in base_n1_n2_n0.py, and increasing the degree of PolynomialFeatures to 3 in base_n1_n2_n1.py causes the r2_score to go from 0.656 to {:.3f}".format(new_r2_score))