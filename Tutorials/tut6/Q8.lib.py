import numpy as np
from numpy.linalg import inv,matrix_rank,det
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Load the California housing dataset
data = load_breast_cancer(as_frame=True) # Load as a DataFrame

X = data.data
y = data.target

print(X.head())

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# Feature Selection using training data
# Combine into a single DataFrame
df = X_train.copy()
df['target'] = y_train

# Compute correlation with the target
correlations = df.corr()['target'].drop('target')
print("Correlations with target:")
print(correlations.abs().sort_values(ascending=False))

# Filter features with |correlation| > 0.5
filtered_features = correlations[correlations.abs()>0.5].index.tolist()
print("\nFiltered features with absolute correction > 0.5:")
print(filtered_features)

# Drop features highly correlated with each other (|corr| > 0.9)
selected_features = []
cor_matrix = df[filtered_features].corr().abs()
for feature in filtered_features:
    # Check if it's highly correlated with any already selected feature
    if all(cor_matrix.loc[feature, selected] <= 0.9 for selected in selected_features):
        # all(iterable): returns True if all elements of an iterable are True.
        #                all([]) returns True by definition
        selected_features.append(feature)
print("Final Selected features:")
print(selected_features)

# Subset the DataFrame to only include the selected features.
df_train = X_train[selected_features]
df_val = X_val[selected_features]
df_test = X_test[selected_features]

# Print the shapes of the datasets
print(f"Training set shape after feature selection: {df_train.shape}, {y_train.shape}")
print(f"Validation set shape after feature selection: {df_val.shape}, {y_val.shape}")
print(f"Test set shape after feature selection: {df_test.shape}, {y_test.shape}")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(df_train, y_train)
print("model coefficients:", model.coef_)
print("model intercept:", model.intercept_)
y_pred_train = model.predict(df_train)
train_acc = accuracy_score(y_train, y_pred_train)
y_pred_val = model.predict(df_val)
val_acc = accuracy_score(y_val, y_pred_val)
y_pred_test = model.predict(df_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Training accuracy using sklearn: {train_acc}")
print(f"Validation accuracy using sklearn: {val_acc}")
print(f"Test accuracy using sklearn: {test_acc}")