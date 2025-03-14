import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load your breast cancer dataset
CancerData = pd.read_csv("C:/Users/marya/Documents/PythonProjects/DataSets/data.csv")
# Convert any catergorical data into numerical dataa:
CancerData.loc[CancerData['diagnosis'] == 'M', 'diagnosis'] = 1
CancerData.loc[CancerData['diagnosis'] == 'B', 'diagnosis'] = 0


# Check for any null values
print(f'null columns: {CancerData.isnull().sum()}')

# Separate features and target variable
cancer_X = CancerData[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                       'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
                       'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
                       'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                       'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                       'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                       'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']]
cancer_y = CancerData['diagnosis']
print(cancer_y)

# Scale features using scale() method
X_scaled = scale(cancer_X)


# Split the data into training and testing sets
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(
    X_scaled, cancer_y, test_size=0.2, random_state=4
)

# Initialize the linear regression model
model = LinearRegression()
print(cancer_X_train.shape)

# Train the model using the training sets
model.fit(cancer_X_train, cancer_y_train)

# Make predictions using the test set
cancer_y_pred = model.predict(cancer_X_test)

# Converting regression predictions into binary labels using a threshold
#threshold = 0.5
threshold = np.median(cancer_y_pred)
print(f'Threshold: {threshold}')
binary_preds = [1 if pred > threshold else 0 for pred in cancer_y_pred]

# Checking for null or NaN values in the target variable (cancer_y_test)
null_indices = pd.isnull(cancer_y_test)

# Filtering out null or NaN values from cancer_y_test and binary_preds using zip
cancer_y_test = [val for idx, val in zip(null_indices, cancer_y_test) if not idx]
binary_preds = [pred for idx, pred in zip(null_indices, binary_preds) if not idx]

# Plotting
plt.figure(figsize=(8, 6))

# Scatter plot of actual vs binary predicted values
plt.scatter(cancer_y_test, binary_preds, color='black', label='Actual vs Predicted')

# Plotting the regression line: Not logical
# plt.plot(cancer_y_test, cancer_y_test, color='blue', linewidth=3, label='Regression Line')

plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted')
#plt.legend()
plt.show()

# model coefficients and intercept
print(f'Coefficient: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# Perform cross-validation with k=3
cv_scores = cross_val_score(model, X_scaled, cancer_y, cv=3, scoring='r2')

# Print cross-validation scores
print("Cross-validation R-squared scores:", cv_scores)

# Calculating classification metrics using the cleaned cancer_y_test and binary_preds
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score

accuracy = accuracy_score(cancer_y_test, binary_preds)
precision = precision_score(cancer_y_test, binary_preds)
recall = recall_score(cancer_y_test, binary_preds)
roc_auc = roc_auc_score(cancer_y_test, binary_preds)
conf_matrix = confusion_matrix(cancer_y_test, binary_preds)
f1 = f1_score(cancer_y_test, binary_preds)

# Print the classification metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'ROC-AUC: {roc_auc}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'F1-score: {f1}')
