import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


# Load dataset (without `header=None`)
pima = pd.read_csv('diabetes.csv')

# Print actual column names to check for mismatches
print("Columns in dataset:", pima.columns.tolist())

# Remove whitespace from column names (if any)
pima.columns = pima.columns.str.strip()

# Update feature columns based on actual dataset column names
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = pima[feature_cols]
y = pima['Outcome']  # Update target column name

# Convert all data to numeric to avoid errors
X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
X = X.fillna(0)  # Fill NaN with 0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Instantiate model
logreg = LogisticRegression(random_state=16)

# Fit model
logreg.fit(X_train, y_train)

# Predict
y_pred = logreg.predict(X_test)

print("Model trained successfully!")

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cnf_matrix)

# Create heatmap
class_names = [0, 1]
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.xticks(np.arange(len(class_names)), class_names)
plt.yticks(np.arange(len(class_names)), class_names)
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Classification Report
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))

# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
