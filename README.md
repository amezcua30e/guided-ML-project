# guided-ML-project
This was a guided ML project for my advanced data analytics class. Here is the description: For your final project, you have the option to work on one of three projects, each hosted on Kaggle. Below is a summary of each project along with relevant details:

Binary Classification with a Bank Churn Dataset: Predict whether customers will continue with their bank account or close it. Evaluation is based on the area under the ROC curve. For more details, visit the project page: Bank Churn Prediction.

Multi-Class Prediction of Obesity Risk: Determine the obesity risk levels in individuals, which are indicators of potential cardiovascular diseases. This project requires accuracy score evaluation, noting that the outcomes are not binary. Visit the project page for further information: Obesity Risk Prediction.

Binary Prediction of Smoker Status using Bio-Signals: Utilize health indicators to predict an individual's smoking status. Evaluation will be through the area under the ROC curve. For more details, see the project page: Smoker Status Prediction.

Important Notes:

Project Difficulty: Each project varies in complexity. Review each project carefully before making your choice. You must select a project by April 12, and changes are not permitted afterward.

Submission Requirements: Submit both your notebook and a CSV file in the format required by Kaggle. Ensure your notebook is error-free and that all code and text are organized clearly.

Model Restrictions: Submissions utilizing models not covered in our class will be rejected.

Grading Scheme: The top 30% of submissions for each project will receive 20 points, the middle 30% will receive 18 points, and the remaining will receive 16 points, based on the specific evaluation metric of each project. In the event of a tie, the submission with the more readable notebook will be favored.

Ensure you adhere to these guidelines and choose the project that best aligns with your skills and interests.
Here is my code:
# Import & Data Preprocessing
```python
import pandas as pd
train_ds=pd.read_csv('/content/train.csv')
A quick way to check for data information is using ".info()." From this, there are no catergorical data types. There are also no missing values in the data. Overall, there are no variables that should be taken out. All of them have a relationship between smoking one way or the other.
train_ds.info()
newdata = train_ds
# Define outliers function to count them using IQR method
def count_outliers(df, feature_names):
    outlier_counts = {}
    for feature in feature_names:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers in the dataframe
        count_below = (df[feature] < lower_bound).sum()
        count_above = (df[feature] > upper_bound).sum()
        outlier_counts[feature] = count_below + count_above

    return outlier_counts
features_to_check = [
    'height(cm)', 'weight(kg)', 'systolic', 'relaxation',
    'fasting blood sugar', 'Cholesterol', 'triglyceride',
    'HDL', 'LDL', 'hemoglobin',
    'eyesight(left)', 'eyesight(right)',
    'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp'
]


# Count outliers for each feature
outlier_counts = count_outliers(newdata, features_to_check)
outlier_countsimport matplotlib.pyplot as plt


plt.figure(figsize=(15, 10))

# Number of features to check
num_features = len(features_to_check)
for i, feature in enumerate(features_to_check, 1):
    plt.subplot((num_features + 1) // 2, 2, i)  # Create a subplot for each feature
    plt.boxplot(newdata[feature].dropna())
    plt.title(feature)

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()
import matplotlib.pyplot as plt

# Creating histograms for each numerical feature in the dataset
newdata.hist(figsize=(20, 15), bins=20, layout=(5, 5), color='skyblue', edgecolor='black')
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
# Add a new feature ->HDL:LDL cholesterol ratio
newdata['HDL_LDL_ratio'] = newdata['HDL'] / newdata['LDL']


import matplotlib.pyplot as plt

# Plot the histogram of the HDL to LDL ratio
import matplotlib.pyplot as plt

# Plot the histogram of the HDL to LDL ratio with bins 0 to 1
plt.figure(figsize=(10, 6))
plt.hist(newdata['HDL_LDL_ratio'], bins=20, range=(0, 1), color='blue', alpha=0.7)
plt.title('Distribution of HDL to LDL Cholesterol Ratios (0 to 1)')
plt.xlabel('HDL to LDL Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import pandas as pd

# Create function to remove outliers
def remove_outliers(df, feature_names):
    cleaned_data = df.copy()
    for feature in feature_names:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Condition to identify rows in DataFrame that are within the acceptable range
        outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        cleaned_data = cleaned_data[~outliers]

    return cleaned_data

# Features to check for outliers
features_to_check = [
'height(cm)', 'weight(kg)', 'systolic', 'relaxation',
    'fasting blood sugar', 'Cholesterol', 'triglyceride',
    'HDL', 'LDL', 'hemoglobin',
    'eyesight(left)', 'eyesight(right)',
    'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp'
]



# Remove outliers
cleaned_data = remove_outliers(newdata, features_to_check)


print("Original Data Size:", train_ds.shape)
print("Cleaned Data Size:", cleaned_data.shape)

from sklearn.model_selection import train_test_split


X = cleaned_data.drop('smoking', axis=1)  # Features
y = cleaned_data['smoking']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

# Initialize model
model = LogisticRegression()

# Set up the hyperparameters grid
param_distributions = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Initialize and fit the RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions, n_iter=15,
                                   cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# Print best parameters and best score
print("Best parameters:", random_search.best_params_)
print("Best ROC AUC score: {:.2f}".format(random_search.best_score_))

# Predict probabilities for the test data using the best estimator found by RandomizedSearchCV

from sklearn.metrics import roc_auc_score

# Predict probabilities for the test set
y_pred_proba = random_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC AUC on the test set
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print ROC AUC score for the test set
print("Test ROC AUC score: {:.2f}".format(test_roc_auc))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities for the test data using the best estimator found by RandomizedSearchCV
y_pred_proba = random_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Generate ROC curve values: fpr (False Positive Rate), tpr (True Positive Rate), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


test_df = pd.read_csv('/content/test.csv')

test_df['HDL_LDL_ratio'] = test_df['HDL'] / test_df['LDL']  # Same feature engineering as training
test_df = test_df.dropna()
X_test_final = sc.transform(test_df)  # Apply the same scaler as the training data

# Predict using the trained model
final_predictions = random_search.best_estimator_.predict_proba(X_test_final)[:, 1]

# Create a DataFrame for cvs
test_ids = test_df['id'].copy()
submission_df = pd.DataFrame({
    'id': test_ids,
    'smoking_probability': final_predictions
})

# Save the cvs file
import os
submission_file_path = '/content/data/smoker_submission.csv'
os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
submission_df.to_csv(submission_file_path, index=False)
from google.colab import files
files.download('/content/data/smoker_submission.csv')
```

