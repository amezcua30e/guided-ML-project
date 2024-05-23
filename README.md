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

Here's my code: 


# Binary Prediction of Smoker Status using Bio-Signals
# Import & Data Preprocessing
import pandas as pd
train_ds=pd.read_csv('/content/train.csv')
A quick way to check for data information is using ".info()." From this, there are no catergorical data types. There are also no missing values in the data. Overall, there are no variables that should be taken out. All of them have a relationship between smoking one way or the other.
train_ds.info()
newdata = train_ds
Check for outliers in the dataset using the IQR method.
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
outlier_counts


