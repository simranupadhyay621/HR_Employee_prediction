# ============================================================
# HR EMPLOYEE ATTRITION PREDICTION
# PART 1 : DATA CLEANING
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv(r"D:\Python Project\Human_Resources.csv")

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

# ------------------------------------------------------------
# Display First Rows
# ------------------------------------------------------------

print(df.head())

# ------------------------------------------------------------
# Shape
# ------------------------------------------------------------

print("\nDataset Shape")
print(df.shape)

# ------------------------------------------------------------
# Information
# ------------------------------------------------------------

print("\nDataset Information")

print(df.info())

# ------------------------------------------------------------
# Data Types
# ------------------------------------------------------------

print("\nData Types")

print(df.dtypes)

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\nMissing Values")

print(df.isnull().sum())

# ------------------------------------------------------------
# Duplicate Rows
# ------------------------------------------------------------

print("\nDuplicate Rows")

print(df.duplicated().sum())

# Remove Duplicate Rows

df = df.drop_duplicates()

# ------------------------------------------------------------
# Unique Values
# ------------------------------------------------------------

print("\nUnique Values")

for col in df.columns:
    print(col,":",df[col].nunique())

# ------------------------------------------------------------
# Target Variable
# ------------------------------------------------------------

print("\nTarget Distribution")

print(df["Attrition"].value_counts())

# ------------------------------------------------------------
# Convert Yes/No Columns
# ------------------------------------------------------------

df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

df["OverTime"] = df["OverTime"].map({"Yes":1,"No":0})

df["Over18"] = df["Over18"].map({"Y":1,"N":0})

# ------------------------------------------------------------
# Drop Unnecessary Columns
# ------------------------------------------------------------

drop_columns = [

    "EmployeeCount",
    "EmployeeNumber",
    "StandardHours",
    "Over18"

]

df.drop(columns=drop_columns,inplace=True)

# ------------------------------------------------------------
# Check Missing Values Again
# ------------------------------------------------------------

print(df.isnull().sum())

# ------------------------------------------------------------
# Save Clean Dataset
# ------------------------------------------------------------

df.to_csv("clean_hr_data.csv",index=False)

print("\nClean Dataset Saved Successfully!")

print(df.head())

print(df.shape)

# ============================================================
# HR EMPLOYEE ATTRITION PREDICTION
# PART 2 : EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset

df = pd.read_csv("clean_hr_data.csv")

# ============================================================
# BASIC INFORMATION
# ============================================================

print("="*70)
print("DATASET SHAPE")
print("="*70)

print(df.shape)

print("\n")

print("="*70)
print("COLUMN NAMES")
print("="*70)

print(df.columns.tolist())

print("\n")

print("="*70)
print("STATISTICAL SUMMARY")
print("="*70)

print(df.describe())

# ============================================================
# ATTRITION DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,5))

df["Attrition"].value_counts().plot(
    kind="bar",
    color=["steelblue","tomato"]
)

plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.xticks([0,1],["No","Yes"],rotation=0)

plt.show()

# ============================================================
# AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Age"],
    bins=20,
    edgecolor="black"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Employees")

plt.show()

# ============================================================
# GENDER DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,6))

df["Gender"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")
plt.title("Gender Distribution")

plt.show()

# ============================================================
# DEPARTMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,5))

df["Department"].value_counts().plot(
    kind="bar",
    color="orange"
)

plt.title("Department Distribution")
plt.xlabel("Department")
plt.ylabel("Employees")

plt.xticks(rotation=20)

plt.show()

# ============================================================
# BUSINESS TRAVEL
# ============================================================

plt.figure(figsize=(8,5))

df["BusinessTravel"].value_counts().plot(
    kind="bar",
    color="green"
)

plt.title("Business Travel")

plt.show()

# ============================================================
# EDUCATION LEVEL
# ============================================================

plt.figure(figsize=(7,5))

df["Education"].value_counts().sort_index().plot(
    kind="bar",
    color="purple"
)

plt.title("Education Level")

plt.show()

# ============================================================
# JOB LEVEL
# ============================================================

plt.figure(figsize=(7,5))

df["JobLevel"].value_counts().sort_index().plot(
    kind="bar",
    color="brown"
)

plt.title("Job Level")

plt.show()

# ============================================================
# JOB ROLE
# ============================================================

plt.figure(figsize=(12,6))

df["JobRole"].value_counts().plot(
    kind="bar"
)

plt.xticks(rotation=45)

plt.title("Job Role Distribution")

plt.show()

# ============================================================
# MARITAL STATUS
# ============================================================

plt.figure(figsize=(7,5))

df["MaritalStatus"].value_counts().plot(
    kind="bar",
    color="cyan"
)

plt.title("Marital Status")

plt.show()

# ============================================================
# OVERTIME
# ============================================================

plt.figure(figsize=(6,5))

df["OverTime"].value_counts().plot(
    kind="bar",
    color=["royalblue","red"]
)

plt.title("Overtime")

plt.xticks([0,1],["No","Yes"],rotation=0)

plt.show()

# ============================================================
# MONTHLY INCOME
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["MonthlyIncome"],
    bins=30,
    edgecolor="black"
)

plt.title("Monthly Income Distribution")

plt.xlabel("Income")

plt.ylabel("Employees")

plt.show()

# ============================================================
# YEARS AT COMPANY
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["YearsAtCompany"],
    bins=15,
    edgecolor="black"
)

plt.title("Years at Company")

plt.xlabel("Years")

plt.ylabel("Employees")

plt.show()

# ============================================================
# DISTANCE FROM HOME
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["DistanceFromHome"],
    bins=15,
    edgecolor="black"
)

plt.title("Distance From Home")

plt.show()

# ============================================================
# ENVIRONMENT SATISFACTION
# ============================================================

plt.figure(figsize=(7,5))

df["EnvironmentSatisfaction"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Environment Satisfaction")

plt.show()

# ============================================================
# JOB SATISFACTION
# ============================================================

plt.figure(figsize=(7,5))

df["JobSatisfaction"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Job Satisfaction")

plt.show()

# ============================================================
# WORK LIFE BALANCE
# ============================================================

plt.figure(figsize=(7,5))

df["WorkLifeBalance"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Work Life Balance")

plt.show()

# ============================================================
# PERFORMANCE RATING
# ============================================================

plt.figure(figsize=(6,5))

df["PerformanceRating"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Performance Rating")

plt.show()

# ============================================================
# CORRELATION HEATMAP
# ============================================================

numeric = df.select_dtypes(include=np.number)

corr = numeric.corr()

plt.figure(figsize=(16,12))

plt.imshow(corr)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.colorbar()

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# HR EMPLOYEE ATTRITION PREDICTION
# PART 3 : FEATURE ENGINEERING & PREPROCESSING
# ============================================================

# Import Libraries

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("clean_hr_data.csv")

print("="*70)
print("DATA LOADED SUCCESSFULLY")
print("="*70)

print(df.head())

# ============================================================
# TARGET VARIABLE
# ============================================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

print("\nTarget Distribution\n")
print(y.value_counts())

# ============================================================
# IDENTIFY CATEGORICAL & NUMERICAL COLUMNS
# ============================================================

cat_cols = X.select_dtypes(include="object").columns.tolist()

num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical Columns")
print(cat_cols)

print("\nNumerical Columns")
print(num_cols)

# ============================================================
# ONE HOT ENCODING
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded = encoder.fit_transform(X[cat_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)

# ============================================================
# NUMERICAL DATA
# ============================================================

numeric_df = X[num_cols].reset_index(drop=True)

# ============================================================
# MERGE DATA
# ============================================================

X = pd.concat(
    [numeric_df, encoded_df],
    axis=1
)

print("\nShape After Encoding")
print(X.shape)

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled Dataset")

print(X_scaled.head())

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape")

print(X_train.shape)

print(y_train.shape)

print("\nTesting Shape")

print(X_test.shape)

print(y_test.shape)

# ============================================================
# SAVE FILES
# ============================================================

joblib.dump(encoder,"encoder.pkl")

joblib.dump(scaler,"scaler.pkl")

X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)

y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

print("\nFiles Saved Successfully")

print("""
Saved Files
-------------------------
encoder.pkl
scaler.pkl
X_train.csv
X_test.csv
y_train.csv
y_test.csv
""")

# ============================================================
# HR EMPLOYEE ATTRITION PREDICTION
# PART 4 : MACHINE LEARNING MODEL TRAINING
# ============================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

print("="*60)
print("DATA LOADED SUCCESSFULLY")
print("="*60)

# ==========================
# Create Models
# ==========================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42)

}

# ==========================
# Train & Evaluate Models
# ==========================

results = []

for name, model in models.items():

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)

    results.append([

        name,
        accuracy,
        precision,
        recall,
        f1,
        roc

    ])

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", roc)

    print("\nClassification Report\n")

    print(classification_report(y_test, pred))

    print("Confusion Matrix\n")

    print(confusion_matrix(y_test, pred))

# ==========================
# Model Comparison
# ==========================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"

    ]

)

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(results_df)

# ==========================
# Accuracy Comparison Plot
# ==========================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.show()

# ==========================
# Cross Validation
# ==========================

print("\n")
print("="*60)
print("5-FOLD CROSS VALIDATION")
print("="*60)

rf = RandomForestClassifier(random_state=42)

scores = cross_val_score(

    rf,

    X_train,

    y_train,

    cv=5,

    scoring="accuracy"

)

print("Scores:", scores)

print("Average Accuracy:", scores.mean())

# ==========================
# Hyperparameter Tuning
# ==========================

print("\n")
print("="*60)
print("GRID SEARCH")
print("="*60)

params = {

    "n_estimators":[100,200],

    "max_depth":[5,10,None],

    "min_samples_split":[2,5]

}

grid = GridSearchCV(

    RandomForestClassifier(random_state=42),

    param_grid=params,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(X_train, y_train)

print("Best Parameters")

print(grid.best_params_)

print("Best Score")

print(grid.best_score_)

# ==========================
# Best Model
# ==========================

best_model = grid.best_estimator_

prediction = best_model.predict(X_test)

print("\nFinal Accuracy")

print(accuracy_score(y_test, prediction))

# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 20 Important Features")

print(importance.head(20))

plt.figure(figsize=(10,8))

plt.barh(

    importance.head(15)["Feature"],

    importance.head(15)["Importance"]

)

plt.title("Top 15 Important Features")

plt.gca().invert_yaxis()

plt.show()

# ==========================
# Save Best Model
# ==========================

joblib.dump(best_model, "model.pkl")

print("\n")
print("="*60)
print("MODEL SAVED SUCCESSFULLY")
print("="*60)

print("Saved File : model.pkl")

