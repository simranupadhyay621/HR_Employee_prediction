import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"D:\Python Project\Human_Resources.csv")
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
sns.countplot(x="Attrition", data=df)
plt.show()
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(numeric_only=True),
            cmap="coolwarm")
plt.show()
sns.histplot(df["Age"], bins=20)
plt.show()
sns.boxplot(x="Attrition",
            y="MonthlyIncome",
            data=df)
plt.show()
sns.countplot(x="Department",
              hue="Attrition",
              data=df)

plt.xticks(rotation=30)
plt.show()
sns.countplot(x="Gender",
              hue="Attrition",
              data=df)
plt.show()
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Human_Resources.csv")

# Drop useless columns
df = df.drop(["EmployeeCount",
              "EmployeeNumber",
              "Over18",
              "StandardHours"], axis=1)

# Encode target
le_target = LabelEncoder()
df["Attrition"] = le_target.fit_transform(df["Attrition"])

# Encode categorical columns
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))
pickle.dump(encoders,open("encoders.pkl","wb"))
