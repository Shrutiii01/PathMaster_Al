import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import sys, sklearn, joblib, numpy
print(sys.version)
print(sklearn.__version__)
print(joblib.__version__)
print(numpy.__version__)

# 1. Load Data
df = pd.read_csv('Updated_Student_Performance.csv')

# 2. Select ONLY the features we want to use
# We drop 'Student ID' and 'Parent Education Level' to avoid errors in the app
features = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities']
X = df[features].copy()
y = df['Passed'].map({'Yes': 1, 'No': 0}).fillna(0)

# Fill missing values
X['Study Hours per Week'] = X['Study Hours per Week'].fillna(X['Study Hours per Week'].median())
X['Attendance Rate'] = X['Attendance Rate'].fillna(X['Attendance Rate'].median())
X['Previous Grades'] = X['Previous Grades'].fillna(X['Previous Grades'].median())
X['Participation in Extracurricular Activities'] = X['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0}).fillna(0)

# 3. Train Model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save
joblib.dump(model, 'student_gb_model.pkl')

print("✅ New Brain Created! Model is now synced with your app inputs.")