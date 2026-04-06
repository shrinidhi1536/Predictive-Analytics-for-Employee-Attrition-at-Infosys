**Sentinel — AI-Powered Employee Attrition Intelligence System**

Sentinel is an intelligent analytics platform designed to predict employee attrition risk and provide actionable insights for HR teams.

**Features**

Attrition Prediction using Machine Learning
HR Analytics Dashboard
AI Risk Lab
Retention Alerts System
Dual Role Access (HR Admin & Employee)

**Tech Stack**
Backend: Python, Flask
ML Libraries: Scikit-learn, Pandas, NumPy
Frontend: HTML, CSS, JavaScript
Deployment: Render

**Install dependencies:**
pip install -r requirements.txt

**Run:**
python app.py

**Open:**
http://localhost:5000

**Login**
HR Admin: admin@infosys.com / admin  
Employee: user@infosys.com / admin

**Deployment**

cd C:\Users\SHRINIDHI\Downloads\sentinel_app_v3\sentinel

python -c "
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib, json, numpy as np

df = pd.read_csv('HR_Data.csv')
df['OverTime_Encoded'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
features = ['JobSatisfaction','MonthlyIncome','YearsAtCompany','DistanceFromHome','OverTime_Encoded']
feature_labels = ['Job Satisfaction','Monthly Income','Years at Company','Distance from Home','Overtime']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'attrition_model.joblib')

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
cm = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, output_dict=True)

import shap
explainer = shap.TreeExplainer(model)
shap_raw = explainer.shap_values(X_test[:100])
sv = np.array(shap_raw)[:,:,1]
mean_shap = dict(zip(feature_labels, np.abs(sv).mean(axis=0).tolist()))

metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred)*100, 1),
    'f1': round(f1_score(y_test, y_pred)*100, 1),
    'auc': round(roc_auc_score(y_test, y_prob)*100, 1),
    'precision': round(report['1']['precision']*100, 1),
    'recall': round(report['1']['recall']*100, 1),
    'confusion_matrix': cm,
    'feature_importance': dict(zip(feature_labels, model.feature_importances_.tolist())),
    'shap_mean': mean_shap,
    'class_report': {
        'attrition': {'p': round(report['1']['precision']*100,1), 'r': round(report['1']['recall']*100,1), 'f1': round(report['1']['f1-score']*100,1), 'support': int(report['1']['support'])},
        'retained':  {'p': round(report['0']['precision']*100,1), 'r': round(report['0']['recall']*100,1), 'f1': round(report['0']['f1-score']*100,1), 'support': int(report['0']['support'])}
    }
}

import os
os.makedirs('static', exist_ok=True)
with open('static/model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print('Done! Accuracy:', metrics['accuracy'], '%  F1:', metrics['f1'], '%  AUC:', metrics['auc'], '%')
"

python app.py

**Deployment link:**
https://predictive-analytics-for-employee.onrender.com