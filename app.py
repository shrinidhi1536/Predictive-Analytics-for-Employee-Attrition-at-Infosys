import start
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'sentinel_secret_2024'

# Load model and data
MODEL_PATH = 'attrition_model.joblib'
DATA_PATH = 'HR_Data.csv'

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df['OverTime_Encoded'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Risk_Score'] = (model.predict_proba(
    df[['JobSatisfaction','MonthlyIncome','YearsAtCompany','DistanceFromHome','OverTime_Encoded']]
)[:, 1] * 100).round(1)

# ─── AUTH ────────────────────────────────────────────────────────────────────

USERS = {
    'admin@infosys.com': {'password': 'admin', 'role': 'hr'},
    'user@infosys.com':  {'password': 'admin', 'role': 'employee'},
}

# ─── MESSAGES STORE ──────────────────────────────────────────────────────────
# Structure: { employee_id: [ { id, from, subject, body, timestamp, read, priority } ] }
MESSAGES = {}
_msg_counter = 0

def _new_message_id():
    global _msg_counter
    _msg_counter += 1
    return f'msg_{_msg_counter}'

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('signin'))

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    role = data.get('role', '')

    user = USERS.get(email)
    if user and user['password'] == password and user['role'] == role:
        session['user'] = email
        session['role'] = role
        redirect_to = '/dashboard' if role == 'hr' else '/user_dashboard'
        return jsonify({'success': True, 'redirect': redirect_to})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('signin'))

@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email', '').strip()
    new_pass = data.get('new_password', '')
    if email in USERS and new_pass:
        USERS[email]['password'] = new_pass
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid email'})

# ─── HR ADMIN PAGES ──────────────────────────────────────────────────────────

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# ─── USER PAGES ──────────────────────────────────────────────────────────────

@app.route('/user_dashboard')
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route('/user_performance')
def user_performance():
    return render_template('user_performance.html')

@app.route('/user_learning')
def user_learning():
    return render_template('user_learning.html')

@app.route('/user_settings')
def user_settings():
    return render_template('user_settings.html')

@app.route('/user_messages')
def user_messages():
    return render_template('user_messages.html')

# ─── MESSAGING API ───────────────────────────────────────────────────────────

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """HR sends a message to an employee. Called from the Alerts page."""
    # Only HR can send
    if session.get('role') != 'hr':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.get_json()
    employee_id = data.get('employee_id', '').strip()   # e.g. 'INF_1001'
    subject     = data.get('subject', '').strip()
    body        = data.get('body', '').strip()
    priority    = data.get('priority', 'normal')        # 'normal' | 'high' | 'urgent'

    if not employee_id or not body:
        return jsonify({'success': False, 'message': 'employee_id and body are required'}), 400

    msg = {
        'id':        _new_message_id(),
        'from':      'HR Team',
        'from_email': session.get('user'),
        'subject':   subject or 'Message from HR',
        'body':      body,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'read':      False,
        'priority':  priority,
    }

    MESSAGES.setdefault(employee_id, []).append(msg)
    return jsonify({'success': True, 'message_id': msg['id']})


@app.route('/api/get_messages', methods=['GET'])
def get_messages():
    """Employee fetches their own messages."""
    if session.get('role') != 'employee':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    # The logged-in user's employee_id — map email → id
    # In your current system employees log in as user@infosys.com → we map to a fixed id.
    # Extend this mapping when you have more users.
    EMAIL_TO_ID = {
        'user@infosys.com': 'INF_1000',
    }
    employee_id = EMAIL_TO_ID.get(session.get('user'), session.get('user'))
    msgs = MESSAGES.get(employee_id, [])

    # Return newest first
    sorted_msgs = sorted(msgs, key=lambda m: m['timestamp'], reverse=True)
    unread_count = sum(1 for m in msgs if not m['read'])

    return jsonify({
        'success': True,
        'messages': sorted_msgs,
        'unread_count': unread_count,
    })


@app.route('/api/mark_read', methods=['POST'])
def mark_read():
    """Employee marks a message (or all messages) as read."""
    if session.get('role') != 'employee':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    EMAIL_TO_ID = {
        'user@infosys.com': 'INF_1000',
    }
    employee_id = EMAIL_TO_ID.get(session.get('user'), session.get('user'))

    data = request.get_json()
    message_id = data.get('message_id')   # pass 'all' to mark all read

    msgs = MESSAGES.get(employee_id, [])
    for m in msgs:
        if message_id == 'all' or m['id'] == message_id:
            m['read'] = True

    return jsonify({'success': True})


@app.route('/api/unread_count', methods=['GET'])
def unread_count():
    """Quick poll endpoint — returns just the unread count for the notification bell."""
    if session.get('role') != 'employee':
        return jsonify({'count': 0})

    EMAIL_TO_ID = {
        'user@infosys.com': 'INF_1000',
    }
    employee_id = EMAIL_TO_ID.get(session.get('user'), session.get('user'))
    msgs = MESSAGES.get(employee_id, [])
    count = sum(1 for m in msgs if not m['read'])
    return jsonify({'count': count})


# ─── EXISTING API ENDPOINTS ──────────────────────────────────────────────────

@app.route('/api/dashboard_stats')
def dashboard_stats():
    total = len(df)
    attrited = int(df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).sum())
    active = total - attrited
    high_risk = int((df['Risk_Score'] > 70).sum())
    attrition_rate = round((attrited / total) * 100, 1)

    dept_attrition = df.groupby('Department').apply(
        lambda g: round((g['Attrition'] == 'Yes').sum() / len(g) * 100, 1)
    ).to_dict()

    monthly = {}
    for q, months in {'Q1': ['Jan','Feb','Mar'], 'Q2': ['Apr','May','Jun'],
                      'Q3': ['Jul','Aug','Sep'], 'Q4': ['Oct','Nov','Dec']}.items():
        monthly[q] = active

    return jsonify({
        'total': total,
        'active': active,
        'attrited': attrited,
        'attrition_rate': attrition_rate,
        'high_risk': high_risk,
        'dept_attrition': dept_attrition,
        'quarters': list(monthly.keys()),
        'active_trend': [1150, 1180, int(active * 0.97), active]
    })

@app.route('/api/insights_data')
def insights_data():
    dept_counts = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
    dept_data = {}
    for dept in dept_counts.index:
        dept_data[dept] = {
            'yes': int(dept_counts.loc[dept, 'Yes']) if 'Yes' in dept_counts.columns else 0,
            'no': int(dept_counts.loc[dept, 'No']) if 'No' in dept_counts.columns else 0
        }

    ot_data = df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0).to_dict()

    sat_data = df.groupby(['JobSatisfaction', 'Attrition']).size().unstack(fill_value=0)
    sat_result = {}
    for level in sat_data.index:
        sat_result[str(level)] = {
            'yes': int(sat_data.loc[level, 'Yes']) if 'Yes' in sat_data.columns else 0,
            'no': int(sat_data.loc[level, 'No']) if 'No' in sat_data.columns else 0
        }

    income_bins = pd.cut(df['MonthlyIncome'], bins=5)
    income_labels = [str(b) for b in income_bins.cat.categories]

    return jsonify({
        'department': dept_data,
        'overtime': {
            'yes_attrited': int(df[(df['OverTime']=='Yes') & (df['Attrition']=='Yes')].shape[0]),
            'yes_retained': int(df[(df['OverTime']=='Yes') & (df['Attrition']=='No')].shape[0]),
            'no_attrited': int(df[(df['OverTime']=='No') & (df['Attrition']=='Yes')].shape[0]),
            'no_retained': int(df[(df['OverTime']=='No') & (df['Attrition']=='No')].shape[0]),
        },
        'satisfaction': sat_result,
        'age_groups': {
            '20-30': int(df[(df['Age']>=20)&(df['Age']<30)]['Attrition'].eq('Yes').sum()),
            '30-40': int(df[(df['Age']>=30)&(df['Age']<40)]['Attrition'].eq('Yes').sum()),
            '40-50': int(df[(df['Age']>=40)&(df['Age']<50)]['Attrition'].eq('Yes').sum()),
            '50+':   int(df[df['Age']>=50]['Attrition'].eq('Yes').sum()),
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        satisfaction = int(data['satisfaction'])
        income = int(data['income'])
        years = int(data['years'])
        distance = int(data['distance'])
        overtime = 1 if data['overtime'] == 'Yes' else 0

        features = [[satisfaction, income, years, distance, overtime]]
        prob = model.predict_proba(features)[0][1]
        risk = round(prob * 100, 1)

        factors = []
        if overtime: factors.append('Overtime work significantly increases burnout risk')
        if satisfaction <= 2: factors.append('Low job satisfaction is a primary churn driver')
        if income < 4000: factors.append('Below-average compensation reduces retention probability')
        if years < 2: factors.append('Short tenure employees have higher attrition rates')
        if distance > 20: factors.append('Long commute distance contributes to disengagement')

        if risk > 75:
            level = 'CRITICAL'
            color = '#ef4444'
            recommendation = 'Immediate HR intervention required. Schedule 1-on-1 meeting within 48 hours.'
        elif risk > 50:
            level = 'ELEVATED'
            color = '#f59e0b'
            recommendation = 'Monitor closely. Review compensation and workload balance.'
        elif risk > 25:
            level = 'MODERATE'
            color = '#3b82f6'
            recommendation = 'Engage proactively. Consider recognition programs and career pathing.'
        else:
            level = 'LOW'
            color = '#10b981'
            recommendation = 'Employee shows strong retention signals. Continue current engagement strategy.'

        return jsonify({
            'risk': risk,
            'level': level,
            'color': color,
            'recommendation': recommendation,
            'factors': factors if factors else ['Profile shows balanced risk factors']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_alerts_data')
def get_alerts_data():
    try:
        high_risk = df[df['Risk_Score'] > 55].copy()
        high_risk = high_risk.sort_values('Risk_Score', ascending=False).head(20)

        employees = []
        for i, (idx, row) in enumerate(high_risk.iterrows()):
            employees.append({
                'id': f'INF_{1000 + i}',
                'name': f'Employee {1000 + i}',
                'dept': row['Department'],
                'role': f"Level {row['JobLevel']} Associate",
                'score': float(row['Risk_Score']),
                'income': int(row['MonthlyIncome']),
                'years': int(row['YearsAtCompany']),
                'dist': int(row['DistanceFromHome']),
                'overtime': row['OverTime'],
                'satisfaction': int(row['JobSatisfaction']),
                'balance': int(row['WorkLifeBalance']),
            })
        return jsonify({'employees': employees})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_analytics')
def model_analytics():
    return render_template('model_analytics.html')

@app.route('/bulk_predict')
def bulk_predict():
    return render_template('bulk_predict.html')

@app.route('/api/model_metrics')
def api_model_metrics():
    import json
    with open('static/model_metrics.json') as f:
        return jsonify(json.load(f))

@app.route('/api/bulk_predictions')
def api_bulk_predictions():
    try:
        dept   = request.args.get('dept', '')
        sort   = request.args.get('sort', 'risk_desc')
        limit  = int(request.args.get('limit', 50))

        result_df = df.copy()
        if dept:
            result_df = result_df[result_df['Department'] == dept]

        result_df = result_df.copy()
        result_df['Predicted'] = model.predict(
            result_df[['JobSatisfaction','MonthlyIncome','YearsAtCompany','DistanceFromHome','OverTime_Encoded']]
        )

        if sort == 'risk_desc':
            result_df = result_df.sort_values('Risk_Score', ascending=False)
        elif sort == 'risk_asc':
            result_df = result_df.sort_values('Risk_Score', ascending=True)
        elif sort == 'income':
            result_df = result_df.sort_values('MonthlyIncome', ascending=False)

        result_df = result_df.head(limit)

        employees = []
        for i, (idx, row) in enumerate(result_df.iterrows()):
            employees.append({
                'id': f'INF_{1000 + i}',
                'dept': row['Department'],
                'income': int(row['MonthlyIncome']),
                'years': int(row['YearsAtCompany']),
                'satisfaction': int(row['JobSatisfaction']),
                'overtime': row['OverTime'],
                'distance': int(row['DistanceFromHome']),
                'risk': float(row['Risk_Score']),
                'predicted_attrition': bool(row['Predicted']),
                'actual': row['Attrition'],
            })
        return jsonify({'employees': employees, 'total': len(result_df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shap_explain', methods=['POST'])
def shap_explain():
    """Return SHAP values for a single employee prediction."""
    try:
        import shap as shap_lib
        data = request.get_json()
        satisfaction = int(data['satisfaction'])
        income       = int(data['income'])
        years        = int(data['years'])
        distance     = int(data['distance'])
        overtime     = 1 if data['overtime'] == 'Yes' else 0

        import pandas as pd
        X_single = pd.DataFrame([[satisfaction, income, years, distance, overtime]],
                                 columns=['JobSatisfaction','MonthlyIncome','YearsAtCompany',
                                          'DistanceFromHome','OverTime_Encoded'])
        explainer   = shap_lib.TreeExplainer(model)
        shap_raw    = explainer.shap_values(X_single)
        sv          = np.array(shap_raw)[0, :, 1]
        base_val    = float(explainer.expected_value[1])

        labels = ['Job Satisfaction','Monthly Income','Years at Company',
                  'Distance from Home','Overtime']
        values = [float(v) for v in sv]

        risk = float(model.predict_proba(X_single)[0][1] * 100)

        return jsonify({
            'risk': round(risk, 1),
            'base_value': round(base_val, 4),
            'features': labels,
            'shap_values': values,
            'feature_values': [satisfaction, income, years, distance,
                               'Yes' if overtime else 'No']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)