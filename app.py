from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import io

app = Flask(__name__)
@app.route('/')
def front():
    return render_template('front.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Read uploaded file
        file = request.files.get('file')
        user_service = request.form.get('service').strip()
        user_month = request.form.get('month').strip()

        if not file or not user_service or not user_month:
            return "❌ Missing CSV or input values.", 400

        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        df = df[df["Service"] != "Total costs"]
        df_long = df.melt(id_vars=["Service"], var_name="Month", value_name="Cost")

        le_service = LabelEncoder()
        le_month = LabelEncoder()
        df_long["ServiceEncoded"] = le_service.fit_transform(df_long["Service"])
        df_long["MonthEncoded"] = le_month.fit_transform(df_long["Month"])

        if user_service not in le_service.classes_ or user_month not in le_month.classes_:
            return "❌ Invalid Service or Month input."

        # 2. Prediction model
        X = df_long[["ServiceEncoded", "MonthEncoded"]]
        y = df_long["Cost"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict cost
        service_encoded = le_service.transform([user_service])[0]
        month_encoded = le_month.transform([user_month])[0]
        prediction = model.predict([[service_encoded, month_encoded]])[0]

        response = f"✅ Predicted AWS Cost for {user_service} in {user_month}: ${prediction:.2f}\n"

        # 3. Anomaly Detection
        response += "\n🔍 Detecting Anomalies (Sudden Cost Spikes):\n"
        anomalies = []

        for service, group in df_long.groupby("Service"):
            group_sorted = group.sort_values("Month")
            group_sorted["CostDiff"] = group_sorted["Cost"].diff()
            mean_diff = group_sorted["CostDiff"].mean()
            std_diff = group_sorted["CostDiff"].std()
            outliers = group_sorted[group_sorted["CostDiff"] > mean_diff + 2 * std_diff]
            for _, row in outliers.iterrows():
                response += f"⚠️ {row['Service']} spike of +${row['CostDiff']:.2f} in {row['Month']}\n"

        # 4. High Average Cost Services
        response += "\n💰 Services with Unusually High Average Costs:\n"
        high_cost_services = []
        service_avg = df_long.groupby("Service")["Cost"].mean()
        overall_avg = df_long["Cost"].mean()
        overall_std = df_long["Cost"].std()

        for service, avg_cost in service_avg.items():
            if avg_cost > overall_avg + 1.5 * overall_std:
                high_cost_services.append(service)
                response += f"🔥 {service} average cost: ${avg_cost:.2f}\n"

        # 5. Recommendations
        tips = {
            "Amazon EC2": [
                "✅ Use reserved instances",
                "✅ Auto-schedule shutdown",
                "✅ Right-size instances"
            ],
            "Amazon S3": [
                "✅ Use S3 Glacier",
                "✅ Delete unused objects",
                "✅ Enable intelligent tiering"
            ],
            "Amazon RDS": [
                "✅ Use auto-pause",
                "✅ Reserved instances",
                "✅ Optimize I/O usage"
            ],
            "Amazon CloudWatch": [
                "✅ Limit detailed monitoring",
                "✅ Delete unused logs",
                "✅ Set log retention policies"
            ]
        }

        response += "\n🧠 Cost Reduction Tips:\n"
        for service in high_cost_services:
            response += f"\n💼 {service} Tips:\n"
            for tip in tips.get(service, ["🔍 Review usage patterns."]):
                response += f"   - {tip}\n"

        return response

    except Exception as e:
        return f"❌ Server error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
