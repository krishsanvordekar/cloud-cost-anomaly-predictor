import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('extended_mock_cost_explorer_data.csv')
df = df[df["Service"] != "Total costs"]

# Reshape to long format
df_long = df.melt(id_vars=["Service"], var_name="Month", value_name="Cost")

# Encode services and months
le_service = LabelEncoder()
le_month = LabelEncoder()
df_long["ServiceEncoded"] = le_service.fit_transform(df_long["Service"])
df_long["MonthEncoded"] = le_month.fit_transform(df_long["Month"])

# Training Random Forest Regressor
X = df_long[["ServiceEncoded", "MonthEncoded"]]
y = df_long["Cost"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("✅ Model trained. RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# List services and months
services = list(le_service.classes_)
months = list(le_month.classes_)
print("\nAvailable Months:", months)

# User input
user_month = input("\n🔎 Enter a Month (e.g., March): ").strip()
if user_month not in months:
    print("❌ Invalid month.")
    exit()

# 1. Highest cost service in selected month
month_df = df_long[df_long["Month"] == user_month]
top_service_row = month_df.loc[month_df["Cost"].idxmax()]
print(f"\n💰 Highest Cost in {user_month}:")
print(f"   Service: {top_service_row['Service']}")
print(f"   Cost: ${top_service_row['Cost']:.2f}")

# 2. Cost comparison with previous and next month
month_index = months.index(user_month)
user_service = top_service_row['Service']

def get_cost(service, month):
    row = df_long[(df_long["Service"] == service) & (df_long["Month"] == month)]
    return row["Cost"].values[0] if not row.empty else None

prev_cost = get_cost(user_service, months[month_index - 1]) if month_index > 0 else None
next_cost = get_cost(user_service, months[month_index + 1]) if month_index < len(months) - 1 else None

if prev_cost:
    diff = top_service_row["Cost"] - prev_cost
    print(f"\n📉 Compared to Previous Month ({months[month_index - 1]}):")
    print(f"   Change: ${diff:.2f} {'increase' if diff > 0 else 'decrease'}")

if next_cost:
    diff = top_service_row["Cost"] - next_cost
    print(f"\n📈 Compared to Next Month ({months[month_index + 1]}):")
    print(f"   Change: ${diff:.2f} {'increase' if diff > 0 else 'decrease'}")

# 3. Anomaly detection
print("\n🔍 Anomaly Detection:")
for service, group in df_long.groupby("Service"):
    group = group.sort_values("MonthEncoded")
    group["CostDiff"] = group["Cost"].diff()
    mean_diff = group["CostDiff"].mean()
    std_diff = group["CostDiff"].std()
    outliers = group[group["CostDiff"] > mean_diff + 2 * std_diff]

    for _, row in outliers.iterrows():
        print(f"⚠️ Anomaly: {service} spiked by +${row['CostDiff']:.2f} in {row['Month']}")

# 4. Clustering using KMeans
pivot_df = df.pivot(index="Service", columns=None)
pivot_df = df.set_index("Service")
kmeans = KMeans(n_clusters=3, random_state=42)
pivot_df['Cluster'] = kmeans.fit_predict(pivot_df)

print("\n📊 Clustering Summary (KMeans on service costs):")
for cluster in range(3):
    members = pivot_df[pivot_df["Cluster"] == cluster].index.tolist()
    avg_cost = pivot_df.loc[members].drop("Cluster", axis=1).mean().mean()
    label = (
        "🟢 Low-cost, stable services" if cluster == 0 else
        "🟡 Medium-cost services" if cluster == 1 else
        "🔴 High-cost, volatile services"
    )
    print(f"\nCluster {cluster} - {label}:")
    print("   Services:", ", ".join(members))
    print(f"   Avg Monthly Cost: ${avg_cost:.2f}")

