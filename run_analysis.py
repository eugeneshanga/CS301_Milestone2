import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


os.makedirs("output", exist_ok=True)

'''
Load data
'''
evictions = pd.read_csv("Evictions_20260428.csv", low_memory=False)

# Standardize column names
evictions.columns = (
    evictions.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

'''
Clean eviction data
'''
evictions["executed_date"] = pd.to_datetime(evictions["executed_date"], errors="coerce")
evictions["year"] = evictions["executed_date"].dt.year
evictions["month"] = evictions["executed_date"].dt.month

evictions = evictions.dropna(subset=["executed_date", "borough"])

monthly = (
    evictions
    .groupby(["borough", "year", "month"])
    .size()
    .reset_index(name="eviction_count")
)

'''
Add borough-level demographic values
Based on final notebook variables
'''
demo_values = {
    "BRONX": {
        "total_population": 1455720,
        "housing_units": 525917,
        "median_age": 33.6,
        "female_pct": 52.9,
        "hispanic_pct": 55.7,
        "white_non_hispanic_pct": 9.3,
        "black_non_hispanic_pct": 29.5,
        "asian_non_hispanic_pct": 4.1,
    },
    "BROOKLYN": {
        "total_population": 2635121,
        "housing_units": 1028383,
        "median_age": 35.4,
        "female_pct": 52.5,
        "hispanic_pct": 19.2,
        "white_non_hispanic_pct": 35.8,
        "black_non_hispanic_pct": 31.7,
        "asian_non_hispanic_pct": 11.6,
    },
    "MANHATTAN": {
        "total_population": 1643734,
        "housing_units": 872645,
        "median_age": 37.5,
        "female_pct": 52.6,
        "hispanic_pct": 25.9,
        "white_non_hispanic_pct": 46.1,
        "black_non_hispanic_pct": 12.9,
        "asian_non_hispanic_pct": 11.9,
    },
    "QUEENS": {
        "total_population": 2339280,
        "housing_units": 850422,
        "median_age": 38.3,
        "female_pct": 51.4,
        "hispanic_pct": 27.7,
        "white_non_hispanic_pct": 25.6,
        "black_non_hispanic_pct": 17.2,
        "asian_non_hispanic_pct": 24.8,
    },
    "STATEN ISLAND": {
        "total_population": 474558,
        "housing_units": 179179,
        "median_age": 39.7,
        "female_pct": 51.4,
        "hispanic_pct": 18.1,
        "white_non_hispanic_pct": 62.9,
        "black_non_hispanic_pct": 9.5,
        "asian_non_hispanic_pct": 7.5,
    },
}

demo_df = pd.DataFrame.from_dict(demo_values, orient="index").reset_index()
demo_df = demo_df.rename(columns={"index": "borough"})

monthly = monthly.merge(demo_df, on="borough", how="left")

monthly["evictions_per_1000_housing_units"] = (
    monthly["eviction_count"] / monthly["housing_units"] * 1000
)

monthly = monthly.dropna()

'''
Eviction rate distribution
'''
plt.figure(figsize=(8, 5))
plt.hist(monthly["evictions_per_1000_housing_units"], bins=30)
plt.title("Distribution of Monthly Eviction Rates")
plt.xlabel("Evictions per 1,000 Housing Units")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/01_eviction_rate_distribution.png")
plt.close()

'''
Average eviction rate by borough
'''
borough_avg = (
    monthly
    .groupby("borough")["evictions_per_1000_housing_units"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
borough_avg.plot(kind="bar")
plt.title("Average Monthly Eviction Rate by Borough")
plt.xlabel("Borough")
plt.ylabel("Evictions per 1,000 Housing Units")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output/02_avg_eviction_rate_by_borough.png")
plt.close()

'''
Eviction trends over time by borough
'''
yearly_borough = (
    monthly
    .groupby(["year", "borough"])["evictions_per_1000_housing_units"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))
for borough in yearly_borough["borough"].unique():
    subset = yearly_borough[yearly_borough["borough"] == borough]
    plt.plot(
        subset["year"],
        subset["evictions_per_1000_housing_units"],
        marker="o",
        label=borough
    )

plt.title("Eviction Rate Trends Over Time by Borough")
plt.xlabel("Year")
plt.ylabel("Evictions per 1,000 Housing Units")
plt.legend()
plt.tight_layout()
plt.savefig("output/03_eviction_trends_by_borough.png")
plt.close()

'''
Correlation heatmap
'''
corr_cols = [
    "eviction_count",
    "evictions_per_1000_housing_units",
    "total_population",
    "housing_units",
    "median_age",
    "female_pct",
    "hispanic_pct",
    "white_non_hispanic_pct",
    "black_non_hispanic_pct",
    "asian_non_hispanic_pct",
]

corr = monthly[corr_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.title("Correlation Matrix of Evictions and Demographics")
plt.tight_layout()
plt.savefig("output/04_correlation_matrix.png")
plt.close()

''''
Random Forest model
No year/month, to focus on demographic/housing features
'''
features = [
    "total_population",
    "housing_units",
    "median_age",
    "female_pct",
    "hispanic_pct",
    "white_non_hispanic_pct",
    "black_non_hispanic_pct",
    "asian_non_hispanic_pct",
]

target = "evictions_per_1000_housing_units"

model_data = monthly[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save metrics
with open("output/model_metrics.txt", "w") as f:
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")

''' 
Actual vs predicted'
'''

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.title("Actual vs Predicted Eviction Rates")
plt.xlabel("Actual Evictions per 1,000 Housing Units")
plt.ylabel("Predicted Evictions per 1,000 Housing Units")
plt.tight_layout()
plt.savefig("output/05_actual_vs_predicted.png")
plt.close()

''' 
Feature importance
'''
importance = pd.Series(rf.feature_importances_, index=features).sort_values()

plt.figure(figsize=(9, 5))
importance.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("output/06_feature_importance.png")
plt.close()

print("Analysis complete. Outputs saved to /output.")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")