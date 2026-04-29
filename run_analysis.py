import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

# load data
evictions = pd.read_csv("Evictions_20260428.csv")

# simple example plot (you can improve this)
evictions["Executed Date"] = pd.to_datetime(evictions["Executed Date"], errors="coerce")
evictions["year"] = evictions["Executed Date"].dt.year

yearly = evictions.groupby("year").size()

plt.figure()
yearly.plot()
plt.title("Evictions Over Time")
plt.xlabel("Year")
plt.ylabel("Count")

plt.savefig("output/evictions_over_time.png")