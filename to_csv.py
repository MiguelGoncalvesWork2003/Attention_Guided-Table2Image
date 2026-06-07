import pandas as pd

FILE = "breast-cancer-wisconsin"   # change dataset name here
DATA_FILE = f"data/raw/{FILE}.data"

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(DATA_FILE, header=None)

# -------------------------------
# Assign target column
# -------------------------------
df = df.rename(columns={10: "Class"})

# -------------------------------
# Replace '?' with marker
# -------------------------------
df.replace('?', pd.NA, inplace=True)

# -------------------------------
# Save CSV
# -------------------------------
df.to_csv(f"data/raw/{FILE}.csv", index=False)

print("Converted to CSV successfully!")
print("Final shape:", df.shape)
print(df.head())
