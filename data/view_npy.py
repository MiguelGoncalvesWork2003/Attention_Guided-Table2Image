import numpy as np
import pandas as pd

import os
DATASET = os.environ.get("DATASET", "BreastCancer")  # default if env not set

# Load the file
X_train_img = np.load(f"data/processed/{DATASET}/feature_names.npy")
print(X_train_img)

# Check shape
print("Shape:", X_train_img.shape)

# Look at first few samples
print("First sample:\n", X_train_img[0])

# Look at last sample
print("Last sample:\n", X_train_img[-1])

# Load array
X_train_img = np.load(f"data/processed/{DATASET}/X_train_img.npy")
sample = X_train_img[0]

# Load step assignment
step_df = pd.read_csv(f"tabnet_fs/outputs/output_{DATASET}/tabnet_step_assignment.csv")

# Build mapping
step_groups = {step: step_df[step_df["dominant_step"] == step]["feature"].tolist()
               for step in range(sample.shape[0])}

# Show sample with feature names
for step_idx, features in step_groups.items():
    values = sample[step_idx]
    print(f"Step {step_idx}:")
    for col_idx, f in enumerate(features):
        print(f"  {f}: {values[col_idx]}")
