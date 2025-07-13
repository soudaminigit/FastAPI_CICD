import pandas as pd
from scipy.stats import ks_2samp
import sys

# Load reference data (previous distribution) and current data
reference = pd.read_csv("reference_data.tsv",delimiter='\t')
current = pd.read_csv("current_data.tsv",delimiter='\t')

# Select columns to monitor for drift
columns_to_check = ["sqft", "bedrooms", "age"]

# Set a threshold for drift detection (p < 0.05 indicates drift)
DRIFT_THRESHOLD = 0.05
drift_detected = False

print(" Checking for drift...")
for column in columns_to_check:
    stat, p_value = ks_2samp(reference[column], current[column])
    print(f"{column}: p-value = {p_value:.4f}")
    if p_value < DRIFT_THRESHOLD:
        print(f"Drift detected in column: {column}")
        drift_detected = True

if drift_detected:
    sys.exit(1)  # Failure in GitHub Actions triggers retraining
else:
    print(" No significant drift detected.")
    sys.exit(0)

