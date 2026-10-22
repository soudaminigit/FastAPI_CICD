import os
import pandas as pd
from scipy.stats import ks_2samp

# Load reference data (what the model was trained on) and the new data
reference = pd.read_csv("reference_data.tsv", delimiter="\t")
current = pd.read_csv("current_data.tsv", delimiter="\t")

# Columns to monitor for drift
columns_to_check = ["sqft", "bedrooms", "age"]

# A LOW p-value means the two distributions differ, i.e. drift
DRIFT_THRESHOLD = 0.05
drift_detected = False

print("Checking for drift...")
for column in columns_to_check:
    stat, p_value = ks_2samp(reference[column], current[column])
    print(f"{column}: KS statistic = {stat:.4f}, p-value = {p_value:.4f}")
    if p_value < DRIFT_THRESHOLD:
        print(f"  -> Drift detected in column: {column}")
        drift_detected = True

print("Drift detected." if drift_detected else "No significant drift detected.")

# Report the result to GitHub Actions. The script always exits 0 so that a
# real crash (missing file, bad column) is not mistaken for drift.
output_file = os.environ.get("GITHUB_OUTPUT")
if output_file:
    with open(output_file, "a") as f:
        f.write(f"drift={'true' if drift_detected else 'false'}\n")
