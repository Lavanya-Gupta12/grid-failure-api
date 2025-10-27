import pandas as pd

# Check predictions CSV
print("=== PREDICTIONS CSV COLUMNS ===")
pred_df = pd.read_csv("predictions_for_person_b.csv")
print(pred_df.columns.tolist())
print(f"\nFirst few rows:")
print(pred_df.head())

print("\n=== FEATURES CSV COLUMNS ===")
feat_df = pd.read_csv("features_v4.csv")
print(feat_df.columns.tolist())
print(f"\nFirst few rows:")
print(feat_df.head())