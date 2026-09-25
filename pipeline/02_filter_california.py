import pandas as pd

INPUT_MAIN = "US_Accidents_Main.csv"
INPUT_DESC = "US_Accidents_Descriptions.csv"
OUTPUT_MAIN = "CA_Accidents_Main.csv"
OUTPUT_DESC = "CA_Accidents_Descriptions.csv"

# Load main data and filter for California
df_main = pd.read_csv(INPUT_MAIN, low_memory=False)
print(f"Original main shape: {df_main.shape}")

df_main_ca = df_main[df_main["State"] == "CA"]
print(f"California main shape: {df_main_ca.shape}")

# Load descriptions and keep only matching IDs
df_desc = pd.read_csv(INPUT_DESC)
ca_ids = set(df_main_ca["ID"])
df_desc_ca = df_desc[df_desc["ID"].isin(ca_ids)]
print(f"California descriptions shape: {df_desc_ca.shape}")

assert len(df_main_ca) == len(df_desc_ca), "Row count mismatch!"

df_main_ca.to_csv(OUTPUT_MAIN, index=False)
df_desc_ca.to_csv(OUTPUT_DESC, index=False)

print(f"California Main: {df_main_ca.shape} -> {OUTPUT_MAIN}")
print(f"California Descriptions: {df_desc_ca.shape} -> {OUTPUT_DESC}")
