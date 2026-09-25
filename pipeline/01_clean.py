import pandas as pd

INPUT_FILE = "US_Accidents_March23.csv"
OUTPUT_MAIN = "US_Accidents_Main.csv"
OUTPUT_DESC = "US_Accidents_Descriptions.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Original shape: {df.shape}")

# Drop End_Lat, End_Lng and rename Start_Lat/Start_Lng
df = df.drop(columns=["End_Lat", "End_Lng"])
df = df.rename(columns={"Start_Lat": "Lat", "Start_Lng": "Lng"})

# Drop Country column (only US)
df = df.drop(columns=["Country"])

# Drop columns with too many nulls
df = df.drop(columns=["Precipitation(in)", "Wind_Chill(F)", "Wind_Speed(mph)"])

# Drop rows with any remaining null values
df = df.dropna()

print(f"Cleaned shape: {df.shape}")

# Split into two files: indicators (without Description) and descriptions
df_main = df.drop(columns=["Description"])
df_desc = df[["ID", "Description"]]

assert len(df_main) == len(df_desc), "Row count mismatch!"

df_main.to_csv(OUTPUT_MAIN, index=False)
df_desc.to_csv(OUTPUT_DESC, index=False)

print(f"Indicators: {df_main.shape} -> {OUTPUT_MAIN}")
print(f"Descriptions: {df_desc.shape} -> {OUTPUT_DESC}")
