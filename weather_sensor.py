import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore

temps = [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23,
         22, 21, 23, 24, 25]
df = pd.DataFrame({"Hour": range(1, 16), "Temperature": temps})
df["Temp_raw"] = df["Temperature"]

# (a) stats before cleaning (on available readings)
before = df["Temperature"].dropna()
print("Stats BEFORE cleaning:")
print(df["Temperature"].describe())
print("Mean:", before.mean(), "Std:", before.std())
print("Skewness:", stats.skew(before), "Kurtosis:", stats.kurtosis(before))

# (b) linear interpolation
df["Temperature"] = df["Temperature"].interpolate(method="linear")
print("After interpolation, missing values:", df["Temperature"].isnull().sum())
mean_i, std_i = df["Temperature"].mean(), df["Temperature"].std()
print("After interpolation (before outlier fix) Mean:", mean_i, "Std:", std_i)

# (c) Z-score outlier (threshold 2) -> replace only flagged value with median
z = zscore(df["Temperature"])
mask = np.abs(z) > 2
median = df["Temperature"].median()
for _, row in df[mask].iterrows():
    print(f"Z-score(>2) outlier detected at Hour: {int(row['Hour'])} (value: {row['Temperature']})")
print("Replaced with column median:", median)
df.loc[mask, "Temperature"] = median
df["Temp_clean"] = df["Temperature"]

# (d) standardize + comparison
df["Temp_Zscore"] = zscore(df["Temp_clean"])
print("Before/After comparison:")
print(df[["Hour", "Temp_raw", "Temp_clean", "Temp_Zscore"]])
print("Stats AFTER cleaning: Mean:", df["Temp_clean"].mean(), "Std:", df["Temp_clean"].std())
