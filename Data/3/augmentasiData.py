import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from rdkit import DataStructs

# 1. Load raw data
df = pd.read_csv("molecule_list_from_genotoxic.csv")

# 2. Drop duplicate molecules
df = df.drop_duplicates(subset=["Canonical_SMILES"], keep="first")

# 3. Generate molecular descriptors and fingerprints
def mol_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Compute fingerprint
fps = df["Canonical_SMILES"].apply(lambda smi: mol_to_fingerprint(smi))
fp_array = np.array(list(fps.values))
fp_cols = [f"FP_{i}" for i in range(fp_array.shape[1])]
fp_df = pd.DataFrame(fp_array, columns=fp_cols, index=df.index)
df = pd.concat([df, fp_df], axis=1)

# 4. Remove fingerprint bits active in <1% of samples
min_samples = len(df) * 0.01
mask = (df[fp_cols].sum(axis=0) >= min_samples)
rare_bits = [col for col, keep in zip(fp_cols, mask) if not keep]
df = df.drop(columns=rare_bits)
remaining_fp_cols = [col for col in fp_cols if col not in rare_bits]

# 5. Winsorize 
cont_cols = ["LogP", "TPSA", "hbond_acceptors", "hbond_donors",
             "num_atoms", "num_bonds", "rotatable_bonds", "weight"]
for col in cont_cols:
    col_data = df[col].fillna(df[col].median()).values
    df[col] = winsorize(col_data, limits=[0.05, 0.05])

# 6. Binarize 
binarizer = Binarizer(threshold=0.0, copy=True)
for col in cont_cols:
    threshold = df[col].median()
    df[col] = (df[col] > threshold).astype(int)

# 7. Labeling
df["Label"] = df["Genotoxicity"].map({"Negative": 0, "Positive": 1})
feature_cols = cont_cols + remaining_fp_cols  # combined feature list
X = df[feature_cols].astype(float)
y = df["Label"].astype(int)

# 9. Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# 10. Log label distribution in each split
def log_distribution(name, y_split):
    counts = y_split.value_counts()
    print(f"{name} set - Positive: {counts.get(1, 0)}, Negative: {counts.get(0, 0)}, " +
          f"Total: {len(y_split)}")
log_distribution("Train", y_train)
log_distribution("Validation", y_val)
log_distribution("Test", y_test)

# 11. Oversampling menggunakan Stable SMOTE saja
smote = SMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 12. Normalize 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 13. Save processed data to CSV (features + Label)
train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
train_df["Label"] = y_train_res.reset_index(drop=True)
val_df   = pd.DataFrame(X_val_scaled,   columns=feature_cols)
val_df["Label"]   = y_val.reset_index(drop=True)
test_df  = pd.DataFrame(X_test_scaled,  columns=feature_cols)
test_df["Label"]  = y_test.reset_index(drop=True)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Preprocessing complete: train.csv, val.csv, test.csv saved.")
