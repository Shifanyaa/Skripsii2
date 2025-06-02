import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

print("Membaca file data...")
df = pd.read_csv("Genotoxic_Smiles.csv")

if "SMILES" not in df.columns:
    raise KeyError("Kolom 'SMILES' tidak ditemukan di file.")

df = df[df["SMILES"].notna()]
df["SMILES"] = df["SMILES"].astype(str).str.strip()

def compute_features(smiles):
    if not isinstance(smiles, str) or smiles.strip() == "":
        return [None]*10
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return [
                Descriptors.MolLogP(mol),                         
                Chem.MolToSmiles(mol, canonical=True),            
                rdMolDescriptors.CalcTPSA(mol),                   
                Chem.rdMolDescriptors.CalcMolFormula(mol),        
                rdMolDescriptors.CalcNumLipinskiHBA(mol),         
                rdMolDescriptors.CalcNumLipinskiHBD(mol),         
                mol.GetNumAtoms(),                               
                mol.GetNumBonds(),                               
                Descriptors.NumRotatableBonds(mol),               
                Descriptors.MolWt(mol)                            
            ]
    except Exception as e:
        print(f"Gagal parsing SMILES: {smiles}")
    return [None]*10

print("Menghitung fitur molekul dari SMILES...")
feature_names = [
    "LogP", "Canonical_SMILES", "TPSA", "formula", "hbond_acceptors", 
    "hbond_donors", "num_atoms", "num_bonds", 
    "rotatable_bonds", "weight"
]
df_features = df["SMILES"].apply(compute_features)
df_features = pd.DataFrame(df_features.tolist(), columns=feature_names)

if "Substance" in df.columns and "Genotoxicity" in df.columns:
    df_features = pd.concat([df[["Substance", "Genotoxicity"]].reset_index(drop=True), df_features], axis=1)

output_file = "molecule_list_from_genotoxic.csv"
df_features.dropna(subset=["Canonical_SMILES"], inplace=True)
df_features.to_csv(output_file, index=False)
print(f"Selesai! Fitur disimpan ke file: {output_file}")
