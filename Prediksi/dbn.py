import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen, Descriptors
import pandas as pd
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ==========================
# 1. RBM Layer
# ==========================
class RBMLayer(nn.Module):
    def __init__(self, n_visible, n_hidden, momentum=0.9):
        super(RBMLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.kaiming_uniform_(self.W, a=0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        self.W_mom = torch.zeros_like(self.W)
        self.vb_mom = torch.zeros_like(self.v_bias)
        self.hb_mom = torch.zeros_like(self.h_bias)
        self.momentum = momentum

    def sample_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return torch.bernoulli(p_h), p_h

    def sample_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return torch.bernoulli(p_v), p_v

    def sample_hidden(self, v):
        return torch.sigmoid(F.linear(v, self.W, self.h_bias))

# ==========================
# 2. DBN Model (inference only)
# ==========================
class DBN(nn.Module):
    def __init__(self, n_visible, hidden_sizes=[512, 256, 128]):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([
            RBMLayer(n_visible if i == 0 else hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(len(hidden_sizes))
        ])

        self.token_dim = 16
        self.attn_embed = nn.Linear(1, self.token_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=2,
            dim_feedforward=64,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] * self.token_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        for rbm in self.rbms:
            _, x = rbm.sample_h(x)
        batch_size = x.size(0)
        x_unsq = x.unsqueeze(2)
        x_emb = self.attn_embed(x_unsq)
        x_trans = self.transformer(x_emb)
        x_flat = x_trans.contiguous().view(batch_size, -1)
        out = self.classifier(x_flat)
        return out

# ==========================
# 3. Molecule feature extraction
# ==========================
def mol_to_features(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    weight = Descriptors.MolWt(mol)
    continuous = np.array([logp, tpsa, hba, hbd, num_atoms, num_bonds, rotatable, weight], dtype=float)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return np.concatenate([continuous, arr])

# ==========================
# 4. Main Application
# ==========================
if __name__ == "__main__":
    df_hdr = pd.read_csv("train.csv", nrows=0)
    feature_names = [c for c in df_hdr.columns if c != "Label"]

    df_full = pd.read_csv("train.csv")
    cont_cols = ["LogP", "TPSA", "hbond_acceptors", "hbond_donors",
                 "num_atoms", "num_bonds", "rotatable_bonds", "weight"]
    medians = {col: df_full[col].median() for col in cont_cols}

    scaler = joblib.load("scaler.pkl")
    n_visible = len(feature_names)
    model = DBN(n_visible=n_visible, hidden_sizes=[512, 256, 128]).to("cpu")
    model.load_state_dict(torch.load("dbn_final.pth", map_location="cpu"), strict=False)
    model.eval()

    print("=== Prediksi Genotoksisitas menggunakan DBN Terbaru ===")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        nama = input("Masukkan nama senyawa: ").strip()
        if nama.lower() == "exit":
            break
        try:
            hasil = pcp.get_compounds(nama, 'name')
            if not hasil:
                print(f"[!] Senyawa “{nama}” tidak ditemukan di PubChem.\n")
                continue
            smiles = hasil[0].canonical_smiles
            print(f"→ SMILES untuk {nama}: {smiles}")
            feats_full = mol_to_features(smiles)
            raw_cont = feats_full[:8].astype(float)
            bin_cont = np.zeros((8,), dtype=int)
            for i, col in enumerate(cont_cols):
                bin_cont[i] = 1 if raw_cont[i] > medians[col] else 0
            fp_part = feats_full[8:].astype(int)
            data = {c: bin_cont[i] for i, c in enumerate(cont_cols)}
            data.update({f"FP_{i}": int(fp_part[i]) for i in range(2048)})
            df_input = pd.DataFrame([data], columns=feature_names)
            X_scaled = scaler.transform(df_input.values)

            with torch.no_grad():
                x_t = torch.tensor(X_scaled, dtype=torch.float32)
                logit = model(x_t)
                prob = torch.sigmoid(logit).item()
                label = "GENOTOKSIK" if prob > 0.45 else "NON‐GENOTOKSIK"

            print(f"→ Probabilitas genotoksik: {prob:.4f}   |   Prediksi akhir: {label}\n")

        except Exception as err:
            print(f"[!] Error: {err}\n")

    print("Program selesai.")