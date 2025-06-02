import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen, Descriptors
import pandas as pd


# =====================================================
# 1) DEFINISI KOMPONEN MODEL (RBM + DBN – HANYA FORWARD)
# =====================================================

class RBMLayer(nn.Module):
    def __init__(self, n_visible, n_hidden, momentum=0.9):
        super(RBMLayer, self).__init__()
        # Inisialisasi bobot (Kaiming uniform)
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.kaiming_uniform_(self.W, a=0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        # Buffer momentum (hanya diperlukan jika pre-training)
        self.W_mom = torch.zeros_like(self.W)
        self.vb_mom = torch.zeros_like(self.v_bias)
        self.hb_mom = torch.zeros_like(self.h_bias)
        self.momentum = momentum

    def sample_h(self, v):
        """
        - p(h|v) = sigmoid(W·v + b_h)
        - sampling Bernoulli dari p_h
        """
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return torch.bernoulli(p_h), p_h

    def sample_v(self, h):
        """
        - p(v|h) = sigmoid(Wᵀ·h + b_v)
        - sampling Bernoulli dari p_v
        """
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return torch.bernoulli(p_v), p_v

    def sample_hidden(self, v):
        """
        - Hanya return probabilitas tersembunyi tanpa sampling.
        """
        return torch.sigmoid(F.linear(v, self.W, self.h_bias))

    def contrastive_divergence(self, v, lr=1e-4, k=1):
        """
        Stub agar tidak error di inferensi (tidak digunakan di sini).
        """
        return 0.0


class DBN(nn.Module):
    """
    DBN versi inferensi-only:
    - Pretraining RBM dihilangkan, kita hanya keep `forward`
    - Attention + deep MLP sesuai model final
    """
    def __init__(self, n_visible, hidden_sizes=[512, 256, 128]):
        super(DBN, self).__init__()
        # 1) RBM layers (hanya untuk forward; tidak ada pretraining di sini)
        self.rbms = nn.ModuleList([
            RBMLayer(n_visible if i == 0 else hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(len(hidden_sizes))
        ])

        # 2) Attention block ringan:
        self.attn_embed = nn.Linear(1, 16)
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=2)

        # 3) Deep MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] * 16, 512),
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

            nn.Linear(64, 1)  # logit output
        )

    def forward(self, x):
        # 1) Jalankan sampling h di tiap RBM (tanpa pretraining, parameter W sudah dimuat)
        for rbm in self.rbms:
            _, x = rbm.sample_h(x)

        # 2) Attention block
        batch_size = x.size(0)
        hid_dim = x.size(1)  # hidden_sizes[-1]
        x_unsq = x.unsqueeze(2)                    # [B, hid_dim, 1]
        x_emb = self.attn_embed(x_unsq)            # [B, hid_dim, 16]
        attn_in = x_emb.permute(1, 0, 2)            # [hid_dim, B, 16]
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        attn_out = attn_out.permute(1, 0, 2)        # [B, hid_dim, 16]
        attn_flat = attn_out.contiguous().view(batch_size, -1)  # [B, hid_dim*16]

        # 3) Deep MLP classifier
        out = self.classifier(attn_flat)
        return out

    def extract_features(self, x):
        # Hanya untuk inspect latent features; tidak pernah dipakai di main
        for rbm in self.rbms:
            x = rbm.sample_hidden(x)
        return x


# =====================================================
#    2) EKSTRAKSI & SKALING FITUR MOLEKUL (INFERENSI)
# =====================================================

def mol_to_features(smiles, radius=2, nBits=2048):
    """
    Ekstrak 8 fitur kontinyu + 2048-bit Morgan fingerprint => total 2056 fitur.
    """
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

    continuous = np.array([
        logp, tpsa, hba, hbd,
        num_atoms, num_bonds,
        rotatable, weight
    ], dtype=float)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)

    return np.concatenate([continuous, arr])


# =====================================================
#          3) ANTARMUKA PREDIKSI (MAIN SCRIPT)
# =====================================================

if __name__ == "__main__":
    # 3.1) Baca header train.csv untuk daftar feature names
    df_hdr = pd.read_csv("train.csv", nrows=0)
    feature_names = [c for c in df_hdr.columns if c != "Label"]

    # 3.2) Muat train.csv (sebagian) hanya untuk hitung median tiap kontinyu
    df_full = pd.read_csv("train.csv")
    cont_cols = [
        "LogP", "TPSA", "hbond_acceptors", "hbond_donors",
        "num_atoms", "num_bonds", "rotatable_bonds", "weight"
    ]
    for col in cont_cols:
        if col not in df_full.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan di train.csv.")

    medians = {col: df_full[col].median() for col in cont_cols}

    # 3.3) Muat scaler dan model terlatih
    scaler = joblib.load("scaler.pkl")  # scaler yang sama saat preprocessing+SMOTE
    n_visible = len(feature_names)
    model = DBN(n_visible=n_visible, hidden_sizes=[512, 256, 128]).to("cpu")
    # Menggunakan strict=False agar key "criterion.pos_weight" diabaikan
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

            # 3.4) Ekstraksi fitur
            feats_full = mol_to_features(smiles)  # shape = (2056,)
            raw_cont = feats_full[:8].astype(float)
            bin_cont = np.zeros((8,), dtype=int)
            for i, col in enumerate(cont_cols):
                bin_cont[i] = 1 if raw_cont[i] > medians[col] else 0

            fp_part = feats_full[8:].astype(int)  # fingerprint length 2048

            # 3.5) Susun dictionary input sesuai feature_names
            data = {}
            for i, c in enumerate(cont_cols):
                data[c] = bin_cont[i]
            for i in range(2048):
                data[f"FP_{i}"] = int(fp_part[i])

            df_input = pd.DataFrame([data], columns=(cont_cols + [f"FP_{i}" for i in range(2048)]))
            df_sel = df_input[feature_names]  # pastikan urutan sama

            # 3.6) Scaling
            X_scaled = scaler.transform(df_sel.values)

            # 3.7) Inferensi
            with torch.no_grad():
                x_t = torch.tensor(X_scaled, dtype=torch.float32)
                logit = model(x_t)              # output shape: [1,1]
                prob = torch.sigmoid(logit).item()

                
                label = "GENOTOKSIK" if prob > 0.45 else "NON‐GENOTOKSIK"

            print(f"→ Probabilitas genotoksik: {prob:.4f}   |   Prediksi akhir: {label}\n")

        except Exception as err:
            print(f"[!] Error: {err}\n")

    print("Program selesai.")
