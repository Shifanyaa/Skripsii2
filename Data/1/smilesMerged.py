import pandas as pd
import requests
import time

def name_to_smiles(name):
    print(f"Mencari SMILES untuk: {name}")
    
    # konversi menggunakan OPSIN
    opsin_url = f"https://opsin.ch.cam.ac.uk/opsin/{name}.json"
    try:
        opsin_response = requests.get(opsin_url, timeout=5)
        if opsin_response.status_code == 200:
            opsin_data = opsin_response.json()
            if "smiles" in opsin_data:
                print(f"Ditemukan melalui OPSIN: {opsin_data['smiles']}")
                return opsin_data["smiles"]
    except requests.exceptions.RequestException:
        pass  # Jika OPSIN gagal, lanjut ke PubChem
    
    # Jika OPSIN gagal, coba gunakan PubChem API
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
    try:
        pubchem_response = requests.get(pubchem_url, timeout=5)
        if pubchem_response.status_code == 200:
            pubchem_data = pubchem_response.json()
            smiles = pubchem_data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
            print(f"Ditemukan melalui PubChem: {smiles}")
            return smiles
    except requests.exceptions.RequestException:
        print(f"Gagal menemukan SMILES untuk {name}")
        return None
    except (KeyError, IndexError):
        print(f"Format tidak ditemukan di PubChem untuk {name}")
        return None


df_genotoxic = pd.read_excel("cleaned_genotoxic_data.xlsx")
df_genotoxic["SMILES"] = df_genotoxic["Substance"].apply(name_to_smiles)
df_genotoxic.to_csv("genotoxic_with_smiles1.csv", index=False)
print("Konversi selesai! Cek file genotoxic_with_smiles.csv")
