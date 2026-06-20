# ============================================================
# PHARMAGUARD AI — SMILES Fetcher
# Fetches SMILES for all unique drugs in DDInter from PubChem
# Run this ONCE to generate drug_smiles_map.csv
# ============================================================

import pandas as pd
import requests
import time
import os

# ---- Load merged DDInter dataset ----
df = pd.read_csv("datasets/ddinter_merged.csv")

# ---- Get all unique drug names ----
drugs_a = set(df["Drug_A"].unique())
drugs_b = set(df["Drug_B"].unique())
all_drugs = list(drugs_a.union(drugs_b))
print(f"[INFO] Total unique drugs to fetch: {len(all_drugs)}")

# ---- Check if we already have a partial save ----
save_path = "datasets/drug_smiles_map.csv"
if os.path.exists(save_path):
    existing = pd.read_csv(save_path)
    fetched_drugs = set(existing["Drug_Name"].tolist())
    all_drugs = [d for d in all_drugs if d not in fetched_drugs]
    print(f"[INFO] Already fetched: {len(fetched_drugs)} drugs")
    print(f"[INFO] Remaining to fetch: {len(all_drugs)} drugs")
else:
    existing = pd.DataFrame(columns=["Drug_Name", "SMILES"])
    print(f"[INFO] Starting fresh fetch...")

# ---- PubChem fetch function ----
def get_smiles_from_pubchem(drug_name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        else:
            return None
    except Exception as e:
        return None

# ---- Fetch SMILES for all drugs ----
results = []
batch_size = 50  # Save every 50 drugs

for i, drug in enumerate(all_drugs):
    smiles = get_smiles_from_pubchem(drug)
    results.append({"Drug_Name": drug, "SMILES": smiles})

    status = "[OK]" if smiles else "[MISS]"
    print(f"  {status} ({i+1}/{len(all_drugs)}) {drug}")

    # Save every 50 drugs so you don't lose progress
    if (i + 1) % batch_size == 0:
        batch_df = pd.DataFrame(results)
        combined = pd.concat([existing, batch_df], ignore_index=True)
        combined.to_csv(save_path, index=False)
        existing = combined
        results = []
        print(f"\n[SAVED] Progress saved at {i+1} drugs\n")

    time.sleep(0.3)  # Be polite to PubChem API

# ---- Save any remaining results ----
if results:
    batch_df = pd.DataFrame(results)
    combined = pd.concat([existing, batch_df], ignore_index=True)
    combined.to_csv(save_path, index=False)

# ---- Summary ----
final = pd.read_csv(save_path)
found     = final["SMILES"].notna().sum()
not_found = final["SMILES"].isna().sum()

print(f"\n{'='*50}")
print(f"[DONE] SMILES fetch complete!")
print(f"  Total drugs   : {len(final)}")
print(f"  Found SMILES  : {found}")
print(f"  Not found     : {not_found}")
print(f"  Saved to      : {save_path}")
print(f"{'='*50}")