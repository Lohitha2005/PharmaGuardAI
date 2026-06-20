# ============================================================
# PHARMAGUARD AI — Layer 3
# Dual-View Molecular Intelligence
# Descriptor View + Structural View → Feature Fusion
# Dataset: Tox21 (7831 molecules, 12 toxicity labels)
# ============================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1 — Load and Clean Tox21 Dataset
# ============================================================

def load_tox21(filepath="datasets/tox21.csv"):
    df = pd.read_csv(filepath)
    print(f"[OK] Tox21 loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop rows where SMILES is missing
    df = df.dropna(subset=["smiles"])
    print(f"[OK] After dropping missing SMILES: {df.shape[0]} rows")

    # Toxicity label columns
    label_cols = [c for c in df.columns if c not in ["smiles", "mol_id"]]
    print(f"[OK] Toxicity labels ({len(label_cols)}): {label_cols}")

    return df, label_cols


# ============================================================
# STEP 2 — Descriptor Intelligence View
# Physicochemical descriptors for every molecule
# ============================================================

def extract_descriptors(mol):
    if mol is None:
        return None
    try:
        return {
            "MolecularWeight"  : Descriptors.MolWt(mol),
            "LogP"             : Descriptors.MolLogP(mol),
            "TPSA"             : Descriptors.TPSA(mol),
            "HBondAcceptors"   : rdMolDescriptors.CalcNumHBA(mol),
            "HBondDonors"      : rdMolDescriptors.CalcNumHBD(mol),
            "RotatableBonds"   : rdMolDescriptors.CalcNumRotatableBonds(mol),
            "AromaticRings"    : rdMolDescriptors.CalcNumAromaticRings(mol),
            "HeavyAtomCount"   : mol.GetNumHeavyAtoms(),
            "RingCount"        : rdMolDescriptors.CalcNumRings(mol),
            "FractionCSP3"     : rdMolDescriptors.CalcFractionCSP3(mol),
        }
    except:
        return None


def build_descriptor_view(df):
    print("\n[Layer 3] Building Descriptor Intelligence View...")
    desc_list  = []
    valid_idx  = []

    for i, row in df.iterrows():
        mol  = Chem.MolFromSmiles(str(row["smiles"]))
        desc = extract_descriptors(mol)
        if desc:
            desc_list.append(desc)
            valid_idx.append(i)

    desc_df = pd.DataFrame(desc_list, index=valid_idx)
    print(f"[OK] Descriptor view shape: {desc_df.shape}")
    return desc_df, valid_idx


# ============================================================
# STEP 3 — Structural Intelligence View
# Morgan Fingerprints (2048 bits) for every molecule
# ============================================================

def generate_fingerprint(mol, radius=2, n_bits=2048):
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


def build_structural_view(df, valid_idx):
    print("\n[Layer 3] Building Structural Intelligence View...")
    fp_list = []

    for i in valid_idx:
        mol = Chem.MolFromSmiles(str(df.loc[i, "smiles"]))
        fp  = generate_fingerprint(mol)
        if fp is not None:
            fp_list.append(fp)
        else:
            fp_list.append(np.zeros(2048))

    fp_matrix = np.array(fp_list)
    print(f"[OK] Structural view (fingerprint) shape: {fp_matrix.shape}")
    return fp_matrix


# ============================================================
# STEP 4 — Graph Feature View
# Structural graph-level features for every molecule
# ============================================================

def extract_graph_features(mol):
    if mol is None:
        return None
    try:
        return {
            "NumAtoms"         : mol.GetNumAtoms(),
            "NumBonds"         : mol.GetNumBonds(),
            "NumAromaticAtoms" : sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
            "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "NumHeterocycles"  : rdMolDescriptors.CalcNumHeterocycles(mol),
            "NumStereocenters" : len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "NumAmideBonds"    : rdMolDescriptors.CalcNumAmideBonds(mol),
            "BertzComplexity"  : Descriptors.BertzCT(mol),
            "Chi0"             : Descriptors.Chi0(mol),
        }
    except:
        return None


def build_graph_view(df, valid_idx):
    print("\n[Layer 3] Building Graph Feature View...")
    graph_list = []

    for i in valid_idx:
        mol   = Chem.MolFromSmiles(str(df.loc[i, "smiles"]))
        graph = extract_graph_features(mol)
        if graph:
            graph_list.append(graph)
        else:
            graph_list.append({k: 0 for k in [
                "NumAtoms", "NumBonds", "NumAromaticAtoms",
                "NumSaturatedRings", "NumAliphaticRings",
                "NumHeterocycles", "NumStereocenters",
                "NumAmideBonds", "BertzComplexity", "Chi0"
            ]})

    graph_df = pd.DataFrame(graph_list, index=valid_idx)
    print(f"[OK] Graph feature view shape: {graph_df.shape}")
    return graph_df


# ============================================================
# STEP 5 — Feature Fusion
# Combine Descriptor + Graph views → one fused feature matrix
# Fingerprint kept separate (used directly in deep models)
# ============================================================

def fuse_features(desc_df, graph_df):
    print("\n[Layer 3] Fusing Descriptor + Graph views...")
    fused_df = pd.concat([desc_df, graph_df], axis=1)
    print(f"[OK] Fused feature matrix shape: {fused_df.shape}")
    return fused_df


# ============================================================
# STEP 6 — Feature Standardization
# Scale fused features using StandardScaler
# ============================================================

def standardize(fused_df):
    print("\n[Layer 3] Standardizing fused features...")
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(fused_df.values)
    print(f"[OK] Standardized shape: {scaled.shape}")
    return scaled, scaler


# ============================================================
# STEP 7 — Prepare Labels
# Handle NaN in toxicity labels using majority class imputation
# ============================================================

def prepare_labels(df, label_cols, valid_idx):
    print("\n[Layer 3] Preparing toxicity labels...")
    labels_df = df.loc[valid_idx, label_cols].copy()

    # Fill NaN with 0 (assume non-toxic if unknown — common in Tox21)
    labels_df = labels_df.fillna(0)
    labels_df = labels_df.astype(int)

    print(f"[OK] Labels shape: {labels_df.shape}")
    print(f"\n     Toxicity distribution (1=toxic, 0=non-toxic):")
    for col in label_cols:
        toxic     = labels_df[col].sum()
        non_toxic = len(labels_df) - toxic
        print(f"     {col:20s} → Toxic: {toxic:4d} | Non-toxic: {non_toxic:4d}")

    return labels_df


# ============================================================
# STEP 8 — Save Processed Data for Layer 4
# ============================================================

def save_processed_data(fused_scaled, fp_matrix, labels_df, valid_idx):
    print("\n[Layer 3] Saving processed data for Layer 4...")

    np.save("datasets/fused_features.npy",  fused_scaled)
    np.save("datasets/fingerprints.npy",     fp_matrix)
    labels_df.to_csv("datasets/tox21_labels.csv", index=True)

    print(f"[OK] Saved fused_features.npy   → shape {fused_scaled.shape}")
    print(f"[OK] Saved fingerprints.npy     → shape {fp_matrix.shape}")
    print(f"[OK] Saved tox21_labels.csv     → shape {labels_df.shape}")


# ============================================================
# MAIN — Run Full Layer 3 Pipeline
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHARMAGUARD AI — Layer 3: Dual-View Molecular Intelligence")
    print("=" * 60)

    # Step 1: Load data
    df, label_cols = load_tox21("datasets/tox21.csv")

    # Step 2: Descriptor view
    desc_df, valid_idx = build_descriptor_view(df)

    # Step 3: Structural view (fingerprints)
    fp_matrix = build_structural_view(df, valid_idx)

    # Step 4: Graph feature view
    graph_df = build_graph_view(df, valid_idx)

    # Step 5: Feature fusion
    fused_df = fuse_features(desc_df, graph_df)

    # Step 6: Standardize
    fused_scaled, scaler = standardize(fused_df)

    # Step 7: Prepare labels
    labels_df = prepare_labels(df, label_cols, valid_idx)

    # Step 8: Save for Layer 4
    save_processed_data(fused_scaled, fp_matrix, labels_df, valid_idx)

    print("\n" + "=" * 60)
    print("[DONE] Layer 3 complete!")
    print("       Fused features + Fingerprints + Labels ready")
    print("       Next: Layer 4 — Toxicity Prediction Model")
    print("=" * 60)