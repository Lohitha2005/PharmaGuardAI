# ============================================================
# PHARMAGUARD AI — Layer 1 & 2
# Data Input + Molecular Preprocessing Pipeline
# ============================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, AllChem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LAYER 1 — INPUT LAYER
# Accepts single drug SMILES or drug pair (Drug A + Drug B)
# ============================================================

def load_single_drug(smiles: str):
    """
    Input: SMILES string of a single drug molecule
    Output: RDKit Mol object or None if invalid
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        print(f"[ERROR] Invalid SMILES: {smiles}")
        return None
    print(f"[OK] Valid SMILES loaded: {smiles}")
    return mol


def load_drug_pair(smiles_a: str, smiles_b: str):
    """
    Input: Two SMILES strings (Drug A and Drug B)
    Output: Tuple of two RDKit Mol objects for DDI analysis
    """
    mol_a = load_single_drug(smiles_a)
    mol_b = load_single_drug(smiles_b)
    if mol_a is None or mol_b is None:
        print("[ERROR] One or both SMILES are invalid.")
        return None, None
    print(f"[OK] Drug pair loaded successfully.")
    return mol_a, mol_b


def load_tox21_dataset(filepath: str):
    """
    Load Tox21 dataset from CSV file
    Expected columns: smiles + toxicity labels
    """
    df = pd.read_csv(filepath)
    print(f"[OK] Tox21 dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"     Columns: {list(df.columns)}")
    return df


def load_ddinter_dataset(filepath: str):
    """
    Load DDInter dataset from CSV file
    Expected columns: Drug_A_SMILES, Drug_B_SMILES, Interaction_Severity
    """
    df = pd.read_csv(filepath)
    print(f"[OK] DDInter dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"     Columns: {list(df.columns)}")
    return df


# ============================================================
# LAYER 2 — MOLECULAR PREPROCESSING LAYER
# RDKit Parsing → Descriptor Extraction → Fingerprint →
# Molecular Graph → Feature Standardization
# ============================================================

# ---- Step 2A: Descriptor Extraction (Descriptor Intelligence View) ----

def extract_descriptors(mol):
    """
    Extracts physicochemical descriptors from a molecule.
    These form the DESCRIPTOR INTELLIGENCE VIEW.
    Returns a dictionary of descriptor name → value
    """
    if mol is None:
        return None

    descriptors = {
        "MolecularWeight"     : Descriptors.MolWt(mol),
        "LogP"                : Descriptors.MolLogP(mol),
        "TPSA"                : Descriptors.TPSA(mol),
        "HBondAcceptors"      : rdMolDescriptors.CalcNumHBA(mol),
        "HBondDonors"         : rdMolDescriptors.CalcNumHBD(mol),
        "RotatableBonds"      : rdMolDescriptors.CalcNumRotatableBonds(mol),
        "AromaticRings"       : rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtomCount"      : mol.GetNumHeavyAtoms(),
        "RingCount"           : rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3"        : rdMolDescriptors.CalcFractionCSP3(mol),
    }
    return descriptors


def extract_descriptors_from_smiles(smiles: str):
    """
    Wrapper: extract descriptors directly from SMILES string
    """
    mol = MolFromSmiles(smiles)
    return extract_descriptors(mol)


# ---- Step 2B: Morgan Fingerprint Generation ----

def generate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Generates Morgan (ECFP) circular fingerprints.
    These capture local chemical environments around each atom.
    Returns a numpy array of bit vector (0s and 1s)
    """
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def generate_fingerprint_from_smiles(smiles: str, radius=2, n_bits=2048):
    """
    Wrapper: generate fingerprint directly from SMILES string
    """
    mol = MolFromSmiles(smiles)
    return generate_morgan_fingerprint(mol, radius, n_bits)


# ---- Step 2C: Molecular Graph Construction (Structural Intelligence View) ----

def extract_graph_features(mol):
    """
    Extracts structural/graph-level features from the molecule.
    These form the STRUCTURAL INTELLIGENCE VIEW.
    Includes ring structures, functional groups, connectivity patterns.
    """
    if mol is None:
        return None

    graph_features = {
        "NumAtoms"              : mol.GetNumAtoms(),
        "NumBonds"              : mol.GetNumBonds(),
        "NumAromaticAtoms"      : sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
        "NumSaturatedRings"     : rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumAliphaticRings"     : rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumHeterocycles"       : rdMolDescriptors.CalcNumHeterocycles(mol),
        "NumStereocenters"      : len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "NumAmideBonds"         : rdMolDescriptors.CalcNumAmideBonds(mol),
        "BertzComplexity"       : Descriptors.BertzCT(mol),
        "Chi0"                  : Descriptors.Chi0(mol),
    }
    return graph_features


def extract_graph_features_from_smiles(smiles: str):
    """
    Wrapper: extract graph features directly from SMILES string
    """
    mol = MolFromSmiles(smiles)
    return extract_graph_features(mol)


# ---- Step 2D: Feature Fusion & Standardization ----

def fuse_features(descriptors: dict, graph_features: dict):
    """
    Combines descriptor view + structural view into one feature vector.
    This is the FEATURE FUSION step in your architecture.
    Returns a flat numpy array.
    """
    if descriptors is None or graph_features is None:
        return None
    combined = {**descriptors, **graph_features}
    feature_vector = np.array(list(combined.values()), dtype=float)
    return feature_vector, list(combined.keys())


def standardize_features(feature_matrix: np.ndarray):
    """
    Applies StandardScaler to normalize the feature matrix.
    Call this on your full dataset matrix, not individual samples.
    Returns scaled matrix + fitted scaler (save scaler for inference)
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    print(f"[OK] Features standardized. Shape: {scaled.shape}")
    return scaled, scaler


# ---- Step 2E: Full Preprocessing Pipeline for a Single SMILES ----

def preprocess_single_molecule(smiles: str):
    """
    Full Layer 2 pipeline for ONE molecule.
    Returns: descriptors dict, graph features dict,
             fused feature vector, morgan fingerprint
    """
    print(f"\n--- Preprocessing: {smiles} ---")

    mol = load_single_drug(smiles)
    if mol is None:
        return None

    # Descriptor view
    descriptors = extract_descriptors(mol)
    print(f"[OK] Descriptors extracted: {len(descriptors)} features")

    # Fingerprint
    fingerprint = generate_morgan_fingerprint(mol)
    print(f"[OK] Morgan fingerprint generated: {len(fingerprint)} bits")

    # Graph / Structural view
    graph_features = extract_graph_features(mol)
    print(f"[OK] Graph features extracted: {len(graph_features)} features")

    # Feature fusion
    fused_vector, feature_names = fuse_features(descriptors, graph_features)
    print(f"[OK] Features fused: {len(fused_vector)} total features")

    return {
        "smiles"        : smiles,
        "mol"           : mol,
        "descriptors"   : descriptors,
        "fingerprint"   : fingerprint,
        "graph_features": graph_features,
        "fused_vector"  : fused_vector,
        "feature_names" : feature_names,
    }


# ---- Step 2F: Batch Preprocessing for Full Dataset ----

def preprocess_dataset(df: pd.DataFrame, smiles_col: str):
    """
    Runs full preprocessing on every molecule in a dataset.
    Input:  DataFrame + name of the SMILES column
    Output: descriptor DataFrame, fingerprint matrix, graph feature DataFrame
    """
    print(f"\n[START] Batch preprocessing {len(df)} molecules...")

    desc_list   = []
    fp_list     = []
    graph_list  = []
    valid_idx   = []

    for i, smiles in enumerate(df[smiles_col]):
        mol = MolFromSmiles(str(smiles))
        if mol is None:
            continue

        desc        = extract_descriptors(mol)
        fp          = generate_morgan_fingerprint(mol)
        graph       = extract_graph_features(mol)

        desc_list.append(desc)
        fp_list.append(fp)
        graph_list.append(graph)
        valid_idx.append(i)

    desc_df     = pd.DataFrame(desc_list, index=valid_idx)
    fp_matrix   = np.array(fp_list)
    graph_df    = pd.DataFrame(graph_list, index=valid_idx)

    print(f"[OK] Preprocessed {len(valid_idx)} valid molecules")
    print(f"     Descriptor shape : {desc_df.shape}")
    print(f"     Fingerprint shape: {fp_matrix.shape}")
    print(f"     Graph feat shape : {graph_df.shape}")

    return desc_df, fp_matrix, graph_df, valid_idx


# ============================================================
# MAIN — Test the full pipeline
# ============================================================

if __name__ == "__main__":

    # ---- Test 1: Single molecule preprocessing ----
    print("=" * 60)
    print("TEST 1: Single Molecule Preprocessing")
    print("=" * 60)

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"   # Aspirin
    result = preprocess_single_molecule(aspirin_smiles)

    if result:
        print("\nDescriptors:")
        for k, v in result["descriptors"].items():
            print(f"  {k}: {v:.4f}")

        print("\nGraph Features:")
        for k, v in result["graph_features"].items():
            print(f"  {k}: {v:.4f}")

        print(f"\nFused vector shape: {result['fused_vector'].shape}")
        print(f"Fingerprint bits (first 20): {result['fingerprint'][:20]}")

    # ---- Test 2: Drug pair preprocessing (for DDI) ----
    print("\n" + "=" * 60)
    print("TEST 2: Drug Pair Preprocessing (DDI Input)")
    print("=" * 60)

    drug_a = "CC(=O)Oc1ccccc1C(=O)O"           # Aspirin
    drug_b = "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"  # Testosterone

    mol_a, mol_b = load_drug_pair(drug_a, drug_b)
    if mol_a and mol_b:
        desc_a = extract_descriptors(mol_a)
        desc_b = extract_descriptors(mol_b)
        fp_a   = generate_morgan_fingerprint(mol_a)
        fp_b   = generate_morgan_fingerprint(mol_b)

        # For DDI: concatenate both fingerprints as one input vector
        ddi_input = np.concatenate([fp_a, fp_b])
        print(f"\n[OK] DDI input vector shape: {ddi_input.shape}")
        print(f"     (Drug A fingerprint + Drug B fingerprint concatenated)")

    # ---- Test 3: Load your Tox21 dataset ----
    print("\n" + "=" * 60)
    print("TEST 3: Load Tox21 Dataset")
    print("=" * 60)
    print("Uncomment and update the path below to load your dataset:")
    print("  df_tox = load_tox21_dataset('path/to/tox21.csv')")
    print("  desc_df, fp_matrix, graph_df, idx = preprocess_dataset(df_tox, smiles_col='smiles')")

    # df_tox = load_tox21_dataset("tox21.csv")
    # desc_df, fp_matrix, graph_df, idx = preprocess_dataset(df_tox, smiles_col="smiles")
    # scaled_desc, scaler = standardize_features(desc_df.values)

    print("\n[DONE] Layer 1 & 2 preprocessing complete.")
    print("Next step: Layer 3 - Dual-View Molecular Intelligence")
    import glob

# Merge all DDInter CSV files into one
all_files = glob.glob("datasets/ddinter_downloads_code_*.csv")
df_ddi = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df_ddi.to_csv("datasets/ddinter_merged.csv", index=False)
print(f"Merged DDInter shape: {df_ddi.shape}")
print(df_ddi.head())
import requests

def get_smiles_from_pubchem(drug_name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
        response = requests.get(url, timeout=10)
        data = response.json()
        return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except:
        return None

# Test it
print(get_smiles_from_pubchem("Aspirin"))