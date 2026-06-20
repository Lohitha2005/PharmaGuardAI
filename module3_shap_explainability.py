# ============================================================
# PHARMAGUARD AI — Module 3
# SHAP Explainable AI Engine
# Feature Attribution + Molecular Hotspots + Decision Explanations
# ============================================================

import numpy as np
import pandas as pd
import shap
import joblib
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D

# ============================================================
# STEP 1 — Feature Names (must match Layer 3 & 4)
# ============================================================

DESCRIPTOR_NAMES = [
    "MolecularWeight", "LogP", "TPSA",
    "HBondAcceptors", "HBondDonors", "RotatableBonds",
    "AromaticRings", "HeavyAtomCount", "RingCount", "FractionCSP3",
    "NumAtoms", "NumBonds", "NumAromaticAtoms",
    "NumSaturatedRings", "NumAliphaticRings", "NumHeterocycles",
    "NumStereocenters", "NumAmideBonds", "BertzComplexity", "Chi0"
]

TOXICITY_LABELS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

LABEL_DESCRIPTIONS = {
    "NR-AR"        : "Androgen Receptor",
    "NR-AR-LBD"    : "Androgen Receptor Ligand Binding",
    "NR-AhR"       : "Aryl Hydrocarbon Receptor",
    "NR-Aromatase" : "Aromatase Enzyme",
    "NR-ER"        : "Estrogen Receptor",
    "NR-ER-LBD"    : "Estrogen Receptor Ligand Binding",
    "NR-PPAR-gamma": "PPAR Gamma Receptor",
    "SR-ARE"       : "Antioxidant Response Element",
    "SR-ATAD5"     : "DNA Damage (ATAD5)",
    "SR-HSE"       : "Heat Shock Response",
    "SR-MMP"       : "Mitochondrial Membrane",
    "SR-p53"       : "DNA Damage (p53 pathway)"
}

# ============================================================
# STEP 2 — Build Feature Vector from SMILES
# ============================================================

def build_feature_vector(smiles: str):
    """
    Builds the same feature vector used in Layer 4 training.
    Returns feature vector + feature names (descriptors only, no fingerprint)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        Descriptors.BertzCT(mol),
        Descriptors.Chi0(mol),
    ])

    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    X  = np.hstack([desc, fp]).reshape(1, -1)

    return X, desc, mol


# ============================================================
# STEP 3 — SHAP Feature Attribution
# Explains which molecular features drive toxicity predictions
# ============================================================

def explain_with_shap(smiles: str, label: str = "NR-AhR"):
    """
    Runs SHAP on a single molecule for one toxicity label.
    Returns SHAP values for the 20 descriptor features.
    """
    print(f"\n[SHAP] Explaining prediction for: {smiles}")
    print(f"       Toxicity label: {label} ({LABEL_DESCRIPTIONS.get(label,'')})")

    # Load model
    model_path = f"models/tox_{label.replace('-','_')}_best.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    model = joblib.load(model_path)

    # Build feature vector
    X, desc_vals, mol = build_feature_vector(smiles)
    if X is None:
        print("[ERROR] Invalid SMILES")
        return None

    # Load background data for SHAP
    fused = np.load("datasets/fused_features.npy")
    fp    = np.load("datasets/fingerprints.npy")
    X_bg  = np.hstack([fused, fp])

    # Use small background sample for speed
    bg_sample = X_bg[np.random.choice(X_bg.shape[0], 100, replace=False)]

    print("[SHAP] Computing SHAP values (this may take a moment)...")

    # TreeExplainer works for RF and XGBoost
    explainer   = shap.TreeExplainer(model, bg_sample)
    shap_values = explainer.shap_values(X)

    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # class 1 (toxic)
    else:
        sv = shap_values[0]

    # Focus on descriptor features only (first 20)
    sv_desc   = sv[:20]
    desc_dict = dict(zip(DESCRIPTOR_NAMES, sv_desc))

    print(f"\n[SHAP] Feature Attribution for {label}:")
    print(f"       (Positive = pushes toward TOXIC, Negative = pushes toward SAFE)")
    print(f"\n  {'Feature':<22} {'SHAP Value':>12} {'Direction'}")
    print(f"  {'─'*50}")

    sorted_desc = sorted(desc_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in sorted_desc:
        direction = "→ TOXIC ↑" if val > 0 else "→ SAFE  ↓"
        bar = "█" * min(int(abs(val) * 200), 20)
        print(f"  {feat:<22} {val:>+12.6f}  {direction}  {bar}")

    return {
        "label"       : label,
        "smiles"      : smiles,
        "shap_values" : sv_desc,
        "feature_names": DESCRIPTOR_NAMES,
        "desc_dict"   : desc_dict,
        "sorted"      : sorted_desc,
        "model"       : model,
        "explainer"   : explainer,
        "X"           : X,
        "shap_all"    : sv
    }


# ============================================================
# STEP 4 — Save SHAP Bar Plot
# ============================================================

def save_shap_bar_plot(shap_result: dict, drug_name: str = "Drug"):
    """
    Saves a horizontal bar chart of SHAP values for descriptor features.
    """
    os.makedirs("shap_plots", exist_ok=True)

    sorted_items = shap_result["sorted"]
    features     = [x[0] for x in sorted_items[:10]]   # top 10
    values       = [x[1] for x in sorted_items[:10]]
    colors       = ["#ff5c5c" if v > 0 else "#00c9ff" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#151827")

    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1],
                   edgecolor="none", height=0.6)

    ax.axvline(0, color="#ffffff30", linewidth=1)
    ax.set_xlabel("SHAP Value (impact on toxicity prediction)",
                  color="#8892b0", fontsize=10)
    ax.set_title(
        f"SHAP Feature Attribution — {drug_name} | {shap_result['label']}",
        color="#e0e0e0", fontsize=12, pad=12
    )
    ax.tick_params(colors="#8892b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f45")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#ff5c5c", label="Pushes toward TOXIC"),
        Patch(color="#00c9ff", label="Pushes toward SAFE"),
    ]
    ax.legend(handles=legend, facecolor="#1e2130",
              labelcolor="#e0e0e0", fontsize=9)

    plt.tight_layout()
    path = f"shap_plots/{drug_name}_{shap_result['label']}_shap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[SAVED] SHAP bar plot → {path}")
    return path


# ============================================================
# STEP 5 — Decision Explanation (Natural Language)
# ============================================================

def generate_decision_explanation(shap_result: dict, drug_name: str) -> str:
    """
    Generates a natural language explanation of the SHAP results.
    This is the 'Decision Explanations' block in your architecture.
    """
    label       = shap_result["label"]
    sorted_feat = shap_result["sorted"]
    desc_dict   = shap_result["desc_dict"]

    # Top 3 features driving toxicity
    top_toxic = [(f,v) for f,v in sorted_feat if v > 0][:3]
    top_safe  = [(f,v) for f,v in sorted_feat if v < 0][:3]

    explanation = []
    explanation.append(f"SHAP Explanation for {drug_name} → {label} "
                      f"({LABEL_DESCRIPTIONS.get(label,'')})")
    explanation.append("─" * 55)

    if top_toxic:
        explanation.append("\nFeatures INCREASING toxicity risk:")
        for feat, val in top_toxic:
            val_actual = dict(zip(DESCRIPTOR_NAMES,
                [0]*20))[feat] if feat not in desc_dict else desc_dict[feat]
            explanation.append(f"  • {feat}: SHAP={val:+.4f}")

    if top_safe:
        explanation.append("\nFeatures DECREASING toxicity risk:")
        for feat, val in top_safe:
            explanation.append(f"  • {feat}: SHAP={val:+.4f}")

    # Molecular property interpretations
    mw  = desc_dict.get("MolecularWeight", 0)
    logp= desc_dict.get("LogP", 0)
    tpsa= desc_dict.get("TPSA", 0)

    explanation.append("\nMolecular Interpretation:")
    if abs(mw) > 0.001:
        direction = "increases" if mw > 0 else "decreases"
        explanation.append(f"  • Molecular weight {direction} {label} risk")
    if abs(logp) > 0.001:
        direction = "increases" if logp > 0 else "decreases"
        explanation.append(f"  • Lipophilicity (LogP) {direction} {label} risk")
    if abs(tpsa) > 0.001:
        direction = "increases" if tpsa > 0 else "decreases"
        explanation.append(f"  • Polar surface area (TPSA) {direction} {label} risk")

    full_explanation = "\n".join(explanation)
    print("\n" + full_explanation)
    return full_explanation


# ============================================================
# STEP 6 — Full SHAP Analysis for All Labels
# ============================================================

def full_shap_analysis(smiles: str, drug_name: str):
    """
    Runs SHAP explanation for all 12 toxicity labels.
    Saves plots and prints explanations.
    """
    print("\n" + "=" * 60)
    print(f"SHAP Full Analysis — {drug_name}")
    print("=" * 60)

    results = {}
    for label in TOXICITY_LABELS:
        print(f"\n{'─'*50}")
        result = explain_with_shap(smiles, label)
        if result:
            save_shap_bar_plot(result, drug_name)
            explanation = generate_decision_explanation(result, drug_name)
            results[label] = {
                "shap_result" : result,
                "explanation" : explanation
            }

    print("\n" + "=" * 60)
    print(f"[DONE] SHAP analysis complete for {drug_name}")
    print(f"       Plots saved in: shap_plots/")
    print("=" * 60)
    return results


# ============================================================
# STEP 7 — Quick Single Label SHAP (for Dashboard integration)
# ============================================================

def quick_shap_explain(smiles: str, drug_name: str, top_n: int = 5):
    """
    Fast SHAP explanation for dashboard — returns top N features.
    Uses SR-ARE label (most commonly flagged in Tox21).
    """
    label  = "SR-ARE"
    result = explain_with_shap(smiles, label)

    if result is None:
        return []

    save_shap_bar_plot(result, drug_name)
    explanation = generate_decision_explanation(result, drug_name)

    # Return top N features for dashboard display
    top_features = [
        {
            "feature"  : feat,
            "shap_val" : round(float(val), 6),
            "direction": "TOXIC" if val > 0 else "SAFE",
            "impact"   : round(abs(float(val)) * 1000, 2)
        }
        for feat, val in result["sorted"][:top_n]
    ]

    return top_features


# ============================================================
# MAIN — Test SHAP on sample molecules
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHARMAGUARD AI — Module 3: SHAP Explainable AI Engine")
    print("=" * 60)

    # ── Test 1: Aspirin ──
    print("\n" + "─" * 60)
    print("TEST 1 — Aspirin (Quick SHAP on SR-ARE)")
    print("─" * 60)
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    result1 = explain_with_shap(aspirin_smiles, "SR-ARE")
    if result1:
        save_shap_bar_plot(result1, "Aspirin")
        generate_decision_explanation(result1, "Aspirin")

    # ── Test 2: Paracetamol ──
    print("\n" + "─" * 60)
    print("TEST 2 — Paracetamol (Quick SHAP on NR-AhR)")
    print("─" * 60)
    paracetamol_smiles = "CC(=O)Nc1ccc(O)cc1"
    result2 = explain_with_shap(paracetamol_smiles, "NR-AhR")
    if result2:
        save_shap_bar_plot(result2, "Paracetamol")
        generate_decision_explanation(result2, "Paracetamol")

    # ── Test 3: Quick dashboard-style explanation ──
    print("\n" + "─" * 60)
    print("TEST 3 — Quick SHAP (Dashboard Style) — Aspirin")
    print("─" * 60)
    top_features = quick_shap_explain(aspirin_smiles, "Aspirin", top_n=5)
    print("\nTop 5 features for dashboard:")
    for f in top_features:
        print(f"  {f['feature']:<22} SHAP={f['shap_val']:+.6f}  "
              f"Direction={f['direction']}  Impact={f['impact']}")

    print("\n" + "=" * 60)
    print("[DONE] Module 3 - SHAP Explainability complete!")
    print("       Plots saved in: shap_plots/")
    print("       Next: Integrate SHAP into Layer 8 Dashboard")
    print("=" * 60)
    