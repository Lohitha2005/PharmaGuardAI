# ============================================================
# PHARMAGUARD AI — Layer 6
# Risk Fusion Engine
# Combines: Toxicity Score + DDI Severity + 
#           Structural Complexity + Patient Vulnerability
# Output: Unified Drug Safety Intelligence Score
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

# Import Layer 5 functions
from layer5_patient_risk import (
    create_patient_profile,
    assess_patient_risk,
    dosage_recommendation
)

# ============================================================
# STEP 1 — Get Toxicity Risk Score from Layer 4 Models
# ============================================================

def get_toxicity_score(smiles: str) -> dict:
    """
    Runs the molecule through all 12 trained toxicity models.
    Returns an overall toxicity risk score (0–1).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"score": 0.0, "label": "Invalid SMILES", "details": {}}

    # Build feature vector (same as Layer 4)
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

    label_cols = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    probs   = []
    details = {}

    for label in label_cols:
        path = f"models/tox_{label.replace('-','_')}_best.pkl"
        if os.path.exists(path):
            model = joblib.load(path)
            prob  = model.predict_proba(X)[0][1]
            probs.append(prob)
            details[label] = round(float(prob), 4)

    # Overall toxicity score = average probability across all labels
    overall = float(np.mean(probs)) if probs else 0.0

    if overall <= 0.2:
        tox_label = "Low Toxicity"
    elif overall <= 0.4:
        tox_label = "Moderate Toxicity"
    elif overall <= 0.6:
        tox_label = "High Toxicity"
    else:
        tox_label = "Very High Toxicity"

    return {
        "score"  : round(overall, 4),
        "label"  : tox_label,
        "details": details
    }


# ============================================================
# STEP 2 — DDI Severity Score
# Since we don't have SMILES for DDInter,
# we use a rule-based severity mapper from drug name pairs
# ============================================================

def get_ddi_severity_score(drug_a: str, drug_b: str) -> dict:
    """
    Looks up DDI severity from the merged DDInter dataset.
    Returns a severity score (0–1).
    """
    severity_map = {
        "Major"    : 1.0,
        "Moderate" : 0.5,
        "Minor"    : 0.2,
        "Unknown"  : 0.1,
    }

    try:
        df = pd.read_csv("datasets/ddinter_merged.csv")

        # Search for drug pair in either direction
        match = df[
            ((df["Drug_A"].str.lower() == drug_a.lower()) &
             (df["Drug_B"].str.lower() == drug_b.lower())) |
            ((df["Drug_A"].str.lower() == drug_b.lower()) &
             (df["Drug_B"].str.lower() == drug_a.lower()))
        ]

        if len(match) > 0:
            level = match.iloc[0]["Level"]
            score = severity_map.get(level, 0.1)
            return {
                "score"    : score,
                "level"    : level,
                "found"    : True,
                "drug_a"   : drug_a,
                "drug_b"   : drug_b
            }
        else:
            return {
                "score"    : 0.0,
                "level"    : "No Interaction Found",
                "found"    : False,
                "drug_a"   : drug_a,
                "drug_b"   : drug_b
            }
    except Exception as e:
        return {
            "score"  : 0.0,
            "level"  : "Dataset Error",
            "found"  : False,
            "error"  : str(e)
        }


# ============================================================
# STEP 3 — Structural Complexity Score
# Complex molecules are harder to metabolize
# ============================================================

def get_structural_complexity_score(smiles: str) -> dict:
    """
    Calculates structural complexity risk from molecular features.
    Complex molecules → harder to metabolize → higher risk.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"score": 0.0, "label": "Invalid"}

    # Bertz complexity normalized (typical range 0–1500)
    bertz     = Descriptors.BertzCT(mol)
    bertz_norm= min(bertz / 1500.0, 1.0)

    # Rotatable bonds (flexibility → affects absorption)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rot_norm  = min(rot_bonds / 15.0, 1.0)

    # Ring count
    rings     = rdMolDescriptors.CalcNumRings(mol)
    ring_norm = min(rings / 8.0, 1.0)

    # Combined structural complexity score
    score = (0.5 * bertz_norm) + (0.3 * rot_norm) + (0.2 * ring_norm)
    score = round(score, 4)

    if score <= 0.2:
        label = "Simple Structure"
    elif score <= 0.4:
        label = "Moderate Complexity"
    elif score <= 0.6:
        label = "Complex Structure"
    else:
        label = "Highly Complex"

    return {
        "score"      : score,
        "label"      : label,
        "bertz"      : round(bertz, 2),
        "rot_bonds"  : rot_bonds,
        "rings"      : rings
    }


# ============================================================
# STEP 4 — Risk Fusion Engine
# Weighted combination of all risk dimensions
# ============================================================

def fuse_risks(
    toxicity_risk    : dict,
    ddi_risk         : dict,
    structural_risk  : dict,
    patient_vuln     : dict,
    ddi_available    : bool = True
) -> dict:
    """
    Fuses all risk scores into one Unified Drug Safety Score.
    
    Weights (from your architecture):
    - Toxicity           : 35%
    - DDI Severity       : 25%
    - Structural Complexity: 15%
    - Patient Vulnerability: 25%
    """

    # Normalize patient vulnerability to 0-1
    patient_score = patient_vuln["vulnerability"]["vulnerability_score"] / 100.0

    if ddi_available:
        weights = {
            "toxicity"   : 0.35,
            "ddi"        : 0.25,
            "structural" : 0.15,
            "patient"    : 0.25,
        }
        unified = (
            weights["toxicity"]   * toxicity_risk["score"]   +
            weights["ddi"]        * ddi_risk["score"]         +
            weights["structural"] * structural_risk["score"]  +
            weights["patient"]    * patient_score
        )
    else:
        # If no DDI data, redistribute DDI weight
        weights = {
            "toxicity"   : 0.45,
            "structural" : 0.20,
            "patient"    : 0.35,
        }
        unified = (
            weights["toxicity"]   * toxicity_risk["score"]   +
            weights["structural"] * structural_risk["score"]  +
            weights["patient"]    * patient_score
        )

    unified_score = round(unified * 100, 2)

    if unified_score <= 20:
        safety_label = "SAFE"
        color        = "GREEN"
        action       = "Proceed with standard dosing"
    elif unified_score <= 40:
        safety_label = "LOW RISK"
        color        = "LIGHT GREEN"
        action       = "Proceed with monitoring"
    elif unified_score <= 60:
        safety_label = "MODERATE RISK"
        color        = "YELLOW"
        action       = "Dose adjustment recommended"
    elif unified_score <= 80:
        safety_label = "HIGH RISK"
        color        = "ORANGE"
        action       = "Specialist review required"
    else:
        safety_label = "CRITICAL RISK"
        color        = "RED"
        action       = "Do not administer — seek alternative"

    return {
        "unified_score" : unified_score,
        "safety_label"  : safety_label,
        "color"         : color,
        "action"        : action,
        "components"    : {
            "Toxicity Score"        : round(toxicity_risk["score"]  * 100, 2),
            "DDI Severity Score"    : round(ddi_risk["score"]       * 100, 2),
            "Structural Complexity" : round(structural_risk["score"]* 100, 2),
            "Patient Vulnerability" : round(patient_score           * 100, 2),
        }
    }


# ============================================================
# STEP 5 — Full Risk Fusion Report
# ============================================================

def generate_risk_report(
    smiles   : str,
    drug_name: str,
    co_drug  : str,
    patient  : dict
) -> dict:
    """
    Master function — runs full Layer 6 pipeline.
    Input : drug SMILES, drug name, co-administered drug, patient profile
    Output: complete unified risk report
    """

    print("\n" + "=" * 60)
    print(f"PHARMAGUARD AI — Risk Fusion Report")
    print(f"Drug     : {drug_name}")
    print(f"Co-Drug  : {co_drug}")
    print("=" * 60)

    # Get all risk scores
    print("\n[1/4] Calculating Toxicity Risk...")
    tox   = get_toxicity_score(smiles)
    print(f"      Toxicity Score : {tox['score']*100:.1f}% — {tox['label']}")

    print("\n[2/4] Checking Drug-Drug Interaction...")
    ddi   = get_ddi_severity_score(drug_name, co_drug)
    print(f"      DDI Level      : {ddi['level']} — Score: {ddi['score']*100:.1f}%")

    print("\n[3/4] Calculating Structural Complexity...")
    struct= get_structural_complexity_score(smiles)
    print(f"      Complexity     : {struct['label']} — Score: {struct['score']*100:.1f}%")

    print("\n[4/4] Assessing Patient Risk...")
    pat   = assess_patient_risk(patient)
    v_score = pat["vulnerability"]["vulnerability_score"]
    print(f"      Vulnerability  : {pat['vulnerability']['risk_level']} — Score: {v_score:.1f}%")

    # Fuse all risks
    print("\n[FUSING] Combining all risk dimensions...")
    fused = fuse_risks(tox, ddi, struct, pat, ddi_available=ddi["found"])

    # Print unified result
    print("\n" + "=" * 60)
    print("UNIFIED DRUG SAFETY INTELLIGENCE SCORE")
    print("=" * 60)
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Safety Score  : {fused['unified_score']:>6.2f} / 100          │")
    print(f"  │  Status        : {fused['safety_label']:<25}│")
    print(f"  │  Action        : {fused['action']:<25}│")
    print(f"  └─────────────────────────────────────┘")

    print(f"\n  Component Breakdown:")
    for k, v in fused["components"].items():
        bar = "█" * int(v / 5)
        print(f"    {k:<26}: {v:>5.1f}%  {bar}")

    # Dosage recommendation
    dose = dosage_recommendation(fused["unified_score"], drug_name)

    return {
        "drug"            : drug_name,
        "co_drug"         : co_drug,
        "toxicity"        : tox,
        "ddi"             : ddi,
        "structural"      : struct,
        "patient"         : pat,
        "unified"         : fused,
        "dosage"          : dose,
    }


# ============================================================
# MAIN — Test with sample cases
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHARMAGUARD AI — Layer 6: Risk Fusion Engine")
    print("=" * 60)

    # ── Case 1: Aspirin + Warfarin, Elderly patient ──
    print("\n" + "─" * 60)
    print("CASE 1 — Aspirin + Warfarin | Elderly Patient")
    print("─" * 60)
    patient1 = create_patient_profile(
        age=72, weight_kg=65, egfr=28,
        alt=45, ast=38, gender="female",
        conditions=["diabetes", "hypertension"]
    )
    report1 = generate_risk_report(
        smiles    = "CC(=O)Oc1ccccc1C(=O)O",
        drug_name = "Aspirin",
        co_drug   = "Warfarin",
        patient   = patient1
    )

    # ── Case 2: Metformin, Healthy adult ──
    print("\n" + "─" * 60)
    print("CASE 2 — Metformin | Healthy Young Adult")
    print("─" * 60)
    patient2 = create_patient_profile(
        age=30, weight_kg=75, egfr=95,
        alt=28, ast=22, gender="male",
        conditions=[]
    )
    report2 = generate_risk_report(
        smiles    = "CN(C)C(=N)NC(=N)N",
        drug_name = "Metformin",
        co_drug   = "Ibuprofen",
        patient   = patient2
    )

    print("\n" + "=" * 60)
    print("[DONE] Layer 6 complete!")
    print("       Unified Drug Safety Scores generated")
    print("       Next: Layer 7 — Development Readiness Scoring")
    print("=" * 60)