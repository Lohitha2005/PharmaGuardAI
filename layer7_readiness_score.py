# ============================================================
# PHARMAGUARD AI — Layer 7
# Development Readiness Scoring Engine
# Generates a Readiness Score (0–100) for drug safety
# Categories: Ready / Promising / Needs Optimization / High Risk
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from layer5_patient_risk import create_patient_profile
from layer6_risk_fusion  import generate_risk_report

# ============================================================
# STEP 1 — Readiness Score Calculator
# Converts unified risk score → readiness score
# Risk and Readiness are inversely related
# High Risk → Low Readiness
# ============================================================

def calculate_readiness_score(unified_score: float) -> dict:
    """
    Converts unified risk score (0–100) to readiness score (0–100).
    Readiness = inverse of risk with clinical adjustments.
    """
    # Base readiness is inverse of risk
    base_readiness = 100 - unified_score

    # Apply clinical penalty bands
    if unified_score <= 20:
        readiness    = base_readiness
        category     = "Ready"
        band_color   = "GREEN"
        band_range   = "80–100"
        description  = "Drug is safe for this patient. Proceed with standard protocol."
    elif unified_score <= 40:
        readiness    = base_readiness - 5   # slight penalty
        category     = "Promising"
        band_color   = "BLUE"
        band_range   = "60–79"
        description  = "Drug shows promise. Minor adjustments needed before administration."
    elif unified_score <= 60:
        readiness    = base_readiness - 10  # moderate penalty
        category     = "Needs Optimization"
        band_color   = "YELLOW"
        band_range   = "40–59"
        description  = "Drug requires dose optimization and close monitoring."
    else:
        readiness    = base_readiness - 15  # high penalty
        category     = "High Risk"
        band_color   = "RED"
        band_range   = "0–39"
        description  = "Drug poses significant risk. Consider alternative therapy."

    readiness = max(0, round(readiness, 2))

    return {
        "readiness_score" : readiness,
        "category"        : category,
        "band_color"      : band_color,
        "band_range"      : band_range,
        "description"     : description,
    }


# ============================================================
# STEP 2 — Sub-Score Breakdown
# Breaks down readiness into individual dimension scores
# ============================================================

def calculate_sub_scores(report: dict) -> dict:
    """
    Calculates individual readiness sub-scores from risk report.
    Each sub-score is out of 100 (higher = more ready/safe).
    """
    tox_score    = report["toxicity"]["score"]   * 100
    ddi_score    = report["ddi"]["score"]         * 100
    struct_score = report["structural"]["score"]  * 100
    pat_score    = report["patient"]["vulnerability"]["vulnerability_score"]

    return {
        "Molecular Safety"       : round(100 - tox_score,    2),
        "Interaction Safety"     : round(100 - ddi_score,    2),
        "Structural Simplicity"  : round(100 - struct_score, 2),
        "Patient Compatibility"  : round(100 - pat_score,    2),
    }


# ============================================================
# STEP 3 — Clinical Recommendations Generator
# ============================================================

def generate_clinical_recommendations(
    report   : dict,
    readiness: dict
) -> list:
    """
    Generates specific clinical recommendations based on
    individual risk components.
    """
    recs = []

    tox   = report["toxicity"]["score"]   * 100
    ddi   = report["ddi"]["score"]        * 100
    pat   = report["patient"]["vulnerability"]["vulnerability_score"]
    kidney= report["patient"]["kidney_risk"]["score"] * 100
    liver = report["patient"]["liver_risk"]["score"]  * 100
    age   = report["patient"]["age_risk"]["score"]    * 100

    # Toxicity recommendations
    if tox > 40:
        recs.append("⚠ High molecular toxicity detected — consider structural analogs.")
    elif tox > 20:
        recs.append("ℹ Moderate toxicity signals — monitor liver enzymes during treatment.")

    # DDI recommendations
    if ddi >= 100:
        recs.append("🚫 Major drug-drug interaction — avoid co-administration.")
    elif ddi >= 50:
        recs.append("⚠ Moderate DDI detected — stagger doses or choose alternative.")
    elif ddi > 0:
        recs.append("ℹ Minor drug interaction — routine monitoring sufficient.")
    else:
        recs.append("✓ No known drug-drug interaction found in database.")

    # Kidney recommendations
    if kidney >= 60:
        recs.append("🚫 Severe kidney impairment — reduce dose significantly or avoid.")
    elif kidney >= 40:
        recs.append("⚠ Moderate kidney impairment — dose reduction and eGFR monitoring needed.")
    elif kidney >= 20:
        recs.append("ℹ Mild kidney reduction — standard dose with periodic eGFR check.")

    # Liver recommendations
    if liver >= 70:
        recs.append("🚫 Critical liver enzyme elevation — hepatotoxic drugs contraindicated.")
    elif liver >= 40:
        recs.append("⚠ Elevated liver enzymes — avoid hepatotoxic drugs, monitor LFTs.")

    # Age recommendations
    if age >= 40:
        recs.append("⚠ Elderly patient — start low, go slow dosing strategy recommended.")
    elif age >= 20:
        recs.append("ℹ Middle-aged patient — standard dosing with periodic review.")

    # Patient vulnerability
    if pat >= 70:
        recs.append("🚫 High patient vulnerability — full specialist review mandatory.")
    elif pat >= 45:
        recs.append("⚠ Moderate patient vulnerability — pharmacist review recommended.")

    # Positive note if everything is fine
    if len(recs) == 0:
        recs.append("✓ All parameters within normal range — standard protocol applicable.")

    return recs


# ============================================================
# STEP 4 — Full Readiness Report
# ============================================================

def generate_readiness_report(
    smiles   : str,
    drug_name: str,
    co_drug  : str,
    patient  : dict
) -> dict:
    """
    Master function for Layer 7.
    Runs full pipeline and generates Development Readiness Score.
    """

    # Get full risk report from Layer 6
    report   = generate_risk_report(smiles, drug_name, co_drug, patient)
    unified  = report["unified"]["unified_score"]

    # Calculate readiness score
    readiness   = calculate_readiness_score(unified)
    sub_scores  = calculate_sub_scores(report)
    recs        = generate_clinical_recommendations(report, readiness)

    # Print Readiness Report
    print("\n" + "=" * 60)
    print("PHARMAGUARD AI — DEVELOPMENT READINESS REPORT")
    print("=" * 60)
    print(f"  Drug          : {drug_name}")
    print(f"  Co-Drug       : {co_drug}")
    print(f"  Patient Age   : {patient['age']} yrs | "
          f"eGFR: {patient['egfr']} | "
          f"ALT: {patient['alt']}")

    print(f"\n  {'─'*50}")
    print(f"  READINESS SCORE  :  {readiness['readiness_score']} / 100")
    print(f"  CATEGORY         :  {readiness['category']}")
    print(f"  BAND             :  {readiness['band_range']} ({readiness['band_color']})")
    print(f"  {'─'*50}")
    print(f"\n  {readiness['description']}")

    # Visual score bar
    filled = int(readiness["readiness_score"] / 5)
    empty  = 20 - filled
    bar    = "█" * filled + "░" * empty
    print(f"\n  [{bar}] {readiness['readiness_score']}%")

    # Sub-scores
    print(f"\n  Sub-Score Breakdown:")
    for k, v in sub_scores.items():
        bar2 = "█" * int(v / 5)
        print(f"    {k:<26}: {v:>6.1f}%  {bar2}")

    # Dosage
    dose = report["dosage"]
    print(f"\n  Dosage Recommendation : {dose['recommendation']}")
    print(f"  Dose Ratio            : {dose['dose_ratio']}")

    # Clinical recommendations
    print(f"\n  Clinical Recommendations:")
    for r in recs:
        print(f"    {r}")

    print("\n" + "=" * 60)

    return {
        "drug"       : drug_name,
        "co_drug"    : co_drug,
        "unified"    : unified,
        "readiness"  : readiness,
        "sub_scores" : sub_scores,
        "dosage"     : dose,
        "recommendations": recs,
        "full_report": report,
    }


# ============================================================
# MAIN — Test with 3 clinical cases
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHARMAGUARD AI — Layer 7: Development Readiness Scoring")
    print("=" * 60)

    # ── Case 1: Aspirin + Warfarin | Elderly diabetic ──
    print("\n" + "─" * 60)
    print("CASE 1 — Aspirin + Warfarin | Elderly Diabetic Patient")
    print("─" * 60)
    p1 = create_patient_profile(
        age=72, weight_kg=65, egfr=28,
        alt=45, ast=38, gender="female",
        conditions=["diabetes", "hypertension"]
    )
    r1 = generate_readiness_report(
        smiles    = "CC(=O)Oc1ccccc1C(=O)O",
        drug_name = "Aspirin",
        co_drug   = "Warfarin",
        patient   = p1
    )

    # ── Case 2: Metformin + Ibuprofen | Healthy adult ──
    print("\n" + "─" * 60)
    print("CASE 2 — Metformin + Ibuprofen | Healthy Young Adult")
    print("─" * 60)
    p2 = create_patient_profile(
        age=30, weight_kg=75, egfr=95,
        alt=28, ast=22, gender="male",
        conditions=[]
    )
    r2 = generate_readiness_report(
        smiles    = "CN(C)C(=N)NC(=N)N",
        drug_name = "Metformin",
        co_drug   = "Ibuprofen",
        patient   = p2
    )

    # ── Case 3: Paracetamol | Patient with liver disease ──
    print("\n" + "─" * 60)
    print("CASE 3 — Paracetamol | Patient with Liver Disease")
    print("─" * 60)
    p3 = create_patient_profile(
        age=52, weight_kg=88, egfr=65,
        alt=210, ast=190, gender="male",
        conditions=["liver disease", "hypertension"]
    )
    r3 = generate_readiness_report(
        smiles    = "CC(=O)Nc1ccc(O)cc1",
        drug_name = "Paracetamol",
        co_drug   = "Codeine",
        patient   = p3
    )

    print("\n" + "=" * 60)
    print("[DONE] Layer 7 complete!")
    print("       Development Readiness Scores generated")
    print("       Next: Layer 8 — PharmaGuard Dashboard")
    print("=" * 60)