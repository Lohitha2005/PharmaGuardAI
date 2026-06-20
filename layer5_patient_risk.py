# ============================================================
# PHARMAGUARD AI — Layer 5
# Patient-Aware Risk Layer
# Factors: Age, Weight, Kidney Function (eGFR),
#          Liver Function (LFT), Vulnerability Index
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1 — Patient Profile Input
# ============================================================

def create_patient_profile(
    age        : float,
    weight_kg  : float,
    egfr       : float,   # Kidney: eGFR ml/min/1.73m² (normal >60)
    alt        : float,   # Liver: ALT U/L (normal 7–56)
    ast        : float,   # Liver: AST U/L (normal 10–40)
    gender     : str = "male",   # male / female
    conditions : list = []       # e.g. ["diabetes", "hypertension"]
):
    """
    Creates a structured patient profile dictionary.
    All values are validated and flagged if abnormal.
    """
    profile = {
        "age"        : age,
        "weight_kg"  : weight_kg,
        "egfr"       : egfr,
        "alt"        : alt,
        "ast"        : ast,
        "gender"     : gender.lower(),
        "conditions" : [c.lower() for c in conditions],
    }

    print("\n[Layer 5] Patient Profile:")
    print(f"  Age          : {age} years")
    print(f"  Weight       : {weight_kg} kg")
    print(f"  eGFR (Kidney): {egfr} ml/min/1.73m²")
    print(f"  ALT (Liver)  : {alt} U/L")
    print(f"  AST (Liver)  : {ast} U/L")
    print(f"  Gender       : {gender}")
    print(f"  Conditions   : {conditions if conditions else 'None'}")

    return profile


# ============================================================
# STEP 2 — Kidney Function Risk Score (eGFR based)
# CKD Staging: 
#   Stage 1: eGFR >= 90  → Normal
#   Stage 2: eGFR 60–89  → Mildly reduced
#   Stage 3: eGFR 30–59  → Moderately reduced
#   Stage 4: eGFR 15–29  → Severely reduced
#   Stage 5: eGFR < 15   → Kidney failure
# ============================================================

def kidney_risk_score(egfr: float) -> dict:
    if egfr >= 90:
        score, stage, label = 0.0,  "Stage 1", "Normal"
    elif egfr >= 60:
        score, stage, label = 0.2,  "Stage 2", "Mildly Reduced"
    elif egfr >= 45:
        score, stage, label = 0.4,  "Stage 3a", "Mild-Moderate Reduction"
    elif egfr >= 30:
        score, stage, label = 0.6,  "Stage 3b", "Moderate-Severe Reduction"
    elif egfr >= 15:
        score, stage, label = 0.8,  "Stage 4", "Severely Reduced"
    else:
        score, stage, label = 1.0,  "Stage 5", "Kidney Failure"

    return {"score": score, "stage": stage, "label": label, "egfr": egfr}


# ============================================================
# STEP 3 — Liver Function Risk Score (ALT + AST based)
# Normal ALT: 7–56 U/L | Normal AST: 10–40 U/L
# ============================================================

def liver_risk_score(alt: float, ast: float) -> dict:
    # Use the higher of the two values for risk
    max_enzyme = max(alt, ast)

    if max_enzyme <= 56:
        score, label = 0.0, "Normal"
    elif max_enzyme <= 100:
        score, label = 0.2, "Mildly Elevated"
    elif max_enzyme <= 200:
        score, label = 0.4, "Moderately Elevated"
    elif max_enzyme <= 500:
        score, label = 0.7, "Severely Elevated"
    else:
        score, label = 1.0, "Critical"

    return {"score": score, "label": label, "alt": alt, "ast": ast}


# ============================================================
# STEP 4 — Age Risk Score
# Older patients have reduced drug metabolism
# ============================================================

def age_risk_score(age: float) -> dict:
    if age < 18:
        score, label = 0.5, "Pediatric — Special dosing required"
    elif age <= 40:
        score, label = 0.0, "Young Adult — Normal metabolism"
    elif age <= 60:
        score, label = 0.2, "Middle-Aged — Slightly reduced"
    elif age <= 75:
        score, label = 0.4, "Senior — Reduced metabolism"
    else:
        score, label = 0.7, "Elderly — Significantly reduced"

    return {"score": score, "label": label, "age": age}


# ============================================================
# STEP 5 — Weight / BMI Risk Score
# ============================================================

def weight_risk_score(weight_kg: float, age: float) -> dict:
    # Simplified weight-based risk (affects drug volume of distribution)
    if weight_kg < 40:
        score, label = 0.4, "Underweight — Low distribution volume"
    elif weight_kg <= 100:
        score, label = 0.0, "Normal — Standard dosing"
    elif weight_kg <= 130:
        score, label = 0.2, "Overweight — Adjusted dosing needed"
    else:
        score, label = 0.4, "Obese — Significant dosing adjustment"

    return {"score": score, "label": label, "weight_kg": weight_kg}


# ============================================================
# STEP 6 — Comorbidity Risk Score
# Additional risk from existing medical conditions
# ============================================================

def comorbidity_risk_score(conditions: list) -> dict:
    high_risk = {
        "diabetes"          : 0.3,
        "hypertension"      : 0.2,
        "heart failure"     : 0.5,
        "liver disease"     : 0.6,
        "kidney disease"    : 0.6,
        "cancer"            : 0.4,
        "hiv"               : 0.3,
        "copd"              : 0.3,
        "epilepsy"          : 0.2,
        "thyroid disorder"  : 0.2,
    }

    total_score = 0.0
    matched = []
    for cond in conditions:
        for key, val in high_risk.items():
            if key in cond:
                total_score += val
                matched.append(f"{key} (+{val})")

    # Cap at 1.0
    total_score = min(total_score, 1.0)

    return {
        "score"   : total_score,
        "matched" : matched if matched else ["None"],
        "label"   : "High Risk" if total_score > 0.5 else
                    "Moderate Risk" if total_score > 0.2 else "Low Risk"
    }


# ============================================================
# STEP 7 — Vulnerability Index
# Combined patient vulnerability from all factors
# ============================================================

def calculate_vulnerability_index(
    kidney_risk : dict,
    liver_risk  : dict,
    age_risk    : dict,
    weight_risk : dict,
    comorbidity : dict
) -> dict:
    """
    Weighted combination of all risk factors.
    Weights based on clinical importance for drug safety.
    """
    weights = {
        "kidney" : 0.30,   # Most important — kidney clears most drugs
        "liver"  : 0.30,   # Most important — liver metabolizes most drugs
        "age"    : 0.20,   # Age affects metabolism significantly
        "weight" : 0.10,   # Weight affects drug distribution
        "comorbidity": 0.10  # Additional conditions add risk
    }

    v_index = (
        weights["kidney"]      * kidney_risk["score"] +
        weights["liver"]       * liver_risk["score"]  +
        weights["age"]         * age_risk["score"]    +
        weights["weight"]      * weight_risk["score"] +
        weights["comorbidity"] * comorbidity["score"]
    )

    # Convert to 0–100 scale
    v_score = round(v_index * 100, 2)

    if v_score <= 20:
        risk_level = "LOW"
        color      = "GREEN"
    elif v_score <= 45:
        risk_level = "MODERATE"
        color      = "YELLOW"
    elif v_score <= 70:
        risk_level = "HIGH"
        color      = "ORANGE"
    else:
        risk_level = "CRITICAL"
        color      = "RED"

    return {
        "vulnerability_score" : v_score,
        "risk_level"          : risk_level,
        "color"               : color,
        "breakdown": {
            "Kidney Risk"      : round(kidney_risk["score"] * 100, 1),
            "Liver Risk"       : round(liver_risk["score"]  * 100, 1),
            "Age Risk"         : round(age_risk["score"]    * 100, 1),
            "Weight Risk"      : round(weight_risk["score"] * 100, 1),
            "Comorbidity Risk" : round(comorbidity["score"] * 100, 1),
        }
    }


# ============================================================
# STEP 8 — Full Patient Risk Assessment
# ============================================================

def assess_patient_risk(patient: dict) -> dict:
    """
    Full Layer 5 pipeline for one patient profile.
    Returns complete risk assessment.
    """
    print("\n[Layer 5] Running Patient Risk Assessment...")

    # Individual risk scores
    kidney  = kidney_risk_score(patient["egfr"])
    liver   = liver_risk_score(patient["alt"], patient["ast"])
    age     = age_risk_score(patient["age"])
    weight  = weight_risk_score(patient["weight_kg"], patient["age"])
    comorbid= comorbidity_risk_score(patient["conditions"])

    # Vulnerability index
    vuln    = calculate_vulnerability_index(kidney, liver, age, weight, comorbid)

    # Print results
    print(f"\n  ── Individual Risk Scores ──")
    print(f"  Kidney  : {kidney['label']:<30} Score: {kidney['score']:.2f}")
    print(f"  Liver   : {liver['label']:<30} Score: {liver['score']:.2f}")
    print(f"  Age     : {age['label']:<30} Score: {age['score']:.2f}")
    print(f"  Weight  : {weight['label']:<30} Score: {weight['score']:.2f}")
    print(f"  Comorbid: {comorbid['label']:<30} Score: {comorbid['score']:.2f}")

    print(f"\n  ── Vulnerability Index ──")
    print(f"  Score      : {vuln['vulnerability_score']} / 100")
    print(f"  Risk Level : {vuln['risk_level']} ({vuln['color']})")
    print(f"\n  Breakdown:")
    for k, v in vuln["breakdown"].items():
        bar = "█" * int(v / 5)
        print(f"    {k:<20}: {v:>5.1f}%  {bar}")

    return {
        "patient"         : patient,
        "kidney_risk"     : kidney,
        "liver_risk"      : liver,
        "age_risk"        : age,
        "weight_risk"     : weight,
        "comorbidity"     : comorbid,
        "vulnerability"   : vuln,
    }


# ============================================================
# STEP 9 — Dosage Adjustment Recommendation
# Based on vulnerability index
# ============================================================

def dosage_recommendation(vuln_score: float, drug_name: str = "Drug") -> str:
    """
    Gives dosage adjustment recommendation based on patient vulnerability.
    This is the CORE of your agent idea — personalized dosage.
    """
    print(f"\n  ── Dosage Recommendation for {drug_name} ──")

    if vuln_score <= 20:
        rec = "Standard dose — No adjustment needed."
        ratio = "100%"
    elif vuln_score <= 35:
        rec = "Slight reduction recommended — Monitor closely."
        ratio = "75–90%"
    elif vuln_score <= 50:
        rec = "Moderate dose reduction — Regular monitoring required."
        ratio = "50–75%"
    elif vuln_score <= 70:
        rec = "Significant dose reduction — Specialist consultation advised."
        ratio = "25–50%"
    else:
        rec = "Drug may be contraindicated — Seek alternative therapy."
        ratio = "Contraindicated"

    print(f"  Recommendation : {rec}")
    print(f"  Dose Ratio     : {ratio} of standard dose")

    return {"recommendation": rec, "dose_ratio": ratio}


# ============================================================
# MAIN — Test with sample patients
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHARMAGUARD AI — Layer 5: Patient-Aware Risk Layer")
    print("=" * 60)

    # ── Patient 1: Healthy young adult ──
    print("\n" + "─" * 60)
    print("PATIENT 1 — Healthy Young Adult")
    print("─" * 60)
    p1 = create_patient_profile(
        age=28, weight_kg=70, egfr=95,
        alt=30, ast=25, gender="male",
        conditions=[]
    )
    r1 = assess_patient_risk(p1)
    dosage_recommendation(r1["vulnerability"]["vulnerability_score"], "Aspirin")

    # ── Patient 2: Elderly with kidney issues ──
    print("\n" + "─" * 60)
    print("PATIENT 2 — Elderly with Kidney Issues")
    print("─" * 60)
    p2 = create_patient_profile(
        age=72, weight_kg=65, egfr=28,
        alt=45, ast=38, gender="female",
        conditions=["diabetes", "hypertension"]
    )
    r2 = assess_patient_risk(p2)
    dosage_recommendation(r2["vulnerability"]["vulnerability_score"], "Metformin")

    # ── Patient 3: Middle-aged with liver issues ──
    print("\n" + "─" * 60)
    print("PATIENT 3 — Middle-aged with Liver Issues")
    print("─" * 60)
    p3 = create_patient_profile(
        age=52, weight_kg=95, egfr=65,
        alt=180, ast=210, gender="male",
        conditions=["liver disease", "hypertension"]
    )
    r3 = assess_patient_risk(p3)
    dosage_recommendation(r3["vulnerability"]["vulnerability_score"], "Paracetamol")

    print("\n" + "=" * 60)
    print("[DONE] Layer 5 complete!")
    print("       Patient vulnerability scores calculated")
    print("       Next: Layer 6 — Risk Fusion Engine")
    print("=" * 60)