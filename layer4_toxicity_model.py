# ============================================================
# PHARMAGUARD AI — Layer 4
# Module 1: Toxicity Prediction Model
# Models: Random Forest + XGBoost (Best Model Selection)
# Dataset: Tox21 (7823 molecules, 12 toxicity labels)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1 — Load Processed Data from Layer 3
# ============================================================

def load_layer3_data():
    print("[Layer 4] Loading processed data from Layer 3...")

    X_fused = np.load("datasets/fused_features.npy")
    X_fp    = np.load("datasets/fingerprints.npy")
    y_df    = pd.read_csv("datasets/tox21_labels.csv", index_col=0)

    # Combine fused features + fingerprints as full feature set
    X = np.hstack([X_fused, X_fp])

    print(f"[OK] Feature matrix shape : {X.shape}")
    print(f"[OK] Labels shape         : {y_df.shape}")
    return X, y_df


# ============================================================
# STEP 2 — Train Random Forest Model
# ============================================================

def train_random_forest(X_train, y_train):
    print("\n  [RF] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("  [RF] Training complete.")
    return rf


# ============================================================
# STEP 3 — Train XGBoost Model
# ============================================================

def train_xgboost(X_train, y_train):
    print("\n  [XGB] Training XGBoost...")
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,    # handles class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    print("  [XGB] Training complete.")
    return xgb


# ============================================================
# STEP 4 — Evaluate Model
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)

    print(f"\n  [{model_name}] Results:")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")

    return {"model": model_name, "accuracy": acc, "f1": f1, "auc": auc}


# ============================================================
# STEP 5 — Best Model Selection per Label
# ============================================================

def select_best_model(rf_metrics, xgb_metrics, rf_model, xgb_model):
    """Select best model based on ROC-AUC score"""
    if xgb_metrics["auc"] >= rf_metrics["auc"]:
        print(f"  [BEST] XGBoost selected (AUC: {xgb_metrics['auc']:.4f})")
        return xgb_model, "XGBoost"
    else:
        print(f"  [BEST] Random Forest selected (AUC: {rf_metrics['auc']:.4f})")
        return rf_model, "RandomForest"


# ============================================================
# STEP 6 — Full Training Pipeline per Toxicity Label
# ============================================================

def train_all_labels(X, y_df):
    print("\n" + "=" * 60)
    print("Training toxicity models for all 12 labels...")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    results_summary = []

    for label in y_df.columns:
        print(f"\n{'─'*50}")
        print(f"[LABEL] {label}")
        print(f"{'─'*50}")

        y = y_df[label].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        print(f"  Toxic in test: {y_test.sum()} | Non-toxic: {(y_test==0).sum()}")

        # Train both models
        rf_model  = train_random_forest(X_train, y_train)
        xgb_model = train_xgboost(X_train, y_train)

        # Evaluate both
        rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "RandomForest")
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

        # Select best
        best_model, best_name = select_best_model(
            rf_metrics, xgb_metrics, rf_model, xgb_model
        )

        # Save best model
        model_path = f"models/tox_{label.replace('-','_')}_best.pkl"
        joblib.dump(best_model, model_path)
        print(f"  [SAVED] {model_path}")

        # Store summary
        best_metrics = xgb_metrics if best_name == "XGBoost" else rf_metrics
        results_summary.append({
            "Label"      : label,
            "Best Model" : best_name,
            "Accuracy"   : round(best_metrics["accuracy"], 4),
            "F1 Score"   : round(best_metrics["f1"], 4),
            "ROC-AUC"    : round(best_metrics["auc"], 4),
        })

    return results_summary


# ============================================================
# STEP 7 — Predict Toxicity for a New Molecule
# ============================================================

def predict_toxicity(smiles: str):
    """
    Given a SMILES string, predict toxicity across all 12 labels.
    Uses saved best models from training.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    from sklearn.preprocessing import StandardScaler

    print(f"\n[PREDICT] Molecule: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("[ERROR] Invalid SMILES")
        return None

    # Extract descriptors
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

    # Extract fingerprint
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    # Combine
    X = np.hstack([desc, fp]).reshape(1, -1)

    # Predict using all saved models
    label_cols = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    predictions = {}
    print(f"\n  Toxicity Predictions:")
    print(f"  {'Label':<20} {'Prediction':<12} {'Probability'}")
    print(f"  {'─'*50}")

    for label in label_cols:
        model_path = f"models/tox_{label.replace('-','_')}_best.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            pred  = model.predict(X)[0]
            prob  = model.predict_proba(X)[0][1]
            result = "TOXIC" if pred == 1 else "Non-toxic"
            print(f"  {label:<20} {result:<12} {prob:.4f}")
            predictions[label] = {"prediction": result, "probability": round(float(prob), 4)}
        else:
            print(f"  {label:<20} [Model not found]")

    return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHARMAGUARD AI — Layer 4: Toxicity Prediction Model")
    print("=" * 60)

    # Load data
    X, y_df = load_layer3_data()

    # Train all 12 toxicity models
    results = train_all_labels(X, y_df)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — Best Models per Toxicity Label")
    print("=" * 60)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv("datasets/model_summary.csv", index=False)
    print("\n[SAVED] Model summary → datasets/model_summary.csv")

    # Test prediction on a sample molecule (Aspirin)
    print("\n" + "=" * 60)
    print("TEST PREDICTION — Aspirin")
    print("=" * 60)
    predict_toxicity("CC(=O)Oc1ccccc1C(=O)O")

    print("\n" + "=" * 60)
    print("[DONE] Layer 4 complete!")
    print("       12 toxicity models trained and saved in /models/")
    print("       Next: Layer 5 — Patient-Aware Risk Layer")
    print("=" * 60)