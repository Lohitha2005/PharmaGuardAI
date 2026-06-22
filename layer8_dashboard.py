# ============================================================
# PHARMAGUARD AI — Layer 8 (Updated)
# PharmaGuard Dashboard with SHAP Integration
# ============================================================

from flask import Flask, render_template_string, request, jsonify, send_from_directory, send_file, abort
import os
import io
import glob
import warnings
import requests
from functools import lru_cache
warnings.filterwarnings("ignore")

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set your Groq API key in a .env file (never commit the key to git):
#   GROQ_API_KEY=gsk_...
# Or set it in your terminal before running:
#   $env:GROQ_API_KEY = "gsk_..."
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY not set — AI Explanation tab will not work.")
    print("       Add GROQ_API_KEY=gsk_... to your .env file.")

from rdkit import Chem
from rdkit.Chem import Draw

from layer5_patient_risk    import create_patient_profile
from layer7_readiness_score import generate_readiness_report
from module3_shap_explainability import quick_shap_explain

app = Flask(__name__)

# ------------------------------------------------------------
# SHAP plot storage
# ------------------------------------------------------------
# Resolve the shap_plots folder relative to THIS file (not the
# process's current working directory). This is what makes the
# image lookup work no matter where `python layer8_dashboard.py`
# is launched from.
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SHAP_PLOTS_DIR = os.path.join(BASE_DIR, "shap_plots")


def find_shap_image(drug_name: str):
    """
    Locate the saved SHAP plot for a given drug, regardless of which
    toxicity label (SR-ARE, NR-AhR, ...) it was generated against.

    Plot filenames follow the pattern: "<DrugName>_<LABEL>_shap.png"
    (e.g. "Aspirin_SR-ARE_shap.png", "Paracetamol_NR-AhR_shap.png").
    Since the label suffix can vary per drug, we glob on a wildcard
    instead of hardcoding one label.

    Returns the absolute file path if found, else None.
    """
    if not drug_name:
        return None

    safe_name = drug_name.strip().replace(" ", "_")
    if not safe_name or not os.path.isdir(SHAP_PLOTS_DIR):
        return None

    # 1) Exact-case match (fast path)
    pattern = os.path.join(SHAP_PLOTS_DIR, f"{safe_name}_*_shap.png")
    matches = glob.glob(pattern)

    # 2) Case-insensitive fallback (handles "aspirin" vs "Aspirin")
    if not matches:
        target_prefix = (safe_name + "_").lower()
        for fname in os.listdir(SHAP_PLOTS_DIR):
            if fname.lower().startswith(target_prefix) and fname.lower().endswith("_shap.png"):
                matches.append(os.path.join(SHAP_PLOTS_DIR, fname))

    if not matches:
        return None

    # A drug can end up with more than one saved plot — e.g. an old
    # example file under one toxicity label, plus a fresh one saved by
    # a live analysis under a different label. glob/listdir order is
    # filesystem-dependent, not meaningful, so picking matches[0] would
    # be non-deterministic. We deterministically prefer whichever file
    # was modified most recently, since that's the most likely to
    # reflect the latest live analysis the user actually ran.
    return max(matches, key=os.path.getmtime)


# ------------------------------------------------------------
# Molecule structure rendering
# ------------------------------------------------------------
@lru_cache(maxsize=128)
def render_molecule_png(smiles: str) -> bytes:
    """
    Renders a SMILES string as a 2D structure diagram (PNG bytes).

    @lru_cache means identical SMILES strings (very common — people
    re-analyze the same drug repeatedly) are rendered once per process
    lifetime instead of re-drawn on every request. Returns None if the
    SMILES is invalid so the route can respond with a clean 400/404
    instead of letting RDKit raise inside the request handler.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=(420, 320))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------
# Drug-name -> SMILES lookup (PubChem)
# ------------------------------------------------------------
PUBCHEM_SMILES_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/"
    "property/CanonicalSMILES/JSON"
)


_smiles_lookup_cache = {}


def lookup_smiles_from_pubchem(drug_name: str):
    """
    Resolves a drug name (e.g. "pantoprazole") to its canonical SMILES
    string via PubChem's public PUG REST API. This is what lets the
    dashboard accept a plain drug name instead of requiring the user
    to already know/paste the SMILES themselves.

    Returns (smiles, error_message):
      - on success: (smiles_string, None)
      - on failure: (None, a specific reason — timeout, SSL error,
        connection error, "not found", unexpected status, etc.)

    Distinguishing failure causes matters here: a network/firewall
    problem and "this drug genuinely isn't on PubChem" look identical
    to the user unless we report them differently.

    Only successful lookups are cached (in a plain dict, not
    @lru_cache) — caching a failure would be wrong if the failure was
    a transient network blip the user could retry past.
    """
    key = drug_name.strip().lower()
    if key in _smiles_lookup_cache:
        return _smiles_lookup_cache[key], None

    url = PUBCHEM_SMILES_URL.format(name=requests.utils.quote(drug_name))

    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.SSLError:
        return None, ("SSL error connecting to PubChem. This is commonly caused by a "
                       "corporate firewall, antivirus, or VPN doing HTTPS inspection.")
    except requests.exceptions.ConnectionError:
        return None, ("Could not reach PubChem — check your internet connection or "
                       "whether a firewall is blocking pubchem.ncbi.nlm.nih.gov.")
    except requests.exceptions.Timeout:
        return None, "PubChem did not respond in time (request timed out)."
    except Exception as e:
        return None, f"Unexpected error contacting PubChem: {e}"

    if resp.status_code == 404:
        return None, f"\"{drug_name}\" was not recognized by PubChem. Try the generic/chemical name."
    if resp.status_code != 200:
        return None, f"PubChem returned an unexpected status code ({resp.status_code})."

    try:
        props = resp.json()["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, ValueError):
        return None, "PubChem responded, but in an unexpected format."

    # PubChem deprecated "CanonicalSMILES"/"IsomericSMILES" in favor of
    # "ConnectivitySMILES"/"SMILES" — the request URL above still works
    # (PubChem accepts the old property name), but the JSON key that
    # actually comes back has the NEW name. We check every key PubChem
    # has used for this field (old and new, stereo and non-stereo) so
    # this keeps working regardless of which naming era responded.
    # Stereo-aware keys are preferred when available (more precise).
    for key in ("IsomericSMILES", "SMILES", "ConnectivitySMILES", "CanonicalSMILES"):
        if props.get(key):
            smiles = props[key]
            break
    else:
        return None, "PubChem responded, but didn't include a usable SMILES property."

    _smiles_lookup_cache[key] = smiles
    return smiles, None

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PharmaGuard AI</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Segoe UI',sans-serif; background:#0f1117; color:#e0e0e0; min-height:100vh; }

.header {
  background:linear-gradient(135deg,#00c9ff,#0077b6);
  padding:24px 40px; display:flex; align-items:center; gap:16px;
}
.header h1 { font-size:28px; font-weight:700; color:#fff; letter-spacing:1px; }
.header p  { font-size:13px; color:#d0f0ff; margin-top:4px; }
.logo { font-size:36px; }

.container { max-width:1300px; margin:0 auto; padding:32px 24px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
.grid4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:16px; margin-top:24px; }

.card {
  background:#1e2130; border-radius:12px;
  padding:24px; border:1px solid #2a2f45;
}
.card h3 {
  font-size:13px; color:#7b8ab8; text-transform:uppercase;
  letter-spacing:1px; margin-bottom:16px;
}

.form-group { margin-bottom:14px; }
label { display:block; font-size:12px; color:#8892b0; margin-bottom:6px; }
input, select {
  width:100%; padding:10px 14px; background:#151827;
  border:1px solid #2a2f45; border-radius:8px;
  color:#e0e0e0; font-size:14px; outline:none; transition:border 0.2s;
}
input:focus, select:focus { border-color:#00c9ff; }
.row2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.row3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }

.btn {
  width:100%; padding:14px;
  background:linear-gradient(135deg,#00c9ff,#0077b6);
  border:none; border-radius:8px; color:#fff;
  font-size:16px; font-weight:600; cursor:pointer;
  margin-top:8px; transition:opacity 0.2s;
}
.btn:hover { opacity:0.9; }
.btn:disabled { opacity:0.5; cursor:not-allowed; }

.score-ring {
  width:150px; height:150px; border-radius:50%;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  margin:0 auto 20px; border:6px solid #2a2f45;
}
.score-number { font-size:34px; font-weight:700; }
.score-label  { font-size:12px; color:#8892b0; margin-top:2px; }
.green  { border-color:#00d084; color:#00d084; }
.blue   { border-color:#00c9ff; color:#00c9ff; }
.yellow { border-color:#f5c542; color:#f5c542; }
.red    { border-color:#ff5c5c; color:#ff5c5c; }

.progress-wrap { margin-bottom:12px; }
.progress-label {
  display:flex; justify-content:space-between;
  font-size:12px; color:#8892b0; margin-bottom:4px;
}
.progress-track { height:8px; background:#151827; border-radius:4px; overflow:hidden; }
.progress-fill {
  height:100%; border-radius:4px;
  background:linear-gradient(90deg,#00c9ff,#0077b6); transition:width 0.8s ease;
}
.progress-fill.danger { background:linear-gradient(90deg,#ff5c5c,#ff8c42); }

.badge {
  display:inline-block; padding:5px 14px; border-radius:20px;
  font-size:12px; font-weight:600; margin-bottom:12px;
}
.badge-green  { background:#00d08422; color:#00d084; border:1px solid #00d084; }
.badge-blue   { background:#00c9ff22; color:#00c9ff; border:1px solid #00c9ff; }
.badge-yellow { background:#f5c54222; color:#f5c542; border:1px solid #f5c542; }
.badge-red    { background:#ff5c5c22; color:#ff5c5c; border:1px solid #ff5c5c; }

.metric {
  background:#151827; border-radius:10px;
  padding:16px; text-align:center; border:1px solid #2a2f45;
}
.metric-value { font-size:26px; font-weight:700; color:#00c9ff; }
.metric-label { font-size:11px; color:#8892b0; margin-top:4px; }

.rec-item {
  padding:10px 14px; margin-bottom:8px;
  background:#151827; border-radius:8px;
  font-size:13px; line-height:1.5; border-left:3px solid #2a2f45;
}

.dose-box {
  background:#151827; border-radius:10px;
  padding:16px; border-left:4px solid #00c9ff; margin-top:16px;
}
.dose-box .ratio { font-size:22px; font-weight:700; color:#00c9ff; }
.dose-box .rec   { font-size:13px; color:#8892b0; margin-top:4px; }

.shap-feature {
  display:flex; align-items:center; gap:10px;
  padding:8px 12px; margin-bottom:6px;
  background:#151827; border-radius:8px;
}
.shap-name { font-size:12px; color:#8892b0; width:160px; flex-shrink:0; }
.shap-bar-wrap { flex:1; height:10px; background:#0f1117; border-radius:5px; overflow:hidden; }
.shap-bar-fill { height:100%; border-radius:5px; transition:width 0.8s ease; }
.shap-toxic { background:linear-gradient(90deg,#ff5c5c,#ff8c42); }
.shap-safe  { background:linear-gradient(90deg,#00d084,#00c9ff); }
.shap-val   { font-size:11px; width:80px; text-align:right; color:#e0e0e0; }
.shap-dir   {
  font-size:10px; font-weight:600; width:60px;
  text-align:center; padding:2px 6px; border-radius:4px;
}
.shap-dir.toxic { background:#ff5c5c22; color:#ff5c5c; }
.shap-dir.safe  { background:#00d08422; color:#00d084; }

.shap-img {
  width:100%; border-radius:10px;
  border:1px solid #2a2f45; margin-top:16px;
}

.tabs { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }
.tab {
  padding:8px 18px; border-radius:8px; font-size:13px;
  cursor:pointer; border:1px solid #2a2f45;
  background:#151827; color:#8892b0; transition:all 0.2s;
}
.tab.active { background:#00c9ff22; color:#00c9ff; border-color:#00c9ff; }
.tab-content { display:none; }
.tab-content.active { display:block; }

.loading { display:none; text-align:center; padding:40px; color:#8892b0; }
.spinner {
  width:40px; height:40px; border:4px solid #2a2f45;
  border-top:4px solid #00c9ff; border-radius:50%;
  animation:spin 0.8s linear infinite; margin:0 auto 16px;
}
@keyframes spin { to { transform:rotate(360deg); } }
#result { display:none; margin-top:24px; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">🧬</div>
  <div>
    <h1>PharmaGuard AI</h1>
    <p>Explainable Drug Safety &amp; Compatibility Intelligence Platform</p>
  </div>
</div>

<div class="container">

  <div class="grid2">
    <div class="card">
      <h3>💊 Drug Information</h3>
      <div class="form-group">
        <label>Drug Name</label>
        <input id="drug_name" type="text" value="Aspirin">
      </div>
      <div class="form-group">
        <label>SMILES String</label>
        <div style="display:flex; gap:8px;">
          <input id="smiles" type="text" value="CC(=O)Oc1ccccc1C(=O)O" style="flex:1;">
          <button type="button" id="lookupBtn" onclick="lookupSmiles()"
            style="flex-shrink:0; padding:0 16px; background:#151827; border:1px solid #2a2f45;
                   border-radius:8px; color:#00c9ff; font-size:13px; cursor:pointer; white-space:nowrap;">
            🔍 Look up
          </button>
        </div>
        <p id="lookup_status" style="font-size:11px; margin-top:6px; min-height:14px;"></p>
      </div>
      <div class="form-group">
        <label>Co-administered Drug (DDI Check)</label>
        <input id="co_drug" type="text" value="Warfarin">
      </div>
    </div>

    <div class="card">
      <h3>👤 Patient Profile</h3>
      <div class="row2">
        <div class="form-group"><label>Age (years)</label><input id="age" type="number" value="45"></div>
        <div class="form-group"><label>Weight (kg)</label><input id="weight" type="number" value="70"></div>
      </div>
      <div class="row3">
        <div class="form-group"><label>eGFR (Kidney)</label><input id="egfr" type="number" value="75"></div>
        <div class="form-group"><label>ALT (Liver U/L)</label><input id="alt" type="number" value="35"></div>
        <div class="form-group"><label>AST (Liver U/L)</label><input id="ast" type="number" value="30"></div>
      </div>
      <div class="row2">
        <div class="form-group">
          <label>Gender</label>
          <select id="gender">
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
        <div class="form-group">
          <label>Conditions (comma-separated)</label>
          <input id="conditions" type="text" placeholder="e.g. diabetes, hypertension">
        </div>
      </div>
    </div>
  </div>

  <button class="btn" onclick="analyze()" id="analyzeBtn">🔍 Analyze Drug Safety</button>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>Running PharmaGuard AI + SHAP analysis... please wait</p>
  </div>

  <div id="result">

    <div class="grid4">
      <div class="metric"><div class="metric-value" id="m_readiness">—</div><div class="metric-label">Readiness Score</div></div>
      <div class="metric"><div class="metric-value" id="m_risk">—</div><div class="metric-label">Risk Score</div></div>
      <div class="metric"><div class="metric-value" id="m_tox">—</div><div class="metric-label">Toxicity Score</div></div>
      <div class="metric"><div class="metric-value" id="m_vuln">—</div><div class="metric-label">Patient Vulnerability</div></div>
    </div>

    <div class="tabs" style="margin-top:24px;">
      <div class="tab active" onclick="showTab('overview',this)">📊 Overview</div>
      <div class="tab" onclick="showTab('scores',this)">📈 Scores</div>
      <div class="tab" onclick="showTab('shap',this)">🔬 SHAP Explanation</div>
      <div class="tab" onclick="showTab('recs',this)">🩺 Recommendations</div>
      <div class="tab" onclick="showTab('aiexplain',this)">🤖 AI Explanation</div>
    </div>

    <!-- Overview Tab -->
    <div class="tab-content active" id="tab-overview">
      <div class="grid2">
        <div class="card" style="text-align:center;">
          <h3>📊 Development Readiness</h3>
          <div class="score-ring" id="score_ring">
            <span class="score-number" id="score_num">—</span>
            <span class="score-label">/ 100</span>
          </div>
          <div id="category_badge"></div>
          <p id="category_desc" style="font-size:13px;color:#8892b0;margin-top:8px;"></p>
          <div class="dose-box">
            <div class="ratio" id="dose_ratio">—</div>
            <div class="rec" id="dose_rec">Dosage Recommendation</div>
          </div>
        </div>
        <div class="card">
          <h3>⚡ Risk Components</h3>
          <div id="risk_components"></div>
          <h3 style="margin-top:20px;">🏥 DDI Status</h3>
          <div id="ddi_status" style="padding:12px;background:#151827;border-radius:8px;font-size:13px;"></div>
        </div>
      </div>

      <div class="card" style="margin-top:24px;">
        <h3>🧪 Molecular Structure</h3>
        <div id="structure_wrap" style="text-align:center;">
          <img id="structure_img" src="" alt="2D molecular structure"
               style="max-width:100%; border-radius:10px; background:#fff; padding:8px; display:none;">
          <p id="structure_missing" style="display:none; color:#8892b0; font-size:13px; padding:16px;">
            Could not render a structure for this SMILES string.
          </p>
        </div>
      </div>
    </div>

    <!-- Scores Tab -->
    <div class="tab-content" id="tab-scores">
      <div class="card">
        <h3>📈 Sub-Score Breakdown</h3>
        <div id="sub_scores"></div>
      </div>
    </div>

    <!-- SHAP Tab -->
    <div class="tab-content" id="tab-shap">
      <div class="card">
        <h3>🧠 Why did the model make this prediction?</h3>
        <p style="font-size:12px;color:#8892b0;margin-bottom:16px;">
          PharmaGuard AI uses SHAP (SHapley Additive exPlanations) to break the toxicity
          prediction down into the individual molecular features that pushed it toward
          TOXIC or SAFE, so the result isn't a black box.
        </p>

        <div id="shap_no_data" style="display:none;color:#8892b0;font-size:13px;padding:16px;background:#151827;border-radius:8px;">
          No explainability data available.
        </div>

        <div id="shap_content">
          <div class="row2">
            <div>
              <h3 style="color:#ff5c5c;">⬆ Increasing Toxicity Risk</h3>
              <div id="shap_increasing"></div>
            </div>
            <div>
              <h3 style="color:#00d084;">⬇ Decreasing Toxicity Risk</h3>
              <div id="shap_decreasing"></div>
            </div>
          </div>

          <h3 style="margin-top:20px;">📊 SHAP Bar Chart</h3>
          <img id="shap_img" class="shap-img" src="" alt="SHAP Bar Chart" style="display:none;">
          <p id="shap_img_missing" style="display:none;font-size:12px;color:#8892b0;margin-top:8px;">
            No explainability data available.
          </p>
        </div>
      </div>
    </div>

    <!-- Recommendations Tab -->
    <div class="tab-content" id="tab-recs">
      <div class="card">
        <h3>🩺 Clinical Recommendations</h3>
        <div id="recommendations"></div>
      </div>
    </div>

    <!-- AI Explanation Tab -->
    <div class="tab-content" id="tab-aiexplain">
      <div class="card">
        <h3>🤖 AI Explanation Engine</h3>
        <p style="font-size:12px;color:#8892b0;margin-bottom:16px;">
          Claude AI reads the SHAP values and toxicity scores from this analysis and writes
          a plain-English explanation of why the model made this prediction — the way a
          medicinal chemist would explain it to a clinician.
        </p>

        <div id="aiexplain_loading" style="display:none;padding:24px;text-align:center;">
          <div class="spinner" style="margin:0 auto 12px;"></div>
          <p style="color:#8892b0;font-size:13px;">Claude is reading the SHAP values and writing the explanation...</p>
        </div>

        <div id="aiexplain_content" style="display:none;">
          <!-- Summary badge row -->
          <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;" id="aiexplain_badges"></div>

          <!-- Main explanation paragraph -->
          <div id="aiexplain_text"
               style="background:#151827;border-left:3px solid #00c9ff;border-radius:0 8px 8px 0;
                      padding:18px 20px;font-size:14px;line-height:1.8;color:#ccd6f6;
                      white-space:pre-wrap;"></div>

          <!-- Key drivers table -->
          <h3 style="margin-top:24px;font-size:14px;">🔑 Key Drivers Summary</h3>
          <div id="aiexplain_drivers"
               style="background:#151827;border-radius:8px;padding:16px;font-size:13px;"></div>

          <!-- Regenerate button -->
          <button onclick="generateExplanation()"
                  style="margin-top:18px;padding:10px 22px;background:#00c9ff22;
                         border:1px solid #00c9ff;border-radius:8px;color:#00c9ff;
                         font-size:13px;cursor:pointer;">
            🔄 Regenerate Explanation
          </button>
        </div>

        <div id="aiexplain_empty" style="color:#8892b0;font-size:13px;padding:16px;background:#151827;border-radius:8px;">
          Run an analysis first — the AI explanation will appear here.
        </div>

        <div id="aiexplain_error" style="display:none;color:#ff5c5c;font-size:13px;
             padding:16px;background:#ff5c5c11;border-radius:8px;border:1px solid #ff5c5c44;"></div>
      </div>
    </div>

  </div>
</div>

<script>
// ---- AI Explanation Engine ----
let _lastAnalysisData = null;  // stored so the tab can generate on demand

async function generateExplanation() {
  if (!_lastAnalysisData) return;

  const loading = document.getElementById("aiexplain_loading");
  const content = document.getElementById("aiexplain_content");
  const empty   = document.getElementById("aiexplain_empty");
  const errBox  = document.getElementById("aiexplain_error");

  loading.style.display = "block";
  content.style.display = "none";
  empty.style.display   = "none";
  errBox.style.display  = "none";

  try {
    const res  = await fetch("/explain", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({
        drug_name    : _lastAnalysisData.drug_name,
        toxicity     : _lastAnalysisData.toxicity,
        readiness    : _lastAnalysisData.readiness.readiness_score,
        unified_score: _lastAnalysisData.unified_score,
        ddi_level    : _lastAnalysisData.ddi_level,
        vulnerability: _lastAnalysisData.vulnerability,
        shap_features: _lastAnalysisData.shap_features || [],
        sub_scores   : _lastAnalysisData.sub_scores || {},
      }),
    });
    const data = await res.json();

    loading.style.display = "none";

    if (!res.ok || data.error) {
      errBox.textContent    = "⚠ " + (data.error || "Unknown error.");
      errBox.style.display  = "block";
      return;
    }

    // Render summary badges
    const d = _lastAnalysisData;
    const badgeCfg = [
      { label: "Toxicity",    val: d.toxicity + "%",      color: d.toxicity > 50 ? "#ff5c5c" : "#00d084" },
      { label: "Readiness",   val: d.readiness.readiness_score + "%", color: "#00c9ff" },
      { label: "DDI Level",   val: d.ddi_level,            color: d.ddi_level === "None" ? "#00d084" : d.ddi_level === "Moderate" ? "#ffd700" : "#ff5c5c" },
      { label: "Risk Score",  val: d.unified_score + "%",  color: d.unified_score > 50 ? "#ff5c5c" : "#ffd700" },
    ];
    document.getElementById("aiexplain_badges").innerHTML = badgeCfg.map(b =>
      `<span style="background:${b.color}22;border:1px solid ${b.color}55;
        color:${b.color};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">
        ${b.label}: ${b.val}</span>`
    ).join("");

    // Render explanation text
    document.getElementById("aiexplain_text").textContent = data.explanation;

    // Render key drivers table from SHAP features
    const feats = (d.shap_features || []).slice(0, 6);
    document.getElementById("aiexplain_drivers").innerHTML = feats.length ? `
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="color:#8892b0;font-size:11px;text-align:left;border-bottom:1px solid #2a2f45;">
            <th style="padding:6px 8px;">Feature</th>
            <th style="padding:6px 8px;">SHAP Value</th>
            <th style="padding:6px 8px;">Direction</th>
            <th style="padding:6px 8px;">Impact</th>
          </tr>
        </thead>
        <tbody>
          ${feats.map(f => `
            <tr style="border-bottom:1px solid #2a2f4522;">
              <td style="padding:7px 8px;color:#ccd6f6;">${f.feature}</td>
              <td style="padding:7px 8px;color:${f.direction==="TOXIC"?"#ff5c5c":"#00d084"};">${f.shap_val > 0 ? "+" : ""}${f.shap_val.toFixed(4)}</td>
              <td style="padding:7px 8px;">
                <span style="background:${f.direction==="TOXIC"?"#ff5c5c22":"#00d08422"};
                  color:${f.direction==="TOXIC"?"#ff5c5c":"#00d084"};
                  padding:2px 8px;border-radius:4px;font-size:11px;">
                  ${f.direction==="TOXIC"?"↑ TOXIC":"↓ SAFE"}
                </span>
              </td>
              <td style="padding:7px 8px;color:#8892b0;">${f.impact.toFixed(1)}%</td>
            </tr>`).join("")}
        </tbody>
      </table>` : `<p style="color:#8892b0;">No SHAP feature data available.</p>`;

    content.style.display = "block";

  } catch (e) {
    loading.style.display = "none";
    errBox.textContent    = "⚠ Connection error: " + e.message;
    errBox.style.display  = "block";
  }
}

function showTab(name, el) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  el.classList.add("active");
  document.getElementById("tab-" + name).classList.add("active");
}

async function lookupSmiles() {
  const drugName = document.getElementById("drug_name").value.trim();
  const status   = document.getElementById("lookup_status");
  const btn      = document.getElementById("lookupBtn");

  if (!drugName) {
    status.style.color = "#ff5c5c";
    status.textContent = "Enter a drug name first.";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Looking up...";
  status.style.color = "#8892b0";
  status.textContent = `Searching PubChem for "${drugName}"...`;

  try {
    const res  = await fetch("/lookup-smiles?drug_name=" + encodeURIComponent(drugName));
    const data = await res.json();

    if (!res.ok) {
      status.style.color = "#ff5c5c";
      status.textContent = data.error || "Lookup failed.";
    } else {
      document.getElementById("smiles").value = data.smiles;
      status.style.color = "#00d084";
      status.textContent = `✓ Found via PubChem`;
    }
  } catch (e) {
    status.style.color = "#ff5c5c";
    status.textContent = "Connection error: " + e.message;
  }

  btn.disabled = false;
  btn.textContent = "🔍 Look up";
}

async function analyze() {
  const btn = document.getElementById("analyzeBtn");
  btn.disabled = true;
  document.getElementById("loading").style.display = "block";
  document.getElementById("result").style.display  = "none";

  const payload = {
    smiles    : document.getElementById("smiles").value,
    drug_name : document.getElementById("drug_name").value,
    co_drug   : document.getElementById("co_drug").value,
    age       : parseFloat(document.getElementById("age").value),
    weight    : parseFloat(document.getElementById("weight").value),
    egfr      : parseFloat(document.getElementById("egfr").value),
    alt       : parseFloat(document.getElementById("alt").value),
    ast       : parseFloat(document.getElementById("ast").value),
    gender    : document.getElementById("gender").value,
    conditions: document.getElementById("conditions").value,
  };

  try {
    const res  = await fetch("/analyze", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) { alert("Error: " + data.error); }
    else {
      // Store result for the AI explanation engine to use
      _lastAnalysisData = data;

      // Reset AI explanation tab so it's fresh for this new drug
      document.getElementById("aiexplain_empty").style.display   = "none";
      document.getElementById("aiexplain_content").style.display = "none";
      document.getElementById("aiexplain_error").style.display   = "none";
      document.getElementById("aiexplain_loading").style.display = "none";

      renderResult(data);

      // Auto-generate explanation in the background so it's ready when the user clicks the tab
      generateExplanation();
    }
  } catch(e) { alert("Connection error: " + e.message); }

  document.getElementById("loading").style.display = "none";
  btn.disabled = false;
}

function renderResult(d) {
  document.getElementById("m_readiness").textContent = d.readiness.readiness_score + "%";
  document.getElementById("m_risk").textContent      = d.unified_score + "%";
  document.getElementById("m_tox").textContent       = d.toxicity + "%";
  document.getElementById("m_vuln").textContent      = d.vulnerability + "%";

  // ---- Molecular structure ----
  const smilesUsed    = document.getElementById("smiles").value.trim();
  const structImg      = document.getElementById("structure_img");
  const structMissing  = document.getElementById("structure_missing");
  if (smilesUsed) {
    structImg.onload  = () => { structImg.style.display = "inline-block"; structMissing.style.display = "none"; };
    structImg.onerror = () => { structImg.style.display = "none";  structMissing.style.display = "block"; };
    structImg.src = "/molecule-image?smiles=" + encodeURIComponent(smilesUsed);
  } else {
    structImg.style.display     = "none";
    structMissing.style.display = "block";
  }

  const colorMap = {"Ready":"green","Promising":"blue","Needs Optimization":"yellow","High Risk":"red"};
  const color = colorMap[d.readiness.category] || "blue";
  document.getElementById("score_num").textContent = d.readiness.readiness_score;
  document.getElementById("score_ring").className  = "score-ring " + color;
  document.getElementById("category_badge").innerHTML =
    `<span class="badge badge-${color}">${d.readiness.category}</span>`;
  document.getElementById("category_desc").textContent = d.readiness.description;
  document.getElementById("dose_ratio").textContent = d.dosage.dose_ratio;
  document.getElementById("dose_rec").textContent   = d.dosage.recommendation;

  let riskHTML = "";
  for (const [k,v] of Object.entries(d.components)) {
    riskHTML += `<div class="progress-wrap">
      <div class="progress-label"><span>${k}</span><span>${v}%</span></div>
      <div class="progress-track"><div class="progress-fill danger" style="width:${Math.min(v,100)}%"></div></div>
    </div>`;
  }
  document.getElementById("risk_components").innerHTML = riskHTML;

  const ddiColor = d.ddi_level==="Major"?"#ff5c5c":d.ddi_level==="Moderate"?"#f5c542":
                   d.ddi_level==="Minor"?"#00c9ff":"#8892b0";
  document.getElementById("ddi_status").innerHTML =
    `<span style="color:${ddiColor};font-weight:600;">${d.ddi_level}</span>
     &nbsp;—&nbsp; ${d.drug_name} + ${d.co_drug}`;

  let subHTML = "";
  for (const [k,v] of Object.entries(d.sub_scores)) {
    subHTML += `<div class="progress-wrap">
      <div class="progress-label"><span>${k}</span><span>${v}%</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:${Math.min(v,100)}%"></div></div>
    </div>`;
  }
  document.getElementById("sub_scores").innerHTML = subHTML;

  // ---- SHAP explanation section ----
  function shapRowHTML(f) {
    const isToxic = f.direction === "TOXIC";
    return `<div class="shap-feature">
      <span class="shap-name">${f.feature}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar-fill ${isToxic?"shap-toxic":"shap-safe"}" style="width:${Math.min(f.impact,100)}%"></div>
      </div>
      <span class="shap-val">${f.shap_val>0?"+":""}${f.shap_val.toFixed(4)}</span>
      <span class="shap-dir ${isToxic?"toxic":"safe"}">${isToxic?"↑ TOXIC":"↓ SAFE"}</span>
    </div>`;
  }

  const shapFeatures  = d.shap_features || [];
  const increasing    = shapFeatures.filter(f => f.direction === "TOXIC");
  const decreasing    = shapFeatures.filter(f => f.direction === "SAFE");
  const hasFeatures   = shapFeatures.length > 0;
  const hasShapImage  = !!d.shap_image_url;

  const shapNoData  = document.getElementById("shap_no_data");
  const shapContent = document.getElementById("shap_content");

  if (!hasFeatures && !hasShapImage) {
    // Neither a live feature breakdown nor a saved plot exists for
    // this drug — show the single fallback message the spec requires.
    shapNoData.style.display  = "block";
    shapContent.style.display = "none";
  } else {
    shapNoData.style.display  = "none";
    shapContent.style.display = "block";

    document.getElementById("shap_increasing").innerHTML = increasing.length
      ? increasing.map(shapRowHTML).join("")
      : `<p style="font-size:12px;color:#8892b0;">No risk-increasing features identified.</p>`;

    document.getElementById("shap_decreasing").innerHTML = decreasing.length
      ? decreasing.map(shapRowHTML).join("")
      : `<p style="font-size:12px;color:#8892b0;">No risk-decreasing features identified.</p>`;

    const shapImg     = document.getElementById("shap_img");
    const shapImgMiss = document.getElementById("shap_img_missing");

    if (hasShapImage) {
      // onerror covers the edge case where the URL briefly 404s
      // (e.g. file removed between the /analyze call and the GET).
      shapImg.onload  = () => { shapImg.style.display = "block"; shapImgMiss.style.display = "none"; };
      shapImg.onerror = () => { shapImg.style.display = "none";  shapImgMiss.style.display = "block"; };
      shapImg.src = d.shap_image_url + "?t=" + Date.now(); // cache-bust per analysis
    } else {
      shapImg.style.display     = "none";
      shapImgMiss.style.display = "block";
    }
  }

  let recHTML = "";
  for (const r of d.recommendations) {
    recHTML += `<div class="rec-item">${r}</div>`;
  }
  document.getElementById("recommendations").innerHTML = recHTML;
  document.getElementById("result").style.display = "block";
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/shap-image/<path:drug_name>")
def shap_image(drug_name):
    """
    Serves the SHAP bar-chart PNG for a given drug as a real static
    file response (not embedded in JSON).

    Using send_from_directory():
      - streams the file instead of loading it fully into a JSON string
      - sets the correct Content-Type (image/png) automatically
      - adds Last-Modified / ETag headers so browsers can cache it
        and issue conditional GETs (304 Not Modified) on repeat loads
      - guards against path traversal (it verifies the resolved path
        stays inside the given directory)
    Returns 404 (instead of a Python exception) when no plot exists
    for the requested drug, which the frontend uses to show the
    "No explainability data available." message.
    """
    image_path = find_shap_image(drug_name)
    if image_path is None:
        abort(404, description="No explainability data available.")

    directory, filename = os.path.split(image_path)
    return send_from_directory(directory, filename, mimetype="image/png")


@app.route("/molecule-image")
def molecule_image():
    """
    Renders the 2D structure for whatever SMILES is passed as a query
    param, e.g. /molecule-image?smiles=CC(=O)Oc1ccccc1C(=O)O

    A query param (not a path segment) is used here because SMILES
    strings routinely contain characters like '/', '\\', '(', ')',
    '=', '#' that are awkward or unsafe in a URL path but are handled
    correctly by Flask/Werkzeug's query-string parsing and encoding.
    """
    smiles = request.args.get("smiles", "").strip()
    if not smiles:
        abort(400, description="Missing 'smiles' query parameter.")

    png_bytes = render_molecule_png(smiles)
    if png_bytes is None:
        abort(404, description="Invalid SMILES string — could not render structure.")

    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        max_age=3600,  # browser may cache for 1 hour; same SMILES always renders the same image
    )


@app.route("/explain", methods=["POST"])
def explain():
    """
    POST /explain   body: { drug_name, toxicity, readiness, unified_score,
                            ddi_level, shap_features, sub_scores, vulnerability }

    Calls Google Gemini 1.5 Flash (free tier) with the full analysis context
    and asks it to write a plain-English explanation of the prediction.

    We call the Gemini REST API directly using the requests library already
    installed — no extra SDK needed.
    """
    data = request.get_json(force=True)

    drug_name     = data.get("drug_name", "Unknown")
    toxicity      = data.get("toxicity", 0)
    readiness     = data.get("readiness", 0)
    unified       = data.get("unified_score", 0)
    ddi_level     = data.get("ddi_level", "None")
    vulnerability = data.get("vulnerability", 0)
    shap_features = data.get("shap_features", [])
    sub_scores    = data.get("sub_scores", {})

    # Format SHAP features into readable lines for the prompt
    increasing = [f for f in shap_features if f.get("direction") == "TOXIC"]
    decreasing = [f for f in shap_features if f.get("direction") == "SAFE"]

    def fmt_features(feats):
        if not feats:
            return "  None identified"
        return "\n".join(
            f"  - {f['feature']}: SHAP={f['shap_val']:+.4f} (impact={f['impact']:.1f}%)"
            for f in feats
        )

    prompt = f"""You are an expert medicinal chemist and clinical pharmacologist.
A drug safety AI (PharmaGuard AI) has just analyzed the drug "{drug_name}" using
machine learning models trained on the Tox21 toxicity dataset with SHAP explainability.

Here are the results:
- Toxicity Score: {toxicity}% (lower is safer)
- Readiness Score: {readiness}% (higher is better)
- Unified Risk Score: {unified}%
- Patient Vulnerability: {vulnerability}%
- DDI Level with co-administered drug: {ddi_level}
- Sub-scores: {sub_scores}

SHAP Feature Attribution (features pushing toward TOXIC):
{fmt_features(increasing)}

SHAP Feature Attribution (features pushing toward SAFE):
{fmt_features(decreasing)}

Write a clear, concise explanation (3-4 paragraphs) in plain English that:
1. Summarizes what the overall scores mean for this drug's safety profile
2. Explains which molecular features are most responsible for the toxicity prediction and WHY those features matter chemically
3. Interprets the DDI risk level and what it means clinically
4. Gives one specific, actionable insight a medicinal chemist could use to potentially improve the drug's safety profile

Write in a professional but accessible tone — as if explaining to a clinician who understands medicine but not machine learning.
Do not use bullet points. Write in flowing paragraphs.
Do not repeat the raw numbers verbatim — interpret what they mean."""

    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY is not set. Run: $env:GROQ_API_KEY = 'gsk_...' then restart."}), 500

    # Groq API — free tier, runs Llama-3.3-70b-versatile
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type" : "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model"      : "llama-3.3-70b-versatile",
                "max_tokens" : 1024,
                "temperature": 0.7,
                "messages"   : [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        if resp.status_code != 200:
            return jsonify({"error": f"Groq API error {resp.status_code}: {resp.text[:300]}"}), 502

        explanation = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"explanation": explanation})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Gemini API did not respond in time. Try again."}), 504
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route("/lookup-smiles")
def lookup_smiles():
    """
    GET /lookup-smiles?drug_name=pantoprazole
    -> {"drug_name": "pantoprazole", "smiles": "COC1=C(..."}

    Used by the "Look up SMILES" button so the dashboard can be driven
    by drug name alone, instead of requiring the user to already know
    the SMILES string for whatever they type into Drug Name.
    """
    drug_name = request.args.get("drug_name", "").strip()
    if not drug_name:
        return jsonify({"error": "Missing 'drug_name' query parameter."}), 400

    smiles, error = lookup_smiles_from_pubchem(drug_name)
    if smiles is None:
        return jsonify({"error": error}), 404

    return jsonify({"drug_name": drug_name, "smiles": smiles})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data  = request.get_json()
        conds = [c.strip() for c in data.get("conditions","").split(",") if c.strip()]

        patient = create_patient_profile(
            age=float(data["age"]), weight_kg=float(data["weight"]),
            egfr=float(data["egfr"]), alt=float(data["alt"]),
            ast=float(data["ast"]), gender=data.get("gender","male"),
            conditions=conds
        )

        report = generate_readiness_report(
            smiles=data["smiles"], drug_name=data["drug_name"],
            co_drug=data["co_drug"], patient=patient
        )

        # --------------------------------------------------------
        # SHAP explainability
        # --------------------------------------------------------
        # quick_shap_explain() recomputes live SHAP values, which
        # requires trained model files (models/*.pkl) and background
        # data (datasets/*.npy). Those are large, environment-specific
        # artifacts that aren't always present (e.g. fresh checkout
        # before training has been run). We don't want a missing
        # model file to crash the whole /analyze request and hide the
        # toxicity/readiness results the user DOES have — so this is
        # wrapped defensively and degrades to an empty feature list.
        try:
            shap_features = quick_shap_explain(data["smiles"], data["drug_name"], top_n=10)
        except Exception as shap_err:
            print(f"[WARN] Live SHAP computation unavailable: {shap_err}")
            shap_features = []

        # Instead of reading the PNG and embedding it as base64 here,
        # we just check whether a plot exists and hand back the URL
        # of the dedicated /shap-image/<drug_name> route. The browser
        # fetches the image itself via <img src="...">, which is
        # smaller over the wire, cacheable, and keeps this JSON
        # response focused on data rather than binary payloads.
        shap_image_url = (
            f"/shap-image/{data['drug_name']}"
            if find_shap_image(data["drug_name"]) else None
        )

        return jsonify({
            "readiness"      : report["readiness"],
            "unified_score"  : report["unified"],
            "toxicity"       : round(report["full_report"]["toxicity"]["score"]*100, 2),
            "vulnerability"  : report["full_report"]["patient"]["vulnerability"]["vulnerability_score"],
            "sub_scores"     : report["sub_scores"],
            "components"     : report["full_report"]["unified"]["components"],
            "dosage"         : report["dosage"],
            "recommendations": report["recommendations"],
            "ddi_level"      : report["full_report"]["ddi"]["level"],
            "drug_name"      : data["drug_name"],
            "co_drug"        : data["co_drug"],
            "shap_features"  : shap_features,
            "shap_image_url" : shap_image_url,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("="*60)
    print("PHARMAGUARD AI — Layer 8 Dashboard (with SHAP)")
    print("="*60)
    print("\n  Open browser → http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, port=5000)