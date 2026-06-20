# ============================================================
# PHARMAGUARD AI — Layer 8
# PharmaGuard Dashboard (Flask Web Application)
# ============================================================

from flask import Flask, render_template_string, request, jsonify
import warnings
warnings.filterwarnings("ignore")

from layer5_patient_risk import create_patient_profile
from layer7_readiness_score import generate_readiness_report

app = Flask(__name__)

# ============================================================
# HTML TEMPLATE
# ============================================================

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

  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #00c9ff, #0077b6);
    padding: 24px 40px;
    display: flex; align-items: center; gap: 16px;
  }
  .header h1 { font-size:28px; font-weight:700; color:#fff; letter-spacing:1px; }
  .header p  { font-size:13px; color:#d0f0ff; margin-top:4px; }
  .logo { font-size:36px; }

  /* ── Layout ── */
  .container { max-width:1200px; margin:0 auto; padding:32px 24px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  .grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:24px; }

  /* ── Cards ── */
  .card {
    background:#1e2130; border-radius:12px;
    padding:24px; border:1px solid #2a2f45;
  }
  .card h3 { font-size:14px; color:#7b8ab8; text-transform:uppercase;
             letter-spacing:1px; margin-bottom:16px; }

  /* ── Form ── */
  .form-group { margin-bottom:14px; }
  label { display:block; font-size:12px; color:#8892b0; margin-bottom:6px; }
  input, select {
    width:100%; padding:10px 14px; background:#151827;
    border:1px solid #2a2f45; border-radius:8px;
    color:#e0e0e0; font-size:14px; outline:none;
    transition: border 0.2s;
  }
  input:focus, select:focus { border-color:#00c9ff; }
  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }

  /* ── Button ── */
  .btn {
    width:100%; padding:14px; background:linear-gradient(135deg,#00c9ff,#0077b6);
    border:none; border-radius:8px; color:#fff; font-size:16px;
    font-weight:600; cursor:pointer; margin-top:8px; letter-spacing:0.5px;
    transition: opacity 0.2s;
  }
  .btn:hover { opacity:0.9; }
  .btn:disabled { opacity:0.5; cursor:not-allowed; }

  /* ── Score Ring ── */
  .score-ring {
    width:140px; height:140px; border-radius:50%;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    margin:0 auto 20px;
    border:6px solid #2a2f45;
    position:relative;
  }
  .score-number { font-size:32px; font-weight:700; }
  .score-label  { font-size:12px; color:#8892b0; margin-top:2px; }

  .green  { border-color:#00d084; color:#00d084; }
  .blue   { border-color:#00c9ff; color:#00c9ff; }
  .yellow { border-color:#f5c542; color:#f5c542; }
  .red    { border-color:#ff5c5c; color:#ff5c5c; }

  /* ── Progress Bar ── */
  .progress-wrap { margin-bottom:12px; }
  .progress-label {
    display:flex; justify-content:space-between;
    font-size:12px; color:#8892b0; margin-bottom:4px;
  }
  .progress-track {
    height:8px; background:#151827; border-radius:4px; overflow:hidden;
  }
  .progress-fill {
    height:100%; border-radius:4px;
    background:linear-gradient(90deg,#00c9ff,#0077b6);
    transition: width 0.8s ease;
  }

  /* ── Status Badge ── */
  .badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:12px; font-weight:600; margin-bottom:12px;
  }
  .badge-green  { background:#00d08422; color:#00d084; border:1px solid #00d084; }
  .badge-blue   { background:#00c9ff22; color:#00c9ff; border:1px solid #00c9ff; }
  .badge-yellow { background:#f5c54222; color:#f5c542; border:1px solid #f5c542; }
  .badge-red    { background:#ff5c5c22; color:#ff5c5c; border:1px solid #ff5c5c; }

  /* ── Recommendations ── */
  .rec-item {
    padding:10px 14px; margin-bottom:8px;
    background:#151827; border-radius:8px;
    font-size:13px; line-height:1.5;
    border-left:3px solid #2a2f45;
  }

  /* ── Metric Cards ── */
  .metric {
    background:#151827; border-radius:10px;
    padding:16px; text-align:center;
    border:1px solid #2a2f45;
  }
  .metric-value { font-size:24px; font-weight:700; color:#00c9ff; }
  .metric-label { font-size:11px; color:#8892b0; margin-top:4px; }

  /* ── Loading ── */
  .loading {
    display:none; text-align:center;
    padding:40px; color:#8892b0;
  }
  .spinner {
    width:40px; height:40px; border:4px solid #2a2f45;
    border-top:4px solid #00c9ff; border-radius:50%;
    animation:spin 0.8s linear infinite; margin:0 auto 16px;
  }
  @keyframes spin { to { transform:rotate(360deg); } }

  /* ── Result Panel ── */
  #result { display:none; margin-top:24px; }
  .section-title {
    font-size:18px; font-weight:600; color:#e0e0e0;
    margin-bottom:16px; padding-bottom:8px;
    border-bottom:1px solid #2a2f45;
  }
  .dose-box {
    background:#151827; border-radius:10px;
    padding:16px; border-left:4px solid #00c9ff;
    margin-top:16px;
  }
  .dose-box .ratio { font-size:22px; font-weight:700; color:#00c9ff; }
  .dose-box .rec   { font-size:13px; color:#8892b0; margin-top:4px; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="logo">🧬</div>
  <div>
    <h1>PharmaGuard AI</h1>
    <p>Explainable Drug Safety &amp; Compatibility Intelligence Platform</p>
  </div>
</div>

<div class="container">

  <!-- Input Form -->
  <div class="grid2">

    <!-- Drug Info -->
    <div class="card">
      <h3>💊 Drug Information</h3>
      <div class="form-group">
        <label>Drug Name</label>
        <input id="drug_name" type="text" placeholder="e.g. Aspirin" value="Aspirin">
      </div>
      <div class="form-group">
        <label>Drug SMILES</label>
        <input id="smiles" type="text" placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
               value="CC(=O)Oc1ccccc1C(=O)O">
      </div>
      <div class="form-group">
        <label>Co-administered Drug (for DDI check)</label>
        <input id="co_drug" type="text" placeholder="e.g. Warfarin" value="Warfarin">
      </div>
    </div>

    <!-- Patient Info -->
    <div class="card">
      <h3>👤 Patient Profile</h3>
      <div class="row2">
        <div class="form-group">
          <label>Age (years)</label>
          <input id="age" type="number" placeholder="e.g. 45" value="45">
        </div>
        <div class="form-group">
          <label>Weight (kg)</label>
          <input id="weight" type="number" placeholder="e.g. 70" value="70">
        </div>
      </div>
      <div class="row2">
        <div class="form-group">
          <label>eGFR — Kidney (ml/min)</label>
          <input id="egfr" type="number" placeholder="e.g. 75" value="75">
        </div>
        <div class="form-group">
          <label>Gender</label>
          <select id="gender">
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
      </div>
      <div class="row2">
        <div class="form-group">
          <label>ALT — Liver (U/L)</label>
          <input id="alt" type="number" placeholder="e.g. 35" value="35">
        </div>
        <div class="form-group">
          <label>AST — Liver (U/L)</label>
          <input id="ast" type="number" placeholder="e.g. 30" value="30">
        </div>
      </div>
      <div class="form-group">
        <label>Conditions (comma-separated)</label>
        <input id="conditions" type="text" placeholder="e.g. diabetes, hypertension">
      </div>
    </div>
  </div>

  <!-- Analyze Button -->
  <button class="btn" onclick="analyze()" id="analyzeBtn">
    🔍 Analyze Drug Safety
  </button>

  <!-- Loading -->
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>Running PharmaGuard AI analysis...</p>
  </div>

  <!-- Results -->
  <div id="result">

    <!-- Top Metrics -->
    <div class="grid3">
      <div class="metric">
        <div class="metric-value" id="m_readiness">—</div>
        <div class="metric-label">Readiness Score</div>
      </div>
      <div class="metric">
        <div class="metric-value" id="m_unified">—</div>
        <div class="metric-label">Risk Score</div>
      </div>
      <div class="metric">
        <div class="metric-value" id="m_vuln">—</div>
        <div class="metric-label">Patient Vulnerability</div>
      </div>
    </div>

    <div class="grid2" style="margin-top:24px;">

      <!-- Readiness Score -->
      <div class="card" style="text-align:center;">
        <h3>📊 Development Readiness</h3>
        <div class="score-ring" id="score_ring">
          <span class="score-number" id="score_num">—</span>
          <span class="score-label">/ 100</span>
        </div>
        <div id="category_badge"></div>
        <p id="category_desc" style="font-size:13px;color:#8892b0;"></p>

        <!-- Dose Box -->
        <div class="dose-box">
          <div class="ratio" id="dose_ratio">—</div>
          <div class="rec"   id="dose_rec">Dosage Recommendation</div>
        </div>
      </div>

      <!-- Sub Scores -->
      <div class="card">
        <h3>📈 Score Breakdown</h3>
        <div id="sub_scores"></div>

        <h3 style="margin-top:20px;">⚡ Risk Components</h3>
        <div id="risk_components"></div>
      </div>
    </div>

    <!-- Recommendations -->
    <div class="card" style="margin-top:24px;">
      <h3>🩺 Clinical Recommendations</h3>
      <div id="recommendations"></div>
    </div>

  </div>
</div>

<script>
async function analyze() {
  const btn = document.getElementById("analyzeBtn");
  btn.disabled = true;
  document.getElementById("loading").style.display = "block";
  document.getElementById("result").style.display  = "none";

  const payload = {
    smiles     : document.getElementById("smiles").value,
    drug_name  : document.getElementById("drug_name").value,
    co_drug    : document.getElementById("co_drug").value,
    age        : parseFloat(document.getElementById("age").value),
    weight     : parseFloat(document.getElementById("weight").value),
    egfr       : parseFloat(document.getElementById("egfr").value),
    alt        : parseFloat(document.getElementById("alt").value),
    ast        : parseFloat(document.getElementById("ast").value),
    gender     : document.getElementById("gender").value,
    conditions : document.getElementById("conditions").value,
  };

  try {
    const res  = await fetch("/analyze", {
      method : "POST",
      headers: {"Content-Type":"application/json"},
      body   : JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.error) {
      alert("Error: " + data.error);
    } else {
      renderResult(data);
    }
  } catch(e) {
    alert("Connection error: " + e.message);
  }

  document.getElementById("loading").style.display = "none";
  btn.disabled = false;
}

function renderResult(d) {
  // Top metrics
  document.getElementById("m_readiness").textContent = d.readiness.readiness_score + "%";
  document.getElementById("m_unified").textContent   = d.unified_score + "%";
  document.getElementById("m_vuln").textContent      = d.vulnerability + "%";

  // Score ring
  const color = {
    "Ready":"green","Promising":"blue",
    "Needs Optimization":"yellow","High Risk":"red"
  }[d.readiness.category] || "blue";

  document.getElementById("score_num").textContent  = d.readiness.readiness_score;
  const ring = document.getElementById("score_ring");
  ring.className = "score-ring " + color;

  // Badge
  document.getElementById("category_badge").innerHTML =
    `<span class="badge badge-${color}">${d.readiness.category}</span>`;
  document.getElementById("category_desc").textContent = d.readiness.description;

  // Dose
  document.getElementById("dose_ratio").textContent = d.dosage.dose_ratio;
  document.getElementById("dose_rec").textContent   = d.dosage.recommendation;

  // Sub scores
  let subHTML = "";
  for (const [k,v] of Object.entries(d.sub_scores)) {
    subHTML += `
      <div class="progress-wrap">
        <div class="progress-label"><span>${k}</span><span>${v}%</span></div>
        <div class="progress-track">
          <div class="progress-fill" style="width:${v}%"></div>
        </div>
      </div>`;
  }
  document.getElementById("sub_scores").innerHTML = subHTML;

  // Risk components
  let riskHTML = "";
  for (const [k,v] of Object.entries(d.components)) {
    riskHTML += `
      <div class="progress-wrap">
        <div class="progress-label"><span>${k}</span><span>${v}%</span></div>
        <div class="progress-track">
          <div class="progress-fill" style="width:${v}%;
            background:linear-gradient(90deg,#ff5c5c,#ff8c42)"></div>
        </div>
      </div>`;
  }
  document.getElementById("risk_components").innerHTML = riskHTML;

  // Recommendations
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

# ============================================================
# API ROUTE — /analyze
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        # Parse conditions
        conds = [c.strip() for c in data.get("conditions","").split(",") if c.strip()]

        # Build patient profile
        patient = create_patient_profile(
            age        = float(data["age"]),
            weight_kg  = float(data["weight"]),
            egfr       = float(data["egfr"]),
            alt        = float(data["alt"]),
            ast        = float(data["ast"]),
            gender     = data.get("gender","male"),
            conditions = conds
        )

        # Run full pipeline
        report = generate_readiness_report(
            smiles    = data["smiles"],
            drug_name = data["drug_name"],
            co_drug   = data["co_drug"],
            patient   = patient
        )

        # Build response
        return jsonify({
            "readiness"    : report["readiness"],
            "unified_score": report["unified"],
            "vulnerability": report["full_report"]["patient"]["vulnerability"]["vulnerability_score"],
            "sub_scores"   : report["sub_scores"],
            "components"   : report["full_report"]["unified"]["components"],
            "dosage"       : report["dosage"],
            "recommendations": report["recommendations"],
            "ddi_level"    : report["full_report"]["ddi"]["level"],
            "toxicity"     : round(report["full_report"]["toxicity"]["score"] * 100, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHARMAGUARD AI — Layer 8: Dashboard")
    print("=" * 60)
    print("\n  Starting PharmaGuard AI Dashboard...")
    print("  Open your browser and go to:")
    print("  → http://127.0.0.1:5000")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, port=5000)