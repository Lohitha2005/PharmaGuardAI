# ============================================================
# PHARMAGUARD AI — Layer 8 (Updated)
# PharmaGuard Dashboard with SHAP Integration
# ============================================================

from flask import Flask, render_template_string, request, jsonify
import base64
import os
import warnings
warnings.filterwarnings("ignore")

from layer5_patient_risk    import create_patient_profile
from layer7_readiness_score import generate_readiness_report
from module3_shap_explainability import quick_shap_explain

app = Flask(__name__)

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
        <input id="smiles" type="text" value="CC(=O)Oc1ccccc1C(=O)O">
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
        <h3>🔬 SHAP Feature Attribution</h3>
        <p style="font-size:12px;color:#8892b0;margin-bottom:16px;">
          Which molecular features push this drug toward TOXIC or SAFE?
          Based on SR-ARE toxicity label using SHAP (SHapley Additive exPlanations).
        </p>
        <div id="shap_features"></div>
        <h3 style="margin-top:20px;">📊 SHAP Bar Chart</h3>
        <img id="shap_img" class="shap-img" src="" alt="SHAP Bar Chart" style="display:none;">
      </div>
    </div>

    <!-- Recommendations Tab -->
    <div class="tab-content" id="tab-recs">
      <div class="card">
        <h3>🩺 Clinical Recommendations</h3>
        <div id="recommendations"></div>
      </div>
    </div>

  </div>
</div>

<script>
function showTab(name, el) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  el.classList.add("active");
  document.getElementById("tab-" + name).classList.add("active");
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
    else { renderResult(data); }
  } catch(e) { alert("Connection error: " + e.message); }

  document.getElementById("loading").style.display = "none";
  btn.disabled = false;
}

function renderResult(d) {
  document.getElementById("m_readiness").textContent = d.readiness.readiness_score + "%";
  document.getElementById("m_risk").textContent      = d.unified_score + "%";
  document.getElementById("m_tox").textContent       = d.toxicity + "%";
  document.getElementById("m_vuln").textContent      = d.vulnerability + "%";

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

  let shapHTML = "";
  const maxImp = Math.max(...d.shap_features.map(f => f.impact), 1);
  for (const f of d.shap_features) {
    const pct      = (f.impact / maxImp * 100).toFixed(1);
    const isToxic  = f.direction === "TOXIC";
    shapHTML += `<div class="shap-feature">
      <span class="shap-name">${f.feature}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar-fill ${isToxic?"shap-toxic":"shap-safe"}" style="width:${pct}%"></div>
      </div>
      <span class="shap-val">${f.shap_val>0?"+":""}${f.shap_val.toFixed(4)}</span>
      <span class="shap-dir ${isToxic?"toxic":"safe"}">${isToxic?"↑ TOXIC":"↓ SAFE"}</span>
    </div>`;
  }
  document.getElementById("shap_features").innerHTML = shapHTML;

  if (d.shap_image) {
    const img = document.getElementById("shap_img");
    img.src   = "data:image/png;base64," + d.shap_image;
    img.style.display = "block";
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

        shap_features = quick_shap_explain(data["smiles"], data["drug_name"], top_n=10)

        shap_image = None
        img_path   = f"shap_plots/{data['drug_name']}_SR-ARE_shap.png"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                shap_image = base64.b64encode(f.read()).decode("utf-8")

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
            "shap_image"     : shap_image,
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