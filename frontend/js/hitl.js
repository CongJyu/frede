/* HITL view — two-pass evaluation flow (notebook Phase 4 Part 2). */

const $ = (id) => document.getElementById(id);
const reviewPanel = $("review-panel");
const controls = $("controls");
const resultsPanel = $("results-panel");
const errorBanner = $("error-banner");
const reviewText = $("review-text");
const highlighted = $("highlighted");
const reasonCodesBox = $("reason-codes");
const xaiBox = $("xai-box");
const progress = $("progress");
const modeBadge = $("mode-badge");
const status = $("status");
const feedbackInput = $("feedback");
const confidenceInput = $("confidence");
const confidenceVal = $("confidence-val");
const judgmentSelect = $("judgment");

function showError(msg) {
  errorBanner.classList.remove("hidden");
  errorBanner.textContent = "⚠️ " + msg;
}

function clearError() { errorBanner.classList.add("hidden"); }

confidenceInput.addEventListener("input", () => { confidenceVal.textContent = confidenceInput.value; });

/* ---- rendering ---- */
function renderSession(s) {
  if (s.done) { showResults(); return; }
  clearError();
  reviewPanel.classList.remove("hidden");
  controls.classList.remove("hidden");
  resultsPanel.classList.add("hidden");

  progress.textContent = `Pass ${s.pass_number} of 2 · Review ${s.in_pass_idx + 1}/${s.total}`;
  modeBadge.textContent = s.pass_label;
  modeBadge.className = "badge " + (s.mode === "no_xai" ? "badge-neutral" : "badge-xai");

  reviewText.textContent = s.sample.text;

  const showXai = !!s.sample.highlighted_html;
  xaiBox.classList.toggle("hidden", !showXai);
  if (showXai) {
    highlighted.innerHTML = s.sample.highlighted_html;
    reasonCodesBox.innerHTML = reasonCodesHtml(s.sample.reason_codes);
  }
  status.textContent = s.status || "";
  feedbackInput.value = "";
  confidenceInput.value = 3;
  confidenceVal.textContent = "3";
  judgmentSelect.value = "Real";
}

/* ---- results ---- */
async function showResults() {
  reviewPanel.classList.add("hidden");
  controls.classList.add("hidden");
  resultsPanel.classList.remove("hidden");

  const r = await api("/api/hitl/results");
  if (!r.mode_metrics || Object.keys(r.mode_metrics).length === 0) {
    resultsPanel.innerHTML = '<div class="card"><p>No results yet — complete the evaluation first.</p></div>';
    return;
  }

  const rows = Object.entries(r.mode_metrics).map(([mode, m]) => `
    <tr>
      <td><strong>${escapeHtml(m.label)}</strong></td>
      <td>${m.count}</td>
      <td>${fmtPct(m.human_accuracy)}</td>
      <td>${fmtPct(m.human_precision)}</td>
      <td>${fmtPct(m.human_recall)}</td>
      <td>${fmtPct(m.human_f1)}</td>
      <td>${m.kappa.toFixed(3)}</td>
      <td>${m.avg_time.toFixed(1)}s</td>
      <td>${m.avg_confidence.toFixed(1)}</td>
    </tr>`).join("");

  resultsPanel.innerHTML = `
    <div class="card">
      <h3>📊 Evaluation Results</h3>
      <p>${escapeHtml(r.summary)}</p>
      <table>
        <thead>
          <tr>
            <th>Mode</th><th>n</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th>
            <th>Cohen's κ<br>(human vs model)</th><th>Avg time</th><th>Avg conf</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Human accuracy vs model accuracy</h3>
      <div id="acc-chart"></div>
    </div>
    <div class="card">
      <h3>Human F1 (with vs without XAI)</h3>
      <div id="f1-chart"></div>
    </div>
  `;

  renderGroupedBars($("acc-chart"), r.mode_metrics, "human_accuracy", "model_accuracy");
  renderGroupedBars($("f1-chart"), r.mode_metrics, "human_f1");
}

/* Two grouped bars per mode (human + model, or just human). */
function renderGroupedBars(container, metrics, humanKey, modelKey) {
  const modes = Object.values(metrics);
  if (!modes.length) return;
  const html = modes.map((m) => {
    const color = m.label.includes("Without") ? "var(--real)" : "var(--fake)";
    const human = `<div class="bar-row">
        <span class="bar-label">${escapeHtml(m.label)} human</span>
        <div class="bar-track"><div class="bar-fill" style="width:${m[humanKey] * 100}%;background:${color}"></div></div>
        <span class="bar-val">${fmtPct(m[humanKey])}</span>
      </div>`;
    const model = modelKey ? `<div class="bar-row">
        <span class="bar-label">${escapeHtml(m.label)} model</span>
        <div class="bar-track"><div class="bar-fill" style="width:${m[modelKey] * 100}%;background:#9ca3af"></div></div>
        <span class="bar-val">${fmtPct(m[modelKey])}</span>
      </div>` : "";
    return human + model;
  }).join("");
  container.innerHTML = html;
}

/* ---- actions ---- */
async function submit() {
  const btn = $("submit");
  btn.disabled = true;
  try {
    const s = await api("/api/hitl/judgment", {
      method: "POST",
      body: JSON.stringify({
        judgment: judgmentSelect.value,
        confidence: parseInt(confidenceInput.value, 10),
        feedback: feedbackInput.value,
      }),
    });
    renderSession(s);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
  }
}

async function reset() {
  await api("/api/hitl/reset", { method: "POST" });
  renderSession(await api("/api/hitl/session"));
}

async function init() {
  try {
    await waitForModel(120000);
    renderSession(await api("/api/hitl/session"));
  } catch (err) {
    showError(err.message);
  }
}

$("submit").addEventListener("click", submit);
$("reset").addEventListener("click", reset);
init();
