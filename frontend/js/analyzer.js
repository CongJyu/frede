/* Analyzer view — predict + explain a single review (notebook Phase 4 Part 1). */

const SAMPLES = [
  "This is the best restaurant ever! Amazing food, incredible service, perfect atmosphere. Everyone should come here immediately! Trust me, you won't regret it. Five stars!",
  "We visited on a Saturday evening and waited about 20 minutes for a table. The pad thai was decent but a bit too sweet for my taste. Service was friendly. Prices are reasonable for the neighborhood. Would come back to try other dishes.",
  "WORST EXPERIENCE EVER. The food was absolutely terrible, the waiter was incredibly rude, and the place was filthy. DO NOT go here. I would give zero stars if I could. Never coming back.",
  "Absolutely phenomenal! The chef is a genius. Every single dish was outstanding and flawless. Best meal of my entire life. Go now before it's too late! Highly recommend to everyone!",
];

const $ = (id) => document.getElementById(id);
const reviewInput = $("review");
const analyzeBtn = $("analyze");
const resultsBox = $("results");
const banner = $("health-banner");

/* ---- health gate ---- */
async function initHealth() {
  try {
    const h = await waitForModel(120000);
    banner.classList.add("hidden");
    analyzeBtn.disabled = false;
  } catch (err) {
    banner.classList.remove("hidden", "error");
    banner.classList.add("error");
    banner.textContent = "⚠️ " + err.message + " — check that training has been run (`make train`).";
  }
}

/* ---- sample buttons ---- */
function buildSampleButtons() {
  const wrap = $("sample-btns");
  SAMPLES.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = `Sample ${i + 1}: ${s.slice(0, 48)}…`;
    btn.addEventListener("click", () => { reviewInput.value = s; });
    wrap.appendChild(btn);
  });
}

/* ---- analyze flow ---- */
async function analyze() {
  const text = reviewInput.value.trim();
  if (!text) { reviewInput.focus(); return; }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="spinner"></span> Analyzing… (3–8 s)';
  try {
    const r = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    render(r);
  } catch (err) {
    resultsBox.classList.remove("hidden");
    resultsBox.innerHTML = `<div class="card"><p class="verdict" style="background:#fde8e8;color:#7f1d1d;font-size:16px;">Error: ${escapeHtml(err.message)}</p></div>`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "🔍 Analyze Review";
  }
}

function render(r) {
  const isFake = r.prediction === "FAKE";
  const badgeClass = isFake ? "badge-fake" : "badge-real";
  const verdict = isFake ? "⚠️ FAKE" : "✅ REAL";

  resultsBox.classList.remove("hidden");
  resultsBox.innerHTML = `
    <div class="card">
      <span class="verdict ${badgeClass}">${verdict}</span>
      <p class="muted" style="margin-top:8px;">
        Fake probability: <strong>${fmtPct(r.fake_prob)}</strong> &nbsp;·&nbsp;
        Confidence: <strong>${fmtPct(r.confidence)}</strong> &nbsp;·&nbsp;
        analyzed in <strong>${(r.elapsed_ms / 1000).toFixed(1)}s</strong>
      </p>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>Suspicious segments (LIME)</h3>
        ${r.highlighted_html}
      </div>
      <div class="card">
        <h3>Reason codes</h3>
        ${reasonCodesHtml(r.reason_codes)}
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>Feature importance (LIME)</h3>
        <div id="lime-chart"></div>
      </div>
      <div class="card">
        <h3>Integrated Gradients — tokens pushing toward FAKE</h3>
        <div class="tokens" id="ig-tokens"></div>
      </div>
    </div>

    <div class="card">
      <h3>Summary</h3>
      <p>${escapeHtml(r.summary)}</p>
    </div>
  `;

  limeBars($("lime-chart"), r.lime_features);
  $("ig-tokens").innerHTML = tokenSpans(r.ig_tokens, r.ig_attrs);
  $("ig-tokens").insertAdjacentHTML(
    "beforeend",
    '<p class="muted" style="margin-top:8px;">🔴 red = pushes toward FAKE · 🔵 blue = pushes toward REAL · hover for value</p>'
  );
}

/* ---- wire up ---- */
buildSampleButtons();
$("analyze").addEventListener("click", analyze);
$("clear").addEventListener("click", () => { reviewInput.value = ""; resultsBox.classList.add("hidden"); });
reviewInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
});
initHealth();
