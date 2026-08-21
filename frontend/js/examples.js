/* Examples view — precomputed TP/TN/FP/FN reviews with LIME/IG/SHAP. */

async function loadExamples() {
  const main = document.getElementById("examples");
  const banner = document.getElementById("error-banner");
  try {
    const data = await api("/api/examples");
    banner.classList.add("hidden");
    main.innerHTML = data.examples
      .map((e, i) => exampleCard(e, i))
      .join("");
    // Populate bars/tokens after inserting the HTML.
    data.examples.forEach((e, i) => {
      limeBars(document.getElementById(`lime-${i}`), e.lime_features);
      document.getElementById(`ig-${i}`).innerHTML = tokenSpans(e.ig_tokens, e.ig_attrs);
      if (e.shap) {
        shapBars(document.getElementById(`shap-${i}`), e.shap.tokens, e.shap.values);
      }
    });
  } catch (err) {
    banner.classList.remove("hidden");
    banner.textContent = "⚠️ " + err.message;
  }
}

function exampleCard(e, i) {
  const isFake = e.model_pred === 1;
  const badge = isFake ? "badge-fake" : "badge-real";
  const pred = isFake ? "FAKE" : "REAL";
  const truth = e.true_label === 1 ? "FAKE" : "REAL";
  const shapBlock = e.shap
    ? `<h4>SHAP — token attribution (Fake class)</h4><div id="shap-${i}"></div>`
    : `<h4>SHAP</h4><p class="muted">SHAP not precomputed for this run (see `make train --with-shap`).</p>`;

  return `
    <div class="card example">
      <div class="row between">
        <h3 style="margin:0;">${escapeHtml(e.title)}</h3>
        <span class="badge ${badge}">Model: ${pred} · truth: ${truth}</span>
      </div>
      <p class="muted">P(Fake) = ${fmtPct(e.fake_prob)}</p>
      <p class="review-text">${escapeHtml(e.text)}</p>
      <div class="grid-2">
        <div>
          <h4>Reason codes</h4>
          ${reasonCodesHtml(e.reason_codes)}
        </div>
        <div>
          <h4>Highlighted text (LIME)</h4>
          ${e.highlighted_html}
        </div>
      </div>
      <div class="grid-2">
        <div>
          <h4>Feature importance (LIME)</h4>
          <div id="lime-${i}"></div>
        </div>
        <div>
          <h4>Integrated Gradients</h4>
          <div class="tokens" id="ig-${i}"></div>
        </div>
      </div>
      <div>
        ${shapBlock}
      </div>
    </div>`;
}

/* Vertical list of SHAP token contributions (signed bars). */
function shapBars(container, tokens, values) {
  const pairs = tokens
    .map((tok, j) => ({ tok, val: values[j] }))
    .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
    .slice(0, 20);
  const maxAbs = Math.max(...pairs.map((p) => Math.abs(p.val)), 1e-9);
  container.innerHTML = pairs
    .map((p) => {
      const pct = (Math.abs(p.val) / maxAbs) * 100;
      const color = p.val > 0 ? "var(--fake)" : "var(--real)";
      const display = String(p.tok).replace(/Ġ|▁/g, " ");
      return (
        `<div class="bar-row">` +
        `<span class="bar-label" title="${escapeHtml(p.tok)}">${escapeHtml(display) || "·"}</span>` +
        `<div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>` +
        `<span class="bar-val">${p.val.toFixed(3)}</span>` +
        `</div>`
      );
    })
    .join("");
}

loadExamples();
