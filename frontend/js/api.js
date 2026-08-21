/* Shared fetch helpers + small render utilities for the frede SPA. */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* Two-slope color: attr in [-1, 1] → blue (real-push) / red (fake-push). */
function attrColor(t) {
  if (t >= 0) return `rgba(220, 50, 50, ${0.15 + 0.85 * t})`;
  return `rgba(50, 130, 220, ${0.15 + 0.85 * -t})`;
}

/* Colored token spans for IG / SHAP token attributions. */
function tokenSpans(tokens, attrs) {
  if (!tokens || !attrs || !tokens.length) return '<span class="muted">No attributions.</span>';
  const maxAbs = Math.max(...attrs.map(Math.abs), 1e-9);
  return tokens
    .map((tok, i) => {
      const display = String(tok).replace(/Ġ|▁/g, " ") || "·";
      return `<span class="ig-token" style="background:${attrColor(attrs[i] / maxAbs)}" title="attr: ${attrs[i].toFixed(4)}">${escapeHtml(display)}</span>`;
    })
    .join("");
}

/* Signed horizontal bar chart from [{word, weight}]. */
function limeBars(container, features) {
  if (!features || !features.length) {
    container.innerHTML = '<span class="muted">No features to display.</span>';
    return;
  }
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.weight)), 1e-9);
  container.innerHTML = features
    .map((f) => {
      const pct = (Math.abs(f.weight) / maxAbs) * 100;
      const color = f.weight > 0 ? "var(--fake)" : "var(--real)";
      return (
        `<div class="bar-row">` +
        `<span class="bar-label" title="${escapeHtml(f.word)}">${escapeHtml(f.word)}</span>` +
        `<div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>` +
        `<span class="bar-val">${f.weight.toFixed(3)}</span>` +
        `</div>`
      );
    })
    .join("");
}

/* Reason-code cards from [{code, label, description, evidence}]. */
function reasonCodesHtml(items) {
  if (!items || !items.length) {
    return '<p class="muted">No strong fake-review signals detected.</p>';
  }
  return items
    .map((rc) => {
      const evidence = rc.evidence && rc.evidence.length
        ? `<p class="rc-evidence">Evidence: ${rc.evidence.map(escapeHtml).join(", ")}</p>`
        : "";
      return (
        `<div class="rc-card">` +
        `<div class="rc-head"><span class="rc-badge">${escapeHtml(rc.code)}</span><strong>${escapeHtml(rc.label)}</strong></div>` +
        `<p class="rc-desc">${escapeHtml(rc.description)}</p>${evidence}` +
        `</div>`
      );
    })
    .join("");
}

/* Poll /api/health until the model is ready. Returns a promise. */
async function waitForModel(timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const h = await api("/api/health");
      if (h.status === "ready") return h;
    } catch (_) { /* server may still be starting */ }
    await new Promise((r) => setTimeout(r, 1500));
  }
  throw new Error("Timed out waiting for the model to load.");
}
