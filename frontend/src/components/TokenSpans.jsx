/* Colored token spans for Integrated Gradients / SHAP token attributions. */
export default function TokenSpans({ tokens, attrs, hint = true }) {
  if (!tokens || !attrs || tokens.length === 0) {
    return <span className="muted">No attributions.</span>;
  }
  const maxAbs = Math.max(...attrs.map(Math.abs), 1e-9);
  return (
    <div>
      <div className="tokens">
        {tokens.map((tok, i) => {
          const t = attrs[i] / maxAbs; // -1..1
          const bg =
            t >= 0
              ? `rgba(220, 50, 50, ${0.15 + 0.85 * t})`
              : `rgba(50, 130, 220, ${0.15 + 0.85 * -t})`;
          const display = String(tok).replace(/Ġ|▁/g, ' ') || '·';
          return (
            <span
              key={i}
              className="ig-token"
              style={{ background: bg }}
              title={`attr: ${attrs[i].toFixed(4)}`}
            >
              {display}
            </span>
          );
        })}
      </div>
      {hint && (
        <div className="muted" style={{ marginTop: 8 }}>
          red = pushes toward FAKE · blue = pushes toward REAL · hover for value
        </div>
      )}
    </div>
  );
}
