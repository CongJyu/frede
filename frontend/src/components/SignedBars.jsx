import { Typography } from 'antd';

const { Text } = Typography;

/* Signed horizontal bar chart: items = [{ label, value }].
   Positive (fake-push) renders red, negative (real-push) renders blue. */
export default function SignedBars({ items }) {
  if (!items || items.length === 0) {
    return <Text type="secondary">No features to display.</Text>;
  }
  const maxAbs = Math.max(...items.map((it) => Math.abs(it.value)), 1e-9);
  return (
    <div>
      {items.map((it, i) => {
        const pct = (Math.abs(it.value) / maxAbs) * 100;
        const color = it.value > 0 ? '#dc3232' : '#3282dc';
        return (
          <div key={i} className="bar-row">
            <span className="bar-label" title={it.label}>
              {it.label}
            </span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
            </div>
            <span className="bar-val">{it.value.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}
