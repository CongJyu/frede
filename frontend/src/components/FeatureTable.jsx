import { Table, Tag, Tooltip, Typography } from 'antd';

const { Text } = Typography;

/* Feature contributions for one review.

   Each row carries three things, and all three are needed to read it: the raw
   value, where that value sits in the distribution of *real* reviews, and how
   much it moved this particular decision (SHAP). A feature can be far outside
   the human range and still not matter for the verdict, and vice versa — the
   percentile and the SHAP bar answer different questions. */
export default function FeatureTable({ items }) {
  if (!items || items.length === 0) {
    return <Text type="secondary">No features to display.</Text>;
  }
  const maxAbs = Math.max(...items.map((it) => Math.abs(it.shap)), 1e-9);

  const columns = [
    {
      title: 'Feature',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <Text code>{name}</Text>,
    },
    {
      title: 'Value',
      dataIndex: 'value',
      key: 'value',
      align: 'right',
      render: (v) => <Text>{Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1)}</Text>,
    },
    {
      title: (
        <Tooltip title="Where this value falls among real human reviews. 50 = typical; 99 = more extreme than 99% of real reviews.">
          <span style={{ borderBottom: '1px dotted #999', cursor: 'help' }}>
            Human pct
          </span>
        </Tooltip>
      ),
      dataIndex: 'human_percentile',
      key: 'pct',
      align: 'right',
      render: (pct) => {
        if (pct == null) return <Text type="secondary">—</Text>;
        const extreme = pct >= 95 || pct <= 5;
        return (
          <Tag color={extreme ? 'orange' : 'default'} style={{ marginRight: 0 }}>
            {pct <= 1 ? '≤1' : pct >= 99 ? '≥99' : Math.round(pct)}
          </Tag>
        );
      },
    },
    {
      title: (
        <Tooltip title="How much this feature moved the verdict. Positive pushes toward MACHINE, negative toward HUMAN.">
          <span style={{ borderBottom: '1px dotted #999', cursor: 'help' }}>Push</span>
        </Tooltip>
      ),
      dataIndex: 'shap',
      key: 'shap',
      width: 190,
      render: (v) => {
        const pct = (Math.abs(v) / maxAbs) * 100;
        const color = v > 0 ? '#dc3232' : '#3282dc';
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="bar-track" style={{ flex: 1 }}>
              <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
            </div>
            <span className="bar-val">{v.toFixed(2)}</span>
          </div>
        );
      },
    },
  ];

  return (
    <Table
      size="small"
      rowKey="name"
      columns={columns}
      dataSource={items}
      pagination={items.length > 12 ? { pageSize: 12, size: 'small' } : false}
    />
  );
}
