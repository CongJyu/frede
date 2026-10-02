import { Card, Space, Tag, Typography } from 'antd';

const { Text } = Typography;

/* Renders [{ code, label, description, source, aligns, evidence }] as cards.

   `aligns` is false when the signal contradicts the verdict — a machine-typical
   feature in a review called HUMAN. Those are greyed rather than hidden: the
   signal is a fact about the text, and showing it as if it supported the
   verdict would be the misleading option. */
export default function ReasonCodes({ items }) {
  if (!items || items.length === 0) {
    return <Text type="secondary">No machine-authorship signals detected.</Text>;
  }
  return (
    <Space direction="vertical" size="small" style={{ width: '100%' }}>
      {items.map((rc) => {
        const supports = rc.aligns !== false;
        return (
          <Card
            key={rc.code}
            size="small"
            style={{
              borderLeft: `4px solid ${supports ? '#dc3232' : '#bfbfbf'}`,
              opacity: supports ? 1 : 0.72,
            }}
          >
            <Space size={8} wrap>
              <Tag color={supports ? 'red' : 'default'}>{rc.code}</Tag>
              <strong>{rc.label}</strong>
              {rc.source && <Tag>{rc.source}</Tag>}
              {!supports && <Tag color="default">present but outweighed</Tag>}
            </Space>
            <div style={{ marginTop: 4 }}>{rc.description}</div>
            {rc.evidence && rc.evidence.length > 0 && (
              <div style={{ color: '#6b7280', fontSize: 13, marginTop: 4 }}>
                Evidence: {rc.evidence.join(', ')}
              </div>
            )}
          </Card>
        );
      })}
    </Space>
  );
}
