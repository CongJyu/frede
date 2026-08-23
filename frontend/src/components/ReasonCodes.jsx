import { Card, Space, Tag, Typography } from 'antd';

const { Text } = Typography;

/* Renders [{ code, label, description, evidence }] as small red-accented cards. */
export default function ReasonCodes({ items }) {
  if (!items || items.length === 0) {
    return <Text type="secondary">No strong fake-review signals detected.</Text>;
  }
  return (
    <Space direction="vertical" size="small" style={{ width: '100%' }}>
      {items.map((rc) => (
        <Card
          key={rc.code}
          size="small"
          style={{ borderLeft: '4px solid #dc3232' }}
        >
          <Space size={8}>
            <Tag color="red">{rc.code}</Tag>
            <strong>{rc.label}</strong>
          </Space>
          <div style={{ marginTop: 4 }}>{rc.description}</div>
          {rc.evidence && rc.evidence.length > 0 && (
            <div style={{ color: '#6b7280', fontSize: 13, marginTop: 4 }}>
              Evidence: {rc.evidence.join(', ')}
            </div>
          )}
        </Card>
      ))}
    </Space>
  );
}
