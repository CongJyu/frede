import { useEffect, useState } from 'react';
import { Alert, Card, Col, Row, Space, Tag, Typography } from 'antd';
import { api, fmtPct } from '../api';
import FeatureTable from '../components/FeatureTable';
import ReasonCodes from '../components/ReasonCodes';

const { Title, Paragraph, Text } = Typography;

/* One worked case per quadrant of the confusion matrix, so the page shows the
   detector's failure modes and not only its successes. */
export default function Examples() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api('/api/examples')
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <Title level={2} style={{ marginTop: 4 }}>
        Worked examples
      </Title>
      <Paragraph type="secondary">
        One case from each corner of the confusion matrix, from the held-out split only. Correct
        calls show the most confident instance; errors show the one closest to the decision
        boundary, which is where the reasoning actually gives way.
      </Paragraph>

      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />}

      {data && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={`Decision threshold ${fmtPct(data.threshold)} — the cut calibrated to hold false positives on real reviews at 1%, measured over ${data.n_evaluated} held-out reviews.`}
        />
      )}

      {data &&
        data.missing_cases &&
        data.missing_cases.map((m) => (
          <Alert
            key={m.key}
            type="success"
            showIcon
            style={{ marginBottom: 16 }}
            message={`No example for "${m.title}"`}
            description="The detector produced no such case on the held-out split — at this operating point it flagged no real reviewer. That is the result, not a gap in the page."
          />
        ))}

      {data &&
        data.examples.map((e) => {
          const isMachine = e.model_pred === 1;
          const correct = e.model_pred === e.true_label;
          return (
            <Card
              key={e.key}
              style={{ marginBottom: 16 }}
              title={
                <Space style={{ width: '100%', justifyContent: 'space-between', display: 'flex' }}>
                  <span>{e.title}</span>
                  <Space>
                    <Tag color={correct ? 'green' : 'orange'}>
                      {correct ? 'model correct' : 'model error'}
                    </Tag>
                    <Tag color={isMachine ? 'red' : 'blue'}>
                      called {isMachine ? 'MACHINE' : 'HUMAN'}
                    </Tag>
                  </Space>
                </Space>
              }
            >
              <Text type="secondary">
                P(machine) = <Text strong>{fmtPct(e.machine_prob)}</Text> · truth:{' '}
                <Text strong>{e.true_label === 1 ? 'machine-written' : 'human-written'}</Text> ·
                condition: <Text code>{e.condition}</Text>
              </Text>
              <p className="review-text" style={{ marginTop: 8 }}>
                {e.text}
              </p>

              <Row gutter={[16, 16]} style={{ marginTop: 12 }}>
                <Col xs={24} lg={12}>
                  <Text strong>Token predictability</Text>
                  <div
                    className="highlighted-html"
                    dangerouslySetInnerHTML={{ __html: e.highlighted_html }}
                    style={{ marginTop: 8 }}
                  />
                </Col>
                <Col xs={24} lg={12}>
                  <Text strong>Reason codes</Text>
                  <div style={{ marginTop: 8 }}>
                    <ReasonCodes items={e.reason_codes} />
                  </div>
                </Col>
              </Row>

              <div style={{ marginTop: 16 }}>
                <Text strong>Feature contributions</Text>
                <div style={{ marginTop: 8 }}>
                  <FeatureTable items={e.features} />
                </div>
              </div>

              <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                {e.summary}
              </Paragraph>
            </Card>
          );
        })}
    </div>
  );
}
