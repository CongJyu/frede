import { useEffect, useState } from 'react';
import { Alert, Card, Col, Row, Space, Tag, Typography } from 'antd';
import { api, fmtPct } from '../api';
import ReasonCodes from '../components/ReasonCodes';
import SignedBars from '../components/SignedBars';
import TokenSpans from '../components/TokenSpans';

const { Title, Paragraph, Text } = Typography;

function shapItems(example) {
  if (!example.shap) return null;
  return example.shap.tokens
    .map((tok, j) => ({ label: tok, value: example.shap.values[j] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 20);
}

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
        Precomputed Explanations
      </Title>
      <Paragraph type="secondary">
        Representative test reviews with LIME, Integrated Gradients, and SHAP (precomputed offline —
        SHAP is too slow to run live).
      </Paragraph>

      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />}

      {data &&
        data.examples.map((e, i) => {
          const isFake = e.model_pred === 1;
          const shap = shapItems(e);
          return (
            <Card
              key={i}
              style={{ marginBottom: 16 }}
              title={
                <Space style={{ width: '100%', justifyContent: 'space-between', display: 'flex' }}>
                  <span>{e.title}</span>
                  <Tag color={isFake ? 'red' : 'green'}>
                    Model: {isFake ? 'FAKE' : 'REAL'} · truth: {e.true_label === 1 ? 'FAKE' : 'REAL'}
                  </Tag>
                </Space>
              }
            >
              <Text type="secondary">P(Fake) = {fmtPct(e.fake_prob)}</Text>
              <p className="review-text" style={{ marginTop: 8 }}>
                {e.text}
              </p>

              <Row gutter={[16, 16]} style={{ marginTop: 12 }}>
                <Col xs={24} lg={12}>
                  <Text strong>Reason codes</Text>
                  <div style={{ marginTop: 8 }}>
                    <ReasonCodes items={e.reason_codes} />
                  </div>
                </Col>
                <Col xs={24} lg={12}>
                  <Text strong>Highlighted text (LIME)</Text>
                  <div
                    className="highlighted-html"
                    dangerouslySetInnerHTML={{ __html: e.highlighted_html }}
                    style={{ marginTop: 8 }}
                  />
                </Col>
              </Row>

              <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                <Col xs={24} lg={12}>
                  <Text strong>Feature importance (LIME)</Text>
                  <div style={{ marginTop: 8 }}>
                    <SignedBars
                      items={e.lime_features.map((f) => ({ label: f.word, value: f.weight }))}
                    />
                  </div>
                </Col>
                <Col xs={24} lg={12}>
                  <Text strong>Integrated Gradients</Text>
                  <div style={{ marginTop: 8 }}>
                    <TokenSpans tokens={e.ig_tokens} attrs={e.ig_attrs} hint={false} />
                  </div>
                </Col>
              </Row>

              <div style={{ marginTop: 16 }}>
                <Text strong>SHAP — token attribution (Fake class)</Text>
                <div style={{ marginTop: 8 }}>
                  {shap ? (
                    <SignedBars items={shap} />
                  ) : (
                    <Text type="secondary">
                      SHAP not precomputed for this run (see `make train --with-shap`).
                    </Text>
                  )}
                </div>
              </div>
            </Card>
          );
        })}
    </div>
  );
}
