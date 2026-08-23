import { useState } from 'react';
import { Alert, Button, Card, Col, Input, Row, Space, Tag, Typography } from 'antd';
import { api, fmtPct } from '../api';
import { useModelReady } from '../hooks';
import ReasonCodes from '../components/ReasonCodes';
import SignedBars from '../components/SignedBars';
import TokenSpans from '../components/TokenSpans';

const { Title, Paragraph, Text } = Typography;

const SAMPLES = [
  'This is the best restaurant ever! Amazing food, incredible service, perfect atmosphere. Everyone should come here immediately! Trust me, you won\'t regret it. Five stars!',
  'We visited on a Saturday evening and waited about 20 minutes for a table. The pad thai was decent but a bit too sweet for my taste. Service was friendly. Prices are reasonable for the neighborhood. Would come back to try other dishes.',
  'WORST EXPERIENCE EVER. The food was absolutely terrible, the waiter was incredibly rude, and the place was filthy. DO NOT go here. I would give zero stars if I could. Never coming back.',
  'Absolutely phenomenal! The chef is a genius. Every single dish was outstanding and flawless. Best meal of my entire life. Go now before it\'s too late! Highly recommend to everyone!',
];

export default function Analyzer() {
  const health = useModelReady();
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function analyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api('/api/analyze', {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const isFake = result?.prediction === 'FAKE';

  return (
    <div>
      <Title level={2} style={{ marginTop: 4 }}>
        Fake Restaurant Review Detector
      </Title>
      <Paragraph type="secondary">
        Paste a restaurant review to see whether it&apos;s <strong>genuine</strong> or{' '}
        <strong>fake</strong> — and which words pushed the model&apos;s decision.
      </Paragraph>

      {!health.ready && health.error && (
        <Alert
          type="error"
          showIcon
          message={`Error: ${health.error} — check that training has been run (make train).`}
          style={{ marginBottom: 16 }}
        />
      )}

      <Card style={{ marginBottom: 16 }}>
        <Text strong>Enter a Restaurant Review</Text>
        <Input.TextArea
          rows={6}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste a review here..."
          style={{ marginTop: 8 }}
        />
        <Row gutter={8} style={{ marginTop: 12 }}>
          <Col>
            <Button type="primary" loading={loading} disabled={!health.ready} onClick={analyze}>
              Analyze Review
            </Button>
          </Col>
          <Col>
            <Button
              onClick={() => {
                setText('');
                setResult(null);
                setError(null);
              }}
            >
              Clear
            </Button>
          </Col>
        </Row>
        <Space wrap size={[8, 8]} style={{ marginTop: 14 }}>
          <Text type="secondary">Try a sample:</Text>
          {SAMPLES.map((s, i) => (
            <Button key={i} size="small" onClick={() => setText(s)}>
              Sample {i + 1}: {s.slice(0, 48)}…
            </Button>
          ))}
        </Space>
      </Card>

      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />}

      {result && (
        <>
          <Card style={{ marginBottom: 16 }}>
            <Tag
              color={isFake ? 'red' : 'green'}
              style={{ fontSize: 24, fontWeight: 700, padding: '4px 16px', borderRadius: 8 }}
            >
              {result.prediction}
            </Tag>
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">
                Fake probability: <Text strong>{fmtPct(result.fake_prob)}</Text> · Confidence:{' '}
                <Text strong>{fmtPct(result.confidence)}</Text> · analyzed in{' '}
                <Text strong>{(result.elapsed_ms / 1000).toFixed(1)}s</Text>
              </Text>
            </div>
          </Card>

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Suspicious segments (LIME)">
                <div
                  className="highlighted-html"
                  dangerouslySetInnerHTML={{ __html: result.highlighted_html }}
                />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="Reason codes">
                <ReasonCodes items={result.reason_codes} />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col xs={24} lg={12}>
              <Card title="Feature importance (LIME)">
                <SignedBars
                  items={result.lime_features.map((f) => ({ label: f.word, value: f.weight }))}
                />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="Integrated Gradients — tokens pushing toward FAKE">
                <TokenSpans tokens={result.ig_tokens} attrs={result.ig_attrs} />
              </Card>
            </Col>
          </Row>

          <Card title="Summary" style={{ marginTop: 16 }}>
            <Paragraph style={{ marginBottom: 0 }}>{result.summary}</Paragraph>
          </Card>
        </>
      )}
    </div>
  );
}
