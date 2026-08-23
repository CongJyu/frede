import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  Input,
  Progress,
  Radio,
  Row,
  Slider,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd';
import { api, fmtPct } from '../api';
import { useModelReady } from '../hooks';
import ReasonCodes from '../components/ReasonCodes';

const { Title, Paragraph, Text } = Typography;

const METRIC_COLS = [
  { title: 'Mode', dataIndex: 'label', render: (v) => <strong>{v}</strong> },
  { title: 'n', dataIndex: 'count' },
  { title: 'Acc', dataIndex: 'human_accuracy', render: fmtPct },
  { title: 'Prec', dataIndex: 'human_precision', render: fmtPct },
  { title: 'Rec', dataIndex: 'human_recall', render: fmtPct },
  { title: 'F1', dataIndex: 'human_f1', render: fmtPct },
  { title: "Cohen's κ", dataIndex: 'kappa', render: (v) => v.toFixed(3) },
  { title: 'Avg time', dataIndex: 'avg_time', render: (v) => `${v.toFixed(1)}s` },
  { title: 'Avg conf', dataIndex: 'avg_confidence', render: (v) => v.toFixed(1) },
];

function compareBars(metrics, rows) {
  const modes = Object.values(metrics);
  if (!modes.length) return null;
  return modes.map((m) => {
    const color = m.label.includes('Without') ? '#3282dc' : '#dc3232';
    return (
      <div key={m.label} style={{ marginBottom: 16 }}>
        <Text strong>{m.label}</Text>
        {rows.map((row) => (
          <Row key={row.key} align="middle" gutter={8} style={{ marginTop: 6 }}>
            <Col span={7}>
              <Text type="secondary">{row.label}</Text>
            </Col>
            <Col span={13}>
              <Progress percent={m[row.key] * 100} size="small" strokeColor={color} showInfo={false} />
            </Col>
            <Col span={4}>
              <Text strong>{fmtPct(m[row.key])}</Text>
            </Col>
          </Row>
        ))}
      </div>
    );
  });
}

export default function Hitl() {
  const health = useModelReady();
  const [session, setSession] = useState(null);
  const [results, setResults] = useState(null);
  const [judgment, setJudgment] = useState('Real');
  const [confidence, setConfidence] = useState(3);
  const [feedback, setFeedback] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const loadSession = useCallback(async () => {
    try {
      setSession(await api('/api/hitl/session'));
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    if (health.ready) loadSession();
  }, [health.ready, loadSession]);

  useEffect(() => {
    if (session && session.done && !results) {
      api('/api/hitl/results')
        .then(setResults)
        .catch((e) => setError(e.message));
    }
  }, [session, results]);

  async function submit() {
    setSubmitting(true);
    try {
      const s = await api('/api/hitl/judgment', {
        method: 'POST',
        body: JSON.stringify({ judgment, confidence, feedback }),
      });
      setSession(s);
      setFeedback('');
      setJudgment('Real');
      setConfidence(3);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function reset() {
    setResults(null);
    try {
      await api('/api/hitl/reset', { method: 'POST' });
      setSession(await api('/api/hitl/session'));
    } catch (e) {
      setError(e.message);
    }
  }

  const showXai = !!(session && !session.done && session.sample && session.sample.highlighted_html);
  const tableRows = results
    ? Object.values(results.mode_metrics).map((m) => ({ key: m.label, ...m }))
    : [];

  return (
    <div>
      <Title level={2} style={{ marginTop: 4 }}>
        Human-in-the-Loop Evaluation
      </Title>
      <Paragraph type="secondary">
        Judge each review as <strong>Fake</strong> or <strong>Real</strong>. Pass 1 shows the raw
        text; pass 2 shows the <em>same</em> reviews with XAI highlights and reason codes — so you
        can measure whether explanations help.
      </Paragraph>

      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />}

      {session && !session.done && session.sample && (
        <>
          <Card style={{ marginBottom: 16 }}>
            <Space style={{ width: '100%', justifyContent: 'space-between', display: 'flex' }}>
              <Text strong style={{ fontSize: 16 }}>
                Pass {session.pass_number} of 2 · Review {session.in_pass_idx + 1}/{session.total}
              </Text>
              <Tag color={session.mode === 'no_xai' ? 'default' : 'purple'}>
                {session.pass_label}
              </Tag>
            </Space>
            <p className="review-text" style={{ marginTop: 12, marginBottom: 0 }}>
              {session.sample.text}
            </p>
            {showXai && (
              <div style={{ marginTop: 14 }}>
                <Text strong>XAI explanation</Text>
                <div
                  className="highlighted-html"
                  dangerouslySetInnerHTML={{ __html: session.sample.highlighted_html }}
                  style={{ marginTop: 8 }}
                />
                <div style={{ marginTop: 12 }}>
                  <ReasonCodes items={session.sample.reason_codes} />
                </div>
              </div>
            )}
          </Card>

          <Card style={{ marginBottom: 16 }}>
            <Space size="large" wrap>
              <Space direction="vertical" size={4}>
                <Text strong>Your judgment</Text>
                <Radio.Group
                  value={judgment}
                  onChange={(e) => setJudgment(e.target.value)}
                  options={[
                    { label: 'Real', value: 'Real' },
                    { label: 'Fake', value: 'Fake' },
                  ]}
                  optionType="button"
                />
              </Space>
              <Space direction="vertical" size={4} style={{ width: 220 }}>
                <Text strong>
                  Confidence (1–5): <Text>{confidence}</Text>
                </Text>
                <Slider min={1} max={5} step={1} value={confidence} onChange={setConfidence} />
              </Space>
            </Space>
            <Text strong style={{ display: 'block', marginTop: 12 }}>
              Feedback (optional)
            </Text>
            <Input
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="What influenced your decision?"
              style={{ marginTop: 6 }}
            />
            <div style={{ marginTop: 12 }}>
              <Button type="primary" loading={submitting} onClick={submit}>
                Submit and Next
              </Button>
              {session.status && (
                <Text type="secondary" style={{ marginLeft: 12 }}>
                  {session.status}
                </Text>
              )}
            </div>
          </Card>
        </>
      )}

      {session && session.done && (
        <Card style={{ marginBottom: 16 }}>
          <Title level={4} style={{ marginTop: 0 }}>
            Evaluation Results
          </Title>
          {results && <Paragraph>{results.summary}</Paragraph>}
          <Table
            rowKey="label"
            columns={METRIC_COLS}
            dataSource={tableRows}
            pagination={false}
            size="small"
            bordered
          />
          {results && results.mode_metrics && (
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col xs={24} md={12}>
                <Card size="small" title="Human accuracy vs model accuracy">
                  {compareBars(results.mode_metrics, [
                    { key: 'human_accuracy', label: 'Human' },
                    { key: 'model_accuracy', label: 'Model' },
                  ])}
                </Card>
              </Col>
              <Col xs={24} md={12}>
                <Card size="small" title="Human F1 (with vs without XAI)">
                  {compareBars(results.mode_metrics, [{ key: 'human_f1', label: 'Human F1' }])}
                </Card>
              </Col>
            </Row>
          )}
          <div style={{ marginTop: 16, textAlign: 'right' }}>
            <Button onClick={reset}>Reset session</Button>
          </div>
        </Card>
      )}
    </div>
  );
}
