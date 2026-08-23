import { useEffect, useState } from 'react';
import { Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { ConfigProvider, Layout, Menu, Segmented, theme as antdTheme } from 'antd';
import Analyzer from './pages/Analyzer';
import Hitl from './pages/Hitl';
import Examples from './pages/Examples';

const { Header, Content } = Layout;

const NAV = [
  { key: '/', label: 'Analyzer' },
  { key: '/hitl', label: 'HITL Evaluation' },
  { key: '/examples', label: 'Examples' },
];

/* Theme preference: 'system' (follows OS) | 'light' | 'dark'. Persisted locally. */
const THEME_KEY = 'frede-theme';

function getInitialMode() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark' || saved === 'system') return saved;
  } catch (_) {
    /* storage unavailable */
  }
  return 'system';
}

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();

  const [mode, setMode] = useState(getInitialMode);
  const [systemDark, setSystemDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  // Live-follow the OS theme while in "system" mode.
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e) => setSystemDark(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const isDark = mode === 'system' ? systemDark : mode === 'dark';

  useEffect(() => {
    try {
      localStorage.setItem(THEME_KEY, mode);
    } catch (_) {
      /* ignore */
    }
  }, [mode]);

  // Let native controls (scrollbars, form fields) match the resolved theme.
  useEffect(() => {
    document.documentElement.style.colorScheme = isDark ? 'dark' : 'light';
  }, [isDark]);

  return (
    <ConfigProvider
      theme={{
        algorithm: isDark ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        cssVar: true,
        token: {
          colorPrimary: '#2b6cb0',
          colorError: '#dc3232',
          colorSuccess: '#2e8b57',
          borderRadius: 8,
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ display: 'flex', alignItems: 'center', paddingInline: 24 }}>
          <div
            style={{
              color: '#fff',
              fontWeight: 700,
              fontSize: 18,
              marginRight: 24,
              whiteSpace: 'nowrap',
            }}
          >
            Frede
          </div>
          <Menu
            theme="dark"
            mode="horizontal"
            selectedKeys={[location.pathname]}
            items={NAV}
            onClick={({ key }) => navigate(key)}
            style={{ flex: 1, minWidth: 0 }}
          />
          <Segmented
            value={mode}
            onChange={(v) => setMode(v)}
            options={[
              { label: 'System', value: 'system' },
              { label: 'Light', value: 'light' },
              { label: 'Dark', value: 'dark' },
            ]}
            style={{ marginLeft: 16, whiteSpace: 'nowrap' }}
          />
        </Header>
        <Content style={{ width: '100%', maxWidth: 960, margin: '0 auto', padding: '24px 16px' }}>
          <Routes>
            <Route path="/" element={<Analyzer />} />
            <Route path="/hitl" element={<Hitl />} />
            <Route path="/examples" element={<Examples />} />
          </Routes>
        </Content>
      </Layout>
    </ConfigProvider>
  );
}
