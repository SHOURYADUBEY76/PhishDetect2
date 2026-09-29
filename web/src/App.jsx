import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Search,
  ExternalLink,
  Zap,
  Globe,
  Layers,
  HelpCircle,
  BarChart3,
  Server
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';

const DEMO_URLS = [
  {
    label: 'SBI KYC Phish',
    type: 'danger',
    url: 'http://sbi-kyc-update-login.top/account/verify.php'
  },
  {
    label: 'Official SBI Portal',
    type: 'safe',
    url: 'https://onlinesbi.sbi/portal'
  },
  {
    label: 'Paytm Cashback Scam',
    type: 'danger',
    url: 'https://paytm-cashback-rewards-free.xyz/claim'
  },
  {
    label: 'Official IRCTC',
    type: 'safe',
    url: 'https://irctc.co.in'
  },
  {
    label: 'Aadhaar Link Scam',
    type: 'danger',
    url: 'http://uidai-aadhaar-link-portal-12.net/biometric'
  }
];

export default function App() {
  const [url, setUrl] = useState('');
  const [includeRegional, setIncludeRegional] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [showBenchmarkTab, setShowBenchmarkTab] = useState(false);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendHealth(data);
      }
    } catch {
      setBackendHealth({ status: 'offline', model_loaded: false });
    }
  };

  const handleAnalyze = async (targetUrl = url) => {
    const checkUrl = targetUrl || url;
    if (!checkUrl.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: checkUrl.trim(),
          include_regional: includeRegional
        })
      });

      if (!res.ok) {
        throw new Error(`Inference service error (HTTP ${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to backend inference engine.');
    } finally {
      setLoading(false);
    }
  };

  const loadDemo = (demoUrl) => {
    setUrl(demoUrl);
    handleAnalyze(demoUrl);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="badge-model">
          <span className="pulse-dot"></span>
          <span>
            Engine: {backendHealth?.active_model || 'LightGBM Region-Aware'}
            {backendHealth?.status === 'offline' && ' (API Offline)'}
          </span>
        </div>
        <h1 className="app-title">Region-Aware Phishing URL Detector</h1>
        <p className="app-subtitle">
          Real-time threat intelligence engine evaluating geographic, temporal, and adversarial distribution shifts.
        </p>
      </header>

      {/* Input Form */}
      <div className="glass-card">
        <form
          className="search-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleAnalyze();
          }}
        >
          <div className="input-group">
            <div className="url-input-wrapper">
              <Search className="input-icon" size={20} />
              <input
                type="text"
                className="url-input"
                placeholder="Paste any candidate URL string (e.g. http://sbi-kyc-update.top/login)..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
            </div>
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? (
                <>Analyzing...</>
              ) : (
                <>
                  <Zap size={18} />
                  Analyze URL
                </>
              )}
            </button>
          </div>

          <div className="controls-row">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={includeRegional}
                onChange={(e) => setIncludeRegional(e.target.checked)}
              />
              <span>Enable Region-Aware Feature Representation (India Threat Dictionary)</span>
            </label>

            <button
              type="button"
              className="demo-pill"
              onClick={() => setShowBenchmarkTab(!showBenchmarkTab)}
            >
              <BarChart3 size={14} style={{ display: 'inline', marginRight: 4 }} />
              {showBenchmarkTab ? 'Hide Benchmark Data' : 'View Ablation Results'}
            </button>
          </div>

          {/* Quick Demo Buttons */}
          <div className="demo-section">
            <span style={{ fontSize: '0.82rem', color: '#64748b' }}>Quick Test Cases:</span>
            {DEMO_URLS.map((demo, idx) => (
              <button
                key={idx}
                type="button"
                className={`demo-pill ${demo.type}`}
                onClick={() => loadDemo(demo.url)}
              >
                {demo.label}
              </button>
            ))}
          </div>
        </form>
      </div>

      {/* Benchmark Ablation Modal / Drawer */}
      {showBenchmarkTab && (
        <div className="glass-card" style={{ borderLeft: '4px solid #3b82f6' }}>
          <h3 style={{ fontSize: '1.1rem', marginBottom: 8 }}>
            Empirical Results Summary (Experiments E1, E2, E3)
          </h3>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: 12 }}>
            Evaluation across 5 distinct random seeds under leakage-free registered domain partitioning:
          </p>
          <div className="ablation-table-wrapper">
            <table className="ablation-table">
              <thead>
                <tr>
                  <th>Experiment / Configuration</th>
                  <th>Features</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>False Neg. Rate</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>E1: Global Benchmark (PhreshPhish)</td>
                  <td>39 (Lex + Struct)</td>
                  <td>91.07%</td>
                  <td>87.42%</td>
                  <td>89.18%</td>
                  <td>12.58%</td>
                </tr>
                <tr>
                  <td>E2: Regional Transfer Gap (Indian Test)</td>
                  <td>39 (Lex + Struct)</td>
                  <td>94.08%</td>
                  <td>91.00%</td>
                  <td>92.50%</td>
                  <td>9.00%</td>
                </tr>
                <tr>
                  <td>E3 Ablation: Lexical Only</td>
                  <td>24 (Lexical)</td>
                  <td>85.80%</td>
                  <td>78.59%</td>
                  <td>85.80%</td>
                  <td>21.41%</td>
                </tr>
                <tr className="highlight-row">
                  <td>E3: Full Region-Aware Augmented</td>
                  <td>47 (Lex + Struct + Region)</td>
                  <td>99.71%</td>
                  <td>98.22%</td>
                  <td>98.96%</td>
                  <td>1.78%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Error Notice */}
      {error && (
        <div className="glass-card" style={{ borderColor: '#ef4444', color: '#fca5a5' }}>
          {error}
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="glass-card results-card">
          {/* Main Verdict Banner */}
          <div className={`verdict-banner ${result.is_phishing ? 'phish' : 'legit'}`}>
            <div className="verdict-title-group">
              {result.is_phishing ? (
                <ShieldAlert className="verdict-icon" />
              ) : (
                <ShieldCheck className="verdict-icon" />
              )}
              <div>
                <div className="verdict-text-main">
                  {result.is_phishing ? 'PHISHING URL DETECTED' : 'VERIFIED LEGITIMATE URL'}
                </div>
                <div className="verdict-text-sub">
                  Analyzed domain: <strong>{result.registered_domain || 'N/A'}</strong> (TLD: .{result.tld})
                </div>
              </div>
            </div>
            <div className="score-badge">
              <div className="score-number">{(result.risk_score * 100).toFixed(1)}%</div>
              <div className="score-label">Threat Probability</div>
            </div>
          </div>

          {/* Key Metrics Row */}
          <div className="metrics-row">
            <div className="metric-box">
              <div className="metric-title">Inference Latency</div>
              <div className="metric-val">{result.latency_ms} ms</div>
            </div>
            <div className="metric-box">
              <div className="metric-title">Model Confidence</div>
              <div className="metric-val">{(result.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-box">
              <div className="metric-title">Decision Threshold</div>
              <div className="metric-val">{result.decision_threshold}</div>
            </div>
            <div className="metric-box">
              <div className="metric-title">Protocol</div>
              <div className="metric-val">
                {result.key_signals.is_https ? 'HTTPS (Secure)' : 'HTTP (Insecure)'}
              </div>
            </div>
          </div>

          {/* Human-Readable Explanation Layer */}
          <div className="reasons-box">
            <div className="reasons-title">
              <HelpCircle size={18} />
              Reasoning & Signal Attribution:
            </div>
            <ul className="reasons-list">
              {result.reasons && result.reasons.length > 0 ? (
                result.reasons.map((reason, idx) => (
                  <li
                    key={idx}
                    className={`reason-item ${result.is_phishing ? 'danger' : 'safe'}`}
                  >
                    <span className="reason-bullet">&#x2022;</span>
                    <span>{reason}</span>
                  </li>
                ))
              ) : (
                <li className="reason-item">No anomalous threat indicators triggered.</li>
              )}
            </ul>
          </div>

          {/* Extracted Threat Signals Grid */}
          <h4 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: 12 }}>
            Extracted Feature Signals:
          </h4>
          <div className="signals-grid">
            <div className="signal-chip">
              <span className="signal-name">Brand Lure Token</span>
              <span className={`signal-status ${result.key_signals.has_indian_brand_token ? 'flagged' : 'clear'}`}>
                {result.key_signals.has_indian_brand_token ? 'Detected' : 'None'}
              </span>
            </div>
            <div className="signal-chip">
              <span className="signal-name">Brand-TLD Misalignment</span>
              <span className={`signal-status ${result.key_signals.brand_tld_misalignment ? 'flagged' : 'clear'}`}>
                {result.key_signals.brand_tld_misalignment ? 'High Risk' : 'Aligned'}
              </span>
            </div>
            <div className="signal-chip">
              <span className="signal-name">Regional Action Lure</span>
              <span className={`signal-status ${result.key_signals.has_indian_lure ? 'flagged' : 'clear'}`}>
                {result.key_signals.has_indian_lure ? 'Detected' : 'None'}
              </span>
            </div>
            <div className="signal-chip">
              <span className="signal-name">Raw IP Address</span>
              <span className={`signal-status ${result.key_signals.has_ip ? 'flagged' : 'clear'}`}>
                {result.key_signals.has_ip ? 'Detected' : 'No'}
              </span>
            </div>
            <div className="signal-chip">
              <span className="signal-name">Subdomain Depth</span>
              <span className="signal-status clear">
                {result.key_signals.num_subdomains} levels
              </span>
            </div>
            <div className="signal-chip">
              <span className="signal-name">Shannon Entropy</span>
              <span className="signal-status clear">
                {result.key_signals.url_entropy}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
