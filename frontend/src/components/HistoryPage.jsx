import React, { useState, useEffect, useCallback } from "react";
import { getHistory } from "../utils/storage";
import Footer from "./Footer";

// Lazy imports with error boundaries to isolate crashes
let FakeRateChart, ConfidenceHistogram, HistoryTable;
try { FakeRateChart = React.lazy(() => import("./FakeRateChart")); } catch(e) {}
try { ConfidenceHistogram = React.lazy(() => import("./ConfidenceHistogram")); } catch(e) {}
try { HistoryTable = React.lazy(() => import("./HistoryTable")); } catch(e) {}

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 24, background: 'rgba(122,46,50,0.1)', border: '1px solid rgba(122,46,50,0.3)', borderRadius: 12, color: '#A63A3F', marginBottom: 24 }}>
          <strong>Component Error:</strong> {this.state.error?.message}
        </div>
      );
    }
    return this.props.children;
  }
}

const HistoryPage = () => {
  const [quickStats, setQuickStats] = useState({
    maxFake: null, maxReal: null, streak: 0, todayCount: 0
  });

  const loadQuickStats = useCallback(() => {
    const history = getHistory();
    let maxF = null, maxR = null;
    let currentStreak = 0, maxStreak = 0;
    let prevWasFake = false;
    let todayC = 0;
    const today = new Date().toLocaleDateString();

    history.forEach(item => {
      if (item.prediction === 'FAKE') {
        if (!maxF || item.confidence > maxF.confidence) maxF = item;
      } else {
        if (!maxR || item.confidence > maxR.confidence) maxR = item;
      }
      if (item.date === today) todayC++;
    });

    [...history].reverse().forEach(item => {
      if (item.prediction === 'FAKE') {
        if (prevWasFake) currentStreak++;
        else currentStreak = 1;
        prevWasFake = true;
        if (currentStreak > maxStreak) maxStreak = currentStreak;
      } else {
        prevWasFake = false;
        currentStreak = 0;
      }
    });

    setQuickStats({ maxFake: maxF, maxReal: maxR, streak: maxStreak, todayCount: todayC });
  }, []);

  useEffect(() => {
    loadQuickStats();
    window.addEventListener('va_history_updated', loadQuickStats);
    return () => window.removeEventListener('va_history_updated', loadQuickStats);
  }, [loadQuickStats]);

  return (
    <div className="dashboard" style={{ paddingBottom: '100px' }}>



      {/* FakeRateChart */}
      <ErrorBoundary>
        <React.Suspense fallback={
          <div style={{ height: 300, background: 'rgba(232,220,200,0.02)', borderRadius: 16, marginBottom: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            Loading chart...
          </div>
        }>
          {FakeRateChart && <FakeRateChart />}
        </React.Suspense>
      </ErrorBoundary>

      {/* Histogram + Quick Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '40px' }}>

        <ErrorBoundary>
          <React.Suspense fallback={
            <div style={{ height: 300, background: 'rgba(232,220,200,0.02)', borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              Loading chart...
            </div>
          }>
            {ConfidenceHistogram && <ConfidenceHistogram />}
          </React.Suspense>
        </ErrorBoundary>

        {/* Quick Stats Panel */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '16px', padding: '24px', backdropFilter: 'blur(16px)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ fontSize: '10px', letterSpacing: '0.2em', color: '#3d6e6a', textTransform: 'uppercase', fontWeight: 700 }}>
            QUICK STATS
          </div>

          <div style={{ background: 'rgba(122,46,50,0.05)', border: '1px solid rgba(122,46,50,0.1)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Most Confident FAKE</div>
            {quickStats.maxFake ? (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-main)', fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                  {quickStats.maxFake.filename}
                </span>
                <span style={{ color: '#A63A3F', fontWeight: 600 }}>{quickStats.maxFake.confidence.toFixed(1)}%</span>
              </div>
            ) : (
              <div style={{ color: 'rgba(232,220,200,0.3)', fontSize: '13px' }}>No fakes detected yet</div>
            )}
          </div>

          <div style={{ background: 'rgba(123,157,174,0.05)', border: '1px solid rgba(123,157,174,0.1)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Most Confident REAL</div>
            {quickStats.maxReal ? (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-main)', fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                  {quickStats.maxReal.filename}
                </span>
                <span style={{ color: '#C6A75E', fontWeight: 600 }}>{quickStats.maxReal.confidence.toFixed(1)}%</span>
              </div>
            ) : (
              <div style={{ color: 'rgba(232,220,200,0.3)', fontSize: '13px' }}>No real voices detected yet</div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '16px' }}>
            <div style={{ flex: 1, background: 'rgba(232,220,200,0.02)', border: '1px solid rgba(232,220,200,0.05)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#f0a429', marginBottom: '4px' }}>{quickStats.streak}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.2 }}>Longest Fake Streak</div>
            </div>
            <div style={{ flex: 1, background: 'rgba(232,220,200,0.02)', border: '1px solid rgba(232,220,200,0.05)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#C6A75E', marginBottom: '4px' }}>{quickStats.todayCount}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.2 }}>Analyses Today</div>
            </div>
          </div>
        </div>
      </div>

      {/* History Table */}
      <ErrorBoundary>
        <React.Suspense fallback={
          <div style={{ height: 300, background: 'rgba(232,220,200,0.02)', borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            Loading table...
          </div>
        }>
          {HistoryTable && <HistoryTable />}
        </React.Suspense>
      </ErrorBoundary>

      <Footer />
    </div>
  );
};

export default HistoryPage;
