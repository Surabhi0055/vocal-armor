import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getHistory } from '../utils/storage';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#0b1d22', border: '1px solid rgba(123,157,174,0.2)', borderRadius: 8, padding: '10px 14px' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: 11, marginBottom: 4, fontWeight: 600 }}>Confidence: {label}</p>
      <p style={{ color: 'var(--text-card)', fontSize: 13 }}>Count: {payload[0].value}</p>
    </div>
  );
};

const ConfidenceHistogram = () => {
  const [history, setHistory] = useState([]);
  const [type, setType] = useState('FAKE'); // 'FAKE' | 'REAL'

  const loadData = () => setHistory(getHistory());

  useEffect(() => {
    loadData();
    window.addEventListener('va_history_updated', loadData);
    return () => window.removeEventListener('va_history_updated', loadData);
  }, []);

  const chartData = useMemo(() => {
    const buckets = [
      { range: '50-60%', min: 50, max: 60, count: 0 },
      { range: '60-70%', min: 60, max: 70, count: 0 },
      { range: '70-80%', min: 70, max: 80, count: 0 },
      { range: '80-90%', min: 80, max: 90, count: 0 },
      { range: '90-95%', min: 90, max: 95, count: 0 },
      { range: '95-99%', min: 95, max: 99, count: 0 },
      { range: '99-100%', min: 99, max: 101, count: 0 }, // 101 to include 100
    ];

    history
      .filter(e => e.prediction === type)
      .forEach(e => {
        const bucket = buckets.find(b => e.confidence >= b.min && e.confidence < b.max);
        if (bucket) bucket.count++;
      });

    return buckets;
  }, [history, type]);

  const insight = useMemo(() => {
    if (chartData.reduce((sum, b) => sum + b.count, 0) === 0) return null;

    const highConf = chartData.filter(b => b.min >= 90).reduce((s, b) => s + b.count, 0);
    const total = chartData.reduce((s, b) => s + b.count, 0);
    const pct = Math.round((highConf / total) * 100);

    const targetLabel = type === 'FAKE' ? 'fake voices' : 'real voices';
    return `${pct}% of ${targetLabel} were detected with 90%+ confidence — indicating the model is highly certain on most samples.`;
  }, [chartData, type]);

  const color = type === 'FAKE' ? '#A63A3F' : '#C6A75E';

  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '16px', padding: '24px', backdropFilter: 'blur(16px)', height: '100%', display: 'flex', flexDirection: 'column' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ fontSize: '10px', letterSpacing: '0.2em', color: '#3d6e6a', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 700 }}>
            CONFIDENCE DISTRIBUTION
          </div>
        </div>
        
        <div style={{ display: 'flex', background: 'rgba(232,220,200,0.05)', borderRadius: '6px', overflow: 'hidden', border: '1px solid rgba(232,220,200,0.1)' }}>
          <button 
            onClick={() => setType('FAKE')}
            style={{ 
              padding: '6px 12px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
              background: type === 'FAKE' ? 'rgba(122,46,50,0.15)' : 'transparent',
              color: type === 'FAKE' ? '#A63A3F' : 'var(--text-muted)'
            }}
          >
            FAKE
          </button>
          <button 
            onClick={() => setType('REAL')}
            style={{ 
              padding: '6px 12px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
              background: type === 'REAL' ? 'rgba(123,157,174,0.15)' : 'transparent',
              color: type === 'REAL' ? '#C6A75E' : 'var(--text-muted)'
            }}
          >
            REAL
          </button>
        </div>
      </div>

      {chartData.reduce((sum, b) => sum + b.count, 0) === 0 ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', minHeight: 250 }}>
          <i className="ti ti-chart-bar" style={{ fontSize: 48, opacity: 0.3, marginBottom: 16 }}></i>
          <div>No {type.toLowerCase()} data available yet.</div>
        </div>
      ) : (
        <>
          <div style={{ height: 250, width: '100%', marginBottom: '24px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(232,220,200,0.05)" vertical={false} />
                <XAxis dataKey="range" stroke="var(--text-muted)" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(232,220,200,0.05)' }} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={color} fillOpacity={0.8} style={{ transition: 'all 0.2s' }} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Insight Card */}
          <div style={{ background: 'rgba(240,164,41,0.08)', border: '1px solid rgba(240,164,41,0.2)', borderRadius: '12px', padding: '16px', display: 'flex', gap: '16px', alignItems: 'flex-start', marginTop: 'auto' }}>
            <i className="ti ti-bulb" style={{ color: '#f0a429', fontSize: '24px', flexShrink: 0 }}></i>
            <div style={{ color: '#f0a429', fontSize: '13px', lineHeight: 1.5 }}>
              {insight}
            </div>
          </div>
        </>
      )}

    </div>
  );
};

export default ConfidenceHistogram;
