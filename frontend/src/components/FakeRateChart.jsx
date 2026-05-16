import React, { useState, useEffect, useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion, useMotionValue, useSpring } from 'framer-motion';
import { getHistory } from '../utils/storage';

const AnimatedNumber = ({ value, prefix = "", suffix = "" }) => {
  const [display, setDisplay] = useState(0);
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { duration: 1500, bounce: 0 });

  useEffect(() => {
    motionValue.set(value);
  }, [value, motionValue]);

  useEffect(() => {
    return springValue.on("change", (latest) => {
      setDisplay(Math.round(latest));
    });
  }, [springValue]);

  return <>{prefix}{display}{suffix}</>;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0b1d22',
      border: '1px solid rgba(0,212,200,0.2)',
      borderRadius: 8,
      padding: '10px 14px',
    }}>
      <p style={{ color: '#7ea8a4', fontSize: 11, marginBottom: 8, fontWeight: 600 }}>{label}</p>
      <p style={{ color: '#e8521e', fontSize: 13, marginBottom: 4 }}>Fake: {payload[0]?.value}</p>
      <p style={{ color: '#00d4c8', fontSize: 13, marginBottom: 4 }}>Real: {payload[1]?.value}</p>
      <p style={{ color: '#f0a429', fontSize: 13, fontWeight: 600, marginTop: 8 }}>
        Fake rate: {payload[0]?.payload?.fakeRate}%
      </p>
    </div>
  );
};

const FakeRateChart = () => {
  const [history, setHistory] = useState([]);
  const [timeframe, setTimeframe] = useState('7 Days'); // '7 Days', '30 Days', 'All Time'

  const loadData = () => setHistory(getHistory());

  useEffect(() => {
    loadData();
    window.addEventListener('va_history_updated', loadData);
    return () => window.removeEventListener('va_history_updated', loadData);
  }, []);

  // Stats
  const stats = useMemo(() => {
    const total = history.length;
    let fake = 0;
    let real = 0;
    let confSum = 0;

    history.forEach(item => {
      if (item.prediction === 'FAKE') fake++;
      else real++;
      confSum += item.confidence;
    });

    return {
      total,
      fake,
      real,
      fakePct: total > 0 ? Math.round((fake / total) * 100) : 0,
      realPct: total > 0 ? Math.round((real / total) * 100) : 0,
      avgConf: total > 0 ? Math.round(confSum / total) : 0,
    };
  }, [history]);

  // Chart Data
  const chartData = useMemo(() => {
    const days = {};
    const now = new Date();
    
    // Determine cutoff
    let cutoff = new Date(0); // All time
    if (timeframe === '7 Days') cutoff.setDate(now.getDate() - 7);
    if (timeframe === '30 Days') cutoff.setDate(now.getDate() - 30);

    history.forEach(entry => {
      const entryDate = new Date(entry.timestamp);
      if (entryDate < cutoff) return;

      const day = entry.date;
      if (!days[day]) days[day] = { date: day, fake: 0, real: 0, total: 0 };
      if (entry.prediction === 'FAKE') days[day].fake++;
      else days[day].real++;
      days[day].total++;
    });

    return Object.values(days)
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .map(d => ({
        ...d,
        dateStr: d.date.substring(0, 5), // short date
        fakeRate: Math.round((d.fake / d.total) * 100),
      }));
  }, [history, timeframe]);

  return (
    <div style={{ marginBottom: '40px' }}>
      
      {/* Header Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        
        {/* Total */}
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'rgba(0,212,200,0.1)', color: '#00d4c8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>
            <i className="ti ti-waveform"></i>
          </div>
          <div>
            <div style={{ fontSize: '32px', fontWeight: 800, color: '#00d4c8', lineHeight: 1 }}>
              <AnimatedNumber value={stats.total} />
            </div>
            <div style={{ fontSize: '12px', color: '#7ea8a4', marginTop: 4 }}>Total voice samples</div>
          </div>
        </div>

        {/* Fake */}
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'rgba(232,82,30,0.1)', color: '#e8521e', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>
            <i className="ti ti-alert-triangle"></i>
          </div>
          <div>
            <div style={{ fontSize: '32px', fontWeight: 800, color: '#e8521e', lineHeight: 1 }}>
              <AnimatedNumber value={stats.fake} />
            </div>
            <div style={{ fontSize: '12px', color: '#7ea8a4', marginTop: 4 }}>Fake ({stats.fakePct}%)</div>
          </div>
        </div>

        {/* Real */}
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'rgba(0,212,200,0.1)', color: '#00d4c8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>
            <i className="ti ti-shield-check"></i>
          </div>
          <div>
            <div style={{ fontSize: '32px', fontWeight: 800, color: '#00d4c8', lineHeight: 1 }}>
              <AnimatedNumber value={stats.real} />
            </div>
            <div style={{ fontSize: '12px', color: '#7ea8a4', marginTop: 4 }}>Real ({stats.realPct}%)</div>
          </div>
        </div>

        {/* Confidence */}
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'rgba(240,164,41,0.1)', color: '#f0a429', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>
            <i className="ti ti-target"></i>
          </div>
          <div>
            <div style={{ fontSize: '32px', fontWeight: 800, color: '#f0a429', lineHeight: 1 }}>
              <AnimatedNumber value={stats.avgConf} suffix="%" />
            </div>
            <div style={{ fontSize: '12px', color: '#7ea8a4', marginTop: 4 }}>Avg Confidence</div>
          </div>
        </div>

      </div>

      {/* Chart Card */}
      <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '24px', backdropFilter: 'blur(16px)' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px' }}>
          <div>
            <div style={{ fontSize: '10px', letterSpacing: '0.2em', color: '#3d6e6a', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 700 }}>
              DETECTION RATE OVER TIME
            </div>
            <div style={{ fontSize: '16px', color: '#dfe8e6' }}>
              Fake vs Real voice detections — {timeframe.toLowerCase()}
            </div>
          </div>
          
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', borderRadius: '6px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
            {['7 Days', '30 Days', 'All Time'].map(t => (
              <button 
                key={t}
                onClick={() => setTimeframe(t)}
                style={{ 
                  padding: '6px 12px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                  background: timeframe === t ? 'rgba(255,255,255,0.1)' : 'transparent',
                  color: timeframe === t ? 'white' : '#7ea8a4'
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {chartData.length === 0 ? (
          <div style={{ height: 300, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#7ea8a4' }}>
            <i className="ti ti-chart-area-line" style={{ fontSize: 48, opacity: 0.3, marginBottom: 16 }}></i>
            <div>No data available for this timeframe.</div>
          </div>
        ) : (
          <div style={{ height: 300, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorFake" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#e8521e" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#e8521e" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorReal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4c8" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00d4c8" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="dateStr" stroke="#7ea8a4" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#7ea8a4" fontSize={11} tickLine={false} axisLine={false} />
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="fake" name="Fake" stroke="#e8521e" strokeWidth={3} fillOpacity={1} fill="url(#colorFake)" />
                <Area type="monotone" dataKey="real" name="Real" stroke="#00d4c8" strokeWidth={3} fillOpacity={1} fill="url(#colorReal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

      </div>
    </div>
  );
};

export default FakeRateChart;
