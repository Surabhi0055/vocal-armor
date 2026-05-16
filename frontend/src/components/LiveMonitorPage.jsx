import React from 'react';

const LiveMonitorPage = () => {
  return (
    <div className="dashboard" style={{ padding: '40px' }}>
      <div className="hero-section" style={{ textAlign: 'left', marginBottom: '40px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(232, 82, 30, 0.07)', border: '1px solid rgba(232, 82, 30, 0.22)', borderRadius: '100px', padding: '8px 20px', fontSize: '10px', letterSpacing: '0.14em', color: '#e8521e', fontWeight: 600, textTransform: 'uppercase', marginBottom: '24px', display: 'inline-flex' }}>
          <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#e8521e', boxShadow: '0 0 8px #e8521e', animation: 'livePulse 2s infinite' }}></div>
          LIVE MONITORING (BETA)
        </div>
        <h1 className="hero-title" style={{ fontSize: '36px' }}>
          REAL-TIME <span className="text-orange">STREAM ANALYSIS</span>
        </h1>
        <p className="hero-subtitle">Connect your microphone or input stream to continuously monitor audio for deepfakes.</p>
      </div>

      <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '60px 20px', textAlign: 'center', backdropFilter: 'blur(16px)' }}>
        <i className="ti ti-microphone" style={{ fontSize: '64px', color: '#e8521e', opacity: 0.8, marginBottom: '24px', display: 'block' }}></i>
        <h2 style={{ color: 'white', fontSize: '24px', marginBottom: '16px' }}>Microphone Access Required</h2>
        <p style={{ color: '#7ea8a4', maxWidth: '400px', margin: '0 auto 32px' }}>
          To use the live monitoring feature, please allow microphone permissions in your browser. Audio is processed locally before feature extraction.
        </p>
        <button className="btn-primary" style={{ margin: '0 auto', background: 'rgba(232, 82, 30, 0.1)', border: '1px solid #e8521e', color: '#e8521e', boxShadow: '0 0 20px rgba(232, 82, 30, 0.2)' }}>
          <i className="ti ti-player-record"></i> START MONITORING
        </button>
      </div>
    </div>
  );
};

export default LiveMonitorPage;
