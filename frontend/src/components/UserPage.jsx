import React from 'react';

const UserPage = () => {
  return (
    <div className="dashboard" style={{ padding: '40px' }}>
      <div className="hero-section" style={{ textAlign: 'left', marginBottom: '40px' }}>
        <h1 className="hero-title" style={{ fontSize: '36px' }}>
          USER <span className="val-cyan">PROFILE</span>
        </h1>
        <p className="hero-subtitle">Manage your account settings, API keys, and preferences.</p>
      </div>

      <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '32px', backdropFilter: 'blur(16px)', maxWidth: '600px' }}>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '40px' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(0, 209, 224, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '32px', color: '#00d4c8' }}>
            <i className="ti ti-user-circle"></i>
          </div>
          <div>
            <div style={{ fontSize: '24px', fontWeight: 600, color: 'white', marginBottom: '4px' }}>Admin User</div>
            <div style={{ color: '#7ea8a4', fontSize: '14px' }}>admin@vocalarmor.com</div>
          </div>
        </div>

        <div style={{ marginBottom: '32px' }}>
          <div style={{ fontSize: '12px', letterSpacing: '0.1em', color: '#7ea8a4', textTransform: 'uppercase', marginBottom: '16px', fontWeight: 600 }}>Account Details</div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '6px' }}>API Key</div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', color: 'white', fontFamily: 'monospace', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                va_live_98x1n2b3k4j5h6g7f8
                <i className="ti ti-copy" style={{ cursor: 'pointer', color: '#00d4c8' }}></i>
              </div>
            </div>

            <div>
              <div style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '6px' }}>Plan</div>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, color: '#f0a429' }}>Enterprise Tier</span>
                <span style={{ fontSize: '12px', background: 'rgba(240,164,41,0.1)', color: '#f0a429', padding: '4px 8px', borderRadius: '4px' }}>Active</span>
              </div>
            </div>
          </div>
        </div>

        <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }}>
          Save Changes
        </button>
      </div>
    </div>
  );
};

export default UserPage;
