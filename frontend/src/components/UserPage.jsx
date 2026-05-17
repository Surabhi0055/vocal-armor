import React from 'react';
import { Link } from 'react-router-dom';

const UserPage = () => {
  return (
    <div style={{ padding: '40px', maxWidth: '1000px', margin: '0 auto', width: '100%', zIndex: 2, position: 'relative' }}>
      
      <div style={{ marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontFamily: '"Bebas Neue", sans-serif', fontSize: '48px', fontWeight: 400, letterSpacing: '2px', lineHeight: 1, marginBottom: '12px', textTransform: 'uppercase' }}>
            USER <span style={{ color: '#00d4c8', textShadow: '0 0 40px rgba(0,212,200,0.4)' }}>PROFILE</span>
          </h1>
          <p style={{ fontSize: '14px', color: '#7ea8a4', lineHeight: 1.6 }}>
            Manage your account settings, API keys, and engine preferences.
          </p>
        </div>
        <Link to="/" style={{ background: 'transparent', color: '#7ea8a4', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', padding: '8px 16px', cursor: 'pointer', fontSize: '12px', letterSpacing: '1px', textDecoration: 'none' }}>
          RETURN HOME
        </Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
        {/* Left Column: Profile Overview */}
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '32px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
          <div style={{ width: '120px', height: '120px', borderRadius: '50%', background: 'rgba(0, 212, 200, 0.1)', border: '2px solid rgba(0, 212, 200, 0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '48px', color: '#00d4c8', marginBottom: '24px', boxShadow: '0 0 40px rgba(0,212,200,0.2)' }}>
            <i className="ti ti-user-circle"></i>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 600, color: 'white', marginBottom: '8px' }}>Admin User</div>
          <div style={{ color: '#7ea8a4', fontSize: '14px', marginBottom: '24px' }}>admin@vocalarmor.com</div>
          
          <div style={{ background: 'rgba(232, 82, 30, 0.1)', color: '#e8521e', padding: '6px 16px', borderRadius: '100px', fontSize: '12px', fontWeight: 700, letterSpacing: '1px', border: '1px solid rgba(232, 82, 30, 0.3)' }}>
            ENTERPRISE ADMIN
          </div>

          <div style={{ width: '100%', height: '1px', background: 'rgba(255,255,255,0.1)', margin: '32px 0' }}></div>

          <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '16px' }}>
            <span style={{ color: '#7ea8a4', fontSize: '14px' }}>Total Scans</span>
            <span style={{ color: 'white', fontWeight: 600 }}>1,248</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '16px' }}>
            <span style={{ color: '#7ea8a4', fontSize: '14px' }}>Detection Accuracy</span>
            <span style={{ color: '#00d4c8', fontWeight: 600 }}>99.8%</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <span style={{ color: '#7ea8a4', fontSize: '14px' }}>Plan</span>
            <span style={{ color: '#f0a429', fontWeight: 600 }}>Unlimited</span>
          </div>
        </div>

        {/* Right Column: Settings */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Personal Details */}
          <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="ti ti-id" style={{ color: '#00d4c8' }}></i> Personal Details
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>First Name</label>
                  <input type="text" defaultValue="Admin" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
                </div>
                <div>
                  <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>Last Name</label>
                  <input type="text" defaultValue="User" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
                </div>
              </div>

              <div>
                <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>Email Address</label>
                <input type="email" defaultValue="admin@vocalarmor.com" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>Organization / Company</label>
                <input type="text" defaultValue="VocalArmor Engine Security" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
              </div>
            </div>
          </div>

          {/* Engine Preferences */}
          <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="ti ti-settings" style={{ color: '#00d4c8' }}></i> Detection Preferences
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Strict False-Positive Filter</div>
                  <div style={{ fontSize: '12px', color: '#7ea8a4' }}>Increases threshold on v3 model to prevent flagging human voices.</div>
                </div>
                <div style={{ width: '44px', height: '24px', background: '#00d4c8', borderRadius: '12px', position: 'relative', cursor: 'pointer' }}>
                  <div style={{ width: '20px', height: '20px', background: '#0f2229', borderRadius: '50%', position: 'absolute', top: '2px', right: '2px' }}></div>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Auto-Save Analysis History</div>
                  <div style={{ fontSize: '12px', color: '#7ea8a4' }}>Keep a local record of all batch and live recordings in the History page.</div>
                </div>
                <div style={{ width: '44px', height: '24px', background: '#00d4c8', borderRadius: '12px', position: 'relative', cursor: 'pointer' }}>
                  <div style={{ width: '20px', height: '20px', background: '#0f2229', borderRadius: '50%', position: 'absolute', top: '2px', right: '2px' }}></div>
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Email Alerts on High-Risk Audio</div>
                  <div style={{ fontSize: '12px', color: '#7ea8a4' }}>Get notified instantly if Batch Scanner detects 95%+ probability deepfakes.</div>
                </div>
                <div style={{ width: '44px', height: '24px', background: 'rgba(255,255,255,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer' }}>
                  <div style={{ width: '20px', height: '20px', background: '#7ea8a4', borderRadius: '50%', position: 'absolute', top: '2px', left: '2px' }}></div>
                </div>
              </div>
            </div>
          </div>

          <button style={{ background: '#00d4c8', color: '#0f2229', border: 'none', borderRadius: '8px', padding: '16px 24px', cursor: 'pointer', fontWeight: 700, letterSpacing: '1px', boxShadow: '0 4px 16px rgba(0,212,200,0.3)', width: '100%', marginTop: 'auto' }}>
            SAVE ALL CHANGES
          </button>

        </div>
      </div>
    </div>
  );
};

export default UserPage;
