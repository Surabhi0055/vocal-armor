import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import VAIcon from './VAIcon';

const Sidebar = () => {
  const { user } = useAuthStore();
  const navigate = useNavigate();
  
  const fullName = user?.full_name || 'Admin User';
  const email = user?.email || 'admin@vocalarmor.com';
  
  const nameParts = fullName.trim().split(' ');
  const first = nameParts[0] || '';
  const last = nameParts.length > 1 ? nameParts[nameParts.length - 1] : '';
  const initials = ((first[0] || '') + (last[0] || '')).toUpperCase() || 'VA';

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-icon">
          <VAIcon size={32} style={{ borderRadius: '8px' }} />
        </div>
        <span className="sidebar-logo-text" style={{ fontFamily: '"Space Grotesk", sans-serif', letterSpacing: '1px', fontSize: '13px' }}><span style={{ fontWeight: 800, color: 'var(--text-main)' }}>VOCAL</span><span style={{ fontWeight: 300, color: '#C6A75E' }}>ARMOR</span></span>
      </div>

      <div className="sidebar-top">
        <NavLink to="/" end className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-home"></i>
          <span className="side-text">Detector</span>
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-history"></i>
          <span className="side-text">History</span>
        </NavLink>
        <NavLink to="/batch" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-layout-list"></i>
          <span className="side-text">Batch Upload</span>
        </NavLink>
        <NavLink to="/live" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-activity"></i>
          <span className="side-text">Live Monitor</span>
        </NavLink>
      </div>

      <div className="sidebar-bottom">
        <NavLink to="/user" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-settings"></i>
          <span className="side-text">Settings</span>
        </NavLink>
        <a href="mailto:bharalesurabhi05@gmail.com?subject=VocalArmor Support Request" className="side-icon-box" style={{ marginBottom: '16px' }}>
          <i className="ti ti-help"></i>
          <span className="side-text">Support</span>
        </a>
        
        {/* Profile Details Card */}
        <div style={{ width: '100%', overflow: 'hidden', padding: '16px 0 0 0', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          
          <div onClick={() => navigate('/user')} style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', padding: '6px', borderRadius: '30px', width: '90%', transition: 'all 0.3s', cursor: 'pointer', overflow: 'hidden' }}>
            {user?.avatar_url ? (
              <img src={user.avatar_url} alt="Profile" style={{ minWidth: '36px', height: '36px', borderRadius: '50%', objectFit: 'cover', boxShadow: '0 4px 10px rgba(232,220,200,0.3)', flexShrink: 0 }} />
            ) : (
              <div style={{ minWidth: '36px', height: '36px', borderRadius: '50%', background: 'linear-gradient(135deg, #C6A75E, #7A2E32)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: '14px', color: 'var(--text-main)', fontFamily: '"Bebas Neue", sans-serif', letterSpacing: '1px', boxShadow: '0 4px 10px rgba(232,220,200,0.3)', flexShrink: 0 }}>
                {initials}
              </div>
            )}
            <div className="side-text" style={{ margin: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{fullName}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{email}</span>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
