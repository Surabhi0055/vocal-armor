import React from 'react';
import { NavLink } from 'react-router-dom';
import { useUser } from '../UserContext';

const Sidebar = () => {
  const { profile } = useUser();
  const initials = (profile.firstName[0] + profile.lastName[0]).toUpperCase();

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-icon">
          <i className="ti ti-waveform"></i>
        </div>
        <span className="sidebar-logo-text">VOCAL<span style={{fontWeight: 300}}>ARMOR</span></span>
      </div>

      <div className="sidebar-top">
        <NavLink to="/" end className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-home"></i>
          <span className="side-text">Detector</span>
          <span className="sidebar-badge badge-orange" style={{marginLeft: 'auto'}}>v3</span>
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-history"></i>
          <span className="side-text">History</span>
          <span className="sidebar-badge badge-cyan" style={{marginLeft: 'auto'}}>24</span>
        </NavLink>
        <NavLink to="/batch" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-layout-list"></i>
          <span className="side-text">Batch Upload</span>
        </NavLink>
        <NavLink to="/live" className={({ isActive }) => `side-icon-box ${isActive ? 'active' : ''}`}>
          <i className="ti ti-activity"></i>
          <span className="side-text">Live Monitor</span>
          <span className="sidebar-badge badge-orange" style={{marginLeft: 'auto'}}>NEW</span>
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
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', padding: '6px', borderRadius: '30px', width: '90%', transition: 'all 0.3s', cursor: 'pointer' }}>
            <div style={{ minWidth: '36px', height: '36px', borderRadius: '50%', background: 'linear-gradient(135deg, #00d4c8, #0088ff)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: '14px', color: 'white', fontFamily: '"Bebas Neue", sans-serif', letterSpacing: '1px', boxShadow: '0 4px 10px rgba(0,0,0,0.3)' }}>
              {initials}
            </div>
            <div className="side-text" style={{ margin: 0, display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '14px', fontWeight: 600, color: 'white' }}>{profile.firstName} {profile.lastName}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{profile.email}</span>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
