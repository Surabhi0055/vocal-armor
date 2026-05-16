import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <div className="navbar">
      <div className="nav-left">
        <div className="nav-breadcrumbs">
          <span style={{color: 'var(--text-main)', fontWeight: 700}}>VocalArmor</span>
        </div>
      </div>

      <div className="nav-center">
        <div className="nav-status">
          <div className="status-pill">Uptime <span>99.9%</span></div>
          <div className="status-pill">Latency <span>&lt;1.3s</span></div>
        </div>

        <div style={{width: 1, height: 24, background: 'var(--border-color)', margin: '0 24px'}}></div>

        <div className="search-box">
          <div className="search-box-left">
            <i className="ti ti-search" style={{fontSize: 16}}></i>
            <span>Search voices, results...</span>
          </div>
          <div className="search-box-shortcut">⌘K</div>
        </div>
      </div>

      <div className="nav-right">
        <div className="live-badge">
          <div className="live-dot"></div>
          LIVE - v3.1
        </div>

        <div className="nav-icons">
          <div className="icon-btn" style={{position: 'relative', cursor: 'pointer'}}>
            <i className="ti ti-bell"></i>
            <div style={{position: 'absolute', top: 6, right: 6, width: 6, height: 6, background: '#f25c2c', borderRadius: '50%'}}></div>
          </div>
          <div className="icon-btn" style={{cursor: 'pointer'}}><i className="ti ti-adjustments-horizontal"></i></div>
          <Link to="/user" className="icon-btn" style={{textDecoration: 'none', cursor: 'pointer'}}><i className="ti ti-user-circle"></i></Link>
        </div>
      </div>
    </div>
  );
};

export default Navbar;