import React from 'react';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-icon">
          <i className="ti ti-waveform"></i>
        </div>
        <span className="sidebar-logo-text">VOCAL<span style={{fontWeight: 300}}>ARMOR</span></span>
      </div>

      <div className="sidebar-top">
        <a href="#" className="side-icon-box active">
          <i className="ti ti-home"></i>
          <span className="side-text">Detector</span>
          <span className="sidebar-badge badge-orange" style={{marginLeft: 'auto'}}>v3</span>
        </a>
        <a href="#" className="side-icon-box">
          <i className="ti ti-history"></i>
          <span className="side-text">History</span>
          <span className="sidebar-badge badge-cyan" style={{marginLeft: 'auto'}}>24</span>
        </a>
        <a href="#" className="side-icon-box">
          <i className="ti ti-layout-list"></i>
          <span className="side-text">Batch Upload</span>
        </a>
        <a href="#" className="side-icon-box">
          <i className="ti ti-activity"></i>
          <span className="side-text">Live Monitor</span>
          <span className="sidebar-badge badge-orange" style={{marginLeft: 'auto'}}>NEW</span>
        </a>
      </div>

      <div className="sidebar-bottom">
        <a href="#" className="side-icon-box">
          <i className="ti ti-settings"></i>
          <span className="side-text">Settings</span>
        </a>
        <a href="#" className="side-icon-box">
          <i className="ti ti-help"></i>
          <span className="side-text">Support</span>
        </a>
      </div>
    </div>
  );
};

export default Sidebar;
