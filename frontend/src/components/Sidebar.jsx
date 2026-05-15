import React from 'react';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="side-icon active">
        <i className="ti ti-home"></i>
        <span>Home</span>
      </div>
      <div className="side-icon">
        <i className="ti ti-chart-bar"></i>
        <span>Analyze</span>
      </div>
      <div className="side-icon">
        <i className="ti ti-history"></i>
        <span>History</span>
      </div>
      <div className="side-icon">
        <i className="ti ti-code"></i>
        <span>API</span>
      </div>
      
      <div className="side-bottom">
        <div className="side-icon">
          <i className="ti ti-settings"></i>
          <span>Settings</span>
        </div>
        <div className="side-icon">
          <i className="ti ti-help"></i>
          <span>Help</span>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
