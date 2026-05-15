import React from 'react';

const Navbar = () => {
  return (
    <div className="navbar">
        <div className="logo">
            <div className="logo-icon"><i className="ti ti-waveform"></i></div>
            VocalArmor
        </div>
        <div className="nav-links">
            <span>How it Works</span>
            <span>Accuracy</span>
            <span>API</span>
            <span>Docs</span>
            <span>Blog</span>
            <span>Pricing</span>
        </div>
        <div className="nav-actions">
            <div className="avatar">
                <img src="https://i.pravatar.cc/150?u=a042581f4e29026704d" alt="User Profile" />
            </div>
            <button className="btn-primary">Get Started</button>
        </div>
    </div>
  );
};

export default Navbar;
