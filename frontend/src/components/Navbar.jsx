import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const Navbar = () => {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const notifRef = useRef(null);
  const settingsRef = useRef(null);
  const navigate = useNavigate();

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (notifRef.current && !notifRef.current.contains(event.target)) {
        setShowNotifications(false);
      }
      if (settingsRef.current && !settingsRef.current.contains(event.target)) {
        setShowSettings(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = (e) => {
    setSearchQuery(e.target.value);
  };

  const executeSearch = (e) => {
    if (e.key === 'Enter' && searchQuery.trim() !== '') {
      // In a real app, this would route to a search results page
      console.log('Searching for:', searchQuery);
      alert(`Searching for: ${searchQuery}`);
      setSearchQuery('');
    }
  };

  return (
    <div className="navbar">
      <div className="nav-left">
        <div className="nav-breadcrumbs">
          <span style={{color: 'var(--text-main)', fontWeight: 700, fontSize: '18px', letterSpacing: '1px'}}>VocalArmor</span>
        </div>
      </div>

      <div className="nav-right" style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
        
        {/* Interactive Search Bar */}
        <div className="search-box" style={{ padding: '0 16px', display: 'flex', alignItems: 'center', background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <i className="ti ti-search" style={{fontSize: 16, color: 'var(--text-muted)'}}></i>
          <input 
            type="text" 
            placeholder="Search voices, results..." 
            value={searchQuery}
            onChange={handleSearch}
            onKeyDown={executeSearch}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-main)',
              padding: '8px 12px',
              width: '200px',
              outline: 'none',
              fontSize: '13px'
            }}
          />
          <div className="search-box-shortcut" style={{ marginLeft: 'auto' }}>⌘K</div>
        </div>

        <div className="nav-icons" style={{ display: 'flex', gap: '12px' }}>
          
          {/* Notifications Dropdown */}
          <div ref={notifRef} style={{ position: 'relative' }}>
            <div className="icon-btn" style={{position: 'relative', cursor: 'pointer', background: showNotifications ? 'rgba(255,255,255,0.1)' : ''}} onClick={() => { setShowNotifications(!showNotifications); setShowSettings(false); }}>
              <i className="ti ti-bell"></i>
              <div style={{position: 'absolute', top: 6, right: 6, width: 6, height: 6, background: '#f25c2c', borderRadius: '50%'}}></div>
            </div>
            
            {showNotifications && (
              <div style={{ position: 'absolute', top: '120%', right: 0, width: '300px', background: '#0f2229', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', padding: '16px', boxShadow: '0 10px 40px rgba(0,0,0,0.5)', zIndex: 100 }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: 'white', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>Notifications</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                    <div style={{ width: '8px', height: '8px', background: '#e8521e', borderRadius: '50%', marginTop: '6px' }}></div>
                    <div>
                      <div style={{ fontSize: '13px', color: 'white', fontWeight: 600 }}>High Risk Audio Detected</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>2 mins ago • Batch Scan</div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                    <div style={{ width: '8px', height: '8px', background: '#00d4c8', borderRadius: '50%', marginTop: '6px' }}></div>
                    <div>
                      <div style={{ fontSize: '13px', color: 'white', fontWeight: 600 }}>System Update v3.1</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>New false-positive filters applied.</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Settings Dropdown */}
          <div ref={settingsRef} style={{ position: 'relative' }}>
            <div className="icon-btn" style={{cursor: 'pointer', background: showSettings ? 'rgba(255,255,255,0.1)' : ''}} onClick={() => { setShowSettings(!showSettings); setShowNotifications(false); }}>
              <i className="ti ti-adjustments-horizontal"></i>
            </div>
            
            {showSettings && (
              <div style={{ position: 'absolute', top: '120%', right: 0, width: '200px', background: '#0f2229', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', padding: '8px', boxShadow: '0 10px 40px rgba(0,0,0,0.5)', zIndex: 100, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ padding: '8px 12px', fontSize: '13px', color: 'var(--text-main)', cursor: 'pointer', borderRadius: '6px', transition: 'background 0.2s' }} onClick={() => navigate('/user')} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                  <i className="ti ti-user" style={{ marginRight: '8px' }}></i> User Profile
                </div>
                <div style={{ padding: '8px 12px', fontSize: '13px', color: 'var(--text-main)', cursor: 'pointer', borderRadius: '6px', transition: 'background 0.2s' }} onClick={() => navigate('/user')} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                  <i className="ti ti-key" style={{ marginRight: '8px' }}></i> API Keys
                </div>
                <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '4px 0' }}></div>
                <div style={{ padding: '8px 12px', fontSize: '13px', color: '#e8521e', cursor: 'pointer', borderRadius: '6px', transition: 'background 0.2s' }} onMouseOver={(e) => e.target.style.background = 'rgba(232,82,30,0.1)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                  <i className="ti ti-logout" style={{ marginRight: '8px' }}></i> Sign Out
                </div>
              </div>
            )}
          </div>

          {/* Profile Icon */}
          <Link to="/user" className="icon-btn" style={{textDecoration: 'none', cursor: 'pointer'}}><i className="ti ti-user-circle"></i></Link>
        </div>

      </div>
    </div>
  );
};

export default Navbar;