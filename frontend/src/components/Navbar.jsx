import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { getHistory } from '../utils/storage';
import VAIcon from './VAIcon';

const timeAgo = (dateStr) => {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
};

const Navbar = () => {
  const [isSearchActive, setIsSearchActive] = useState(false);
  const [searchValue, setSearchValue] = useState('');
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [notifications, setNotifications] = useState([]);
  
  const searchInputRef = useRef(null);
  
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  
  const fullName = user?.full_name || 'Admin User';
  const nameParts = fullName.trim().split(' ');
  const first = nameParts[0] || '';
  const last = nameParts.length > 1 ? nameParts[nameParts.length - 1] : '';
  const initials = ((first[0] || '') + (last[0] || '')).toUpperCase() || 'VA';

  // Handle clicking outside of dropdowns to close them
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.nav-icon-btn') && !event.target.closest('.dropdown-menu')) {
        setShowNotifications(false);
        setShowProfileMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Load real notifications from history
  const loadNotifications = () => {
    const hist = getHistory();
    // Get recent deepfake detections
    const fakes = hist.filter(h => h.prediction === 'FAKE');
    // Sort descending by timestamp
    fakes.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    setNotifications(fakes.slice(0, 5)); // top 5
  };

  useEffect(() => {
    loadNotifications();
    window.addEventListener('va_history_updated', loadNotifications);
    return () => window.removeEventListener('va_history_updated', loadNotifications);
  }, []);

  const handleSearchClick = () => {
    setIsSearchActive(true);
    setTimeout(() => searchInputRef.current?.focus(), 100);
  };

  const handleSearchBlur = () => {
    if (!searchValue) {
      setIsSearchActive(false);
    }
  };

  const executeSearch = (e) => {
    if (e.key === 'Enter' && searchValue.trim()) {
      navigate('/history', { state: { globalSearch: searchValue.trim() } });
      setShowNotifications(false);
      setShowProfileMenu(false);
      setIsSearchActive(false);
      setSearchValue('');
      setTimeout(() => {
        // Just unfocus the input so it shrinks back
        searchInputRef.current?.blur();
      }, 50);
    }
  };

  return (
    <div className="navbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 32px' }}>
      
      {/* Left side heading */}
      <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-start' }}>
        <Link to="/start" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontFamily: '"Space Grotesk", sans-serif', fontSize: '18px', letterSpacing: '1px', color: '#fff' }}><span style={{ fontWeight: 800 }}>VOCAL</span><span style={{ fontWeight: 300, color: '#1dcfcf' }}>ARMOR</span></span>
        </Link>
      </div>
      
      {/* Center side text navigation */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '32px', flex: 1 }}>
        <Link to="/" style={{ color: '#7ea8a4', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = '#7ea8a4'}>Detector</Link>
        <Link to="/history" style={{ color: '#7ea8a4', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = '#7ea8a4'}>History</Link>
        <Link to="/batch" style={{ color: '#7ea8a4', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = '#7ea8a4'}>Batch Upload</Link>
        <Link to="/live" style={{ color: '#7ea8a4', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = '#7ea8a4'}>Live Monitor</Link>
      </div>

      {/* Right side navigation grouping */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: 1, justifyContent: 'flex-end' }}>
        
        {/* Dynamic Search Bar */}
        <div 
          className="search-box" 
          style={{
            display: 'flex',
            alignItems: 'center',
            background: isSearchActive ? 'rgba(0, 212, 200, 0.1)' : 'rgba(255, 255, 255, 0.03)',
            border: isSearchActive ? '1px solid rgba(0, 212, 200, 0.4)' : '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '24px',
            padding: isSearchActive ? '8px 16px' : '8px',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            width: isSearchActive ? '250px' : '40px',
            height: '40px',
            cursor: isSearchActive ? 'text' : 'pointer',
            overflow: 'hidden'
          }}
          onClick={!isSearchActive ? handleSearchClick : undefined}
        >
          <i className="ti ti-search" style={{ color: isSearchActive ? '#00d4c8' : '#7ea8a4', fontSize: '18px', minWidth: '18px' }}></i>
          <input 
            ref={searchInputRef}
            type="text" 
            placeholder="Search scans, dates, or IDs..."
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            onBlur={handleSearchBlur}
            onKeyDown={executeSearch}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'white',
              outline: 'none',
              marginLeft: '12px',
              width: '100%',
              opacity: isSearchActive ? 1 : 0,
              pointerEvents: isSearchActive ? 'auto' : 'none',
              transition: 'opacity 0.2s',
              fontSize: '14px'
            }}
          />
        </div>

        {/* Separator */}
        <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)' }}></div>

        {/* Notifications Dropdown */}
        <div style={{ position: 'relative' }}>
          <button 
            className="nav-icon-btn" 
            style={{ 
              background: showNotifications ? 'rgba(255,255,255,0.1)' : 'transparent',
              border: 'none',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: '#7ea8a4',
              transition: 'all 0.2s',
              position: 'relative'
            }}
            onClick={() => {
              setShowNotifications(!showNotifications);
              setShowProfileMenu(false);
            }}
          >
            <i className="ti ti-bell" style={{ fontSize: '20px' }}></i>
            {notifications.length > 0 && (
              <div style={{ position: 'absolute', top: '8px', right: '10px', width: '8px', height: '8px', background: '#00d4c8', borderRadius: '50%', boxShadow: '0 0 10px rgba(0,212,200,0.8)' }}></div>
            )}
          </button>
          
          {showNotifications && (
            <div className="dropdown-menu" style={{ position: 'absolute', top: '50px', right: '0', width: '300px', background: '#0a191e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)', zIndex: 100, overflow: 'hidden' }}>
              <div style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '14px', fontWeight: 600, color: 'white', display: 'flex', justifyContent: 'space-between' }}>
                <span>Notifications</span>
                <span style={{ fontSize: '11px', background: 'rgba(0,212,200,0.1)', color: '#00d4c8', padding: '2px 8px', borderRadius: '100px' }}>{notifications.length} New</span>
              </div>
              
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {notifications.length === 0 ? (
                  <div style={{ padding: '30px 16px', textAlign: 'center', color: '#7ea8a4', fontSize: '13px' }}>
                    <i className="ti ti-bell-off" style={{ fontSize: '24px', opacity: 0.5, marginBottom: '8px', display: 'block' }}></i>
                    No recent deepfake alerts.
                  </div>
                ) : (
                  notifications.map((notif, idx) => (
                    <div key={idx} style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', gap: '12px', background: 'rgba(232,82,30,0.05)' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#e8521e', marginTop: '6px', flexShrink: 0, boxShadow: '0 0 8px rgba(232,82,30,0.6)' }}></div>
                      <div style={{ overflow: 'hidden' }}>
                        <div style={{ fontSize: '13px', color: 'white', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ fontWeight: 600 }}>Deepfake Detected</span>
                          <span style={{ color: '#e8521e', fontWeight: 600 }}>{notif.confidence.toFixed(1)}%</span>
                        </div>
                        <div style={{ fontSize: '11px', color: '#7ea8a4', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {notif.filename}
                        </div>
                        <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.3)', marginTop: '8px' }}>{timeAgo(notif.timestamp)}</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              {notifications.length > 0 && (
                <div 
                  style={{ padding: '12px', textAlign: 'center', fontSize: '12px', color: '#00d4c8', cursor: 'pointer', background: 'rgba(0,212,200,0.05)', borderTop: '1px solid rgba(0,212,200,0.1)' }}
                  onClick={() => setNotifications([])}
                >
                  Clear all alerts
                </div>
              )}
            </div>
          )}
        </div>

        {/* Profile Settings Dropdown */}
        <div style={{ position: 'relative' }}>
          <button 
            className="nav-icon-btn" 
            style={{ 
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '20px',
              padding: '4px 12px 4px 4px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              cursor: 'pointer',
              color: 'white',
              transition: 'all 0.2s',
            }}
            onClick={() => {
              setShowProfileMenu(!showProfileMenu);
              setShowNotifications(false);
            }}
          >
            {user?.avatar_url ? (
              <img src={user.avatar_url} alt="Profile" style={{ width: '30px', height: '30px', borderRadius: '50%', objectFit: 'cover' }} />
            ) : (
              <div style={{ width: '30px', height: '30px', borderRadius: '50%', background: 'linear-gradient(135deg, #00d4c8, #0088ff)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', fontWeight: 800, fontFamily: '"Bebas Neue", sans-serif' }}>
                {initials}
              </div>
            )}
            <i className="ti ti-chevron-down" style={{ fontSize: '14px', color: '#7ea8a4' }}></i>
          </button>

          {showProfileMenu && (
            <div className="dropdown-menu" style={{ position: 'absolute', top: '50px', right: '0', width: '200px', background: '#0a191e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)', zIndex: 100, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <Link to="/user" style={{ padding: '12px 16px', color: 'white', textDecoration: 'none', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' }} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                <i className="ti ti-user" style={{ fontSize: '16px', color: '#7ea8a4' }}></i> My Profile
              </Link>
              <Link to="/user" style={{ padding: '12px 16px', color: 'white', textDecoration: 'none', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' }} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                <i className="ti ti-settings" style={{ fontSize: '16px', color: '#7ea8a4' }}></i> Preferences
              </Link>
              <div style={{ width: '100%', height: '1px', background: 'rgba(255,255,255,0.1)' }}></div>
              <div 
                onClick={() => { logout(); navigate('/login'); }} 
                style={{ padding: '12px 16px', color: '#e8521e', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' }} 
                onMouseOver={(e) => e.target.style.background = 'rgba(232,82,30,0.1)'} 
                onMouseOut={(e) => e.target.style.background = 'transparent'}
              >
                <i className="ti ti-logout" style={{ fontSize: '16px' }}></i> Log Out
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default Navbar;