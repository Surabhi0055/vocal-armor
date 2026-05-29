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
          <span style={{ fontFamily: '"Space Grotesk", sans-serif', fontSize: '18px', letterSpacing: '1px', color: 'var(--text-main)' }}><span style={{ fontWeight: 800 }}>VOCAL</span><span style={{ fontWeight: 300, color: '#C6A75E' }}>ARMOR</span></span>
        </Link>
      </div>
      
      {/* Center side text navigation */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '32px', flex: 1 }}>
        <Link to="/" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = 'var(--text-muted)'}>Detector</Link>
        <Link to="/history" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = 'var(--text-muted)'}>History</Link>
        <Link to="/batch" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = 'var(--text-muted)'}>Batch Upload</Link>
        <Link to="/live" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '15px', fontWeight: 500, transition: 'color 0.2s' }} onMouseOver={(e) => e.target.style.color = 'white'} onMouseOut={(e) => e.target.style.color = 'var(--text-muted)'}>Live Monitor</Link>
      </div>

      {/* Right side navigation grouping */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: 1, justifyContent: 'flex-end' }}>
        
        {/* Dynamic Search Bar */}
        <div 
          className="search-box" 
          style={{
            display: 'flex',
            alignItems: 'center',
            background: isSearchActive ? 'rgba(123, 157, 174, 0.10)' : 'rgba(255, 255, 255, 0.03)',
            border: isSearchActive ? '1px solid rgba(123, 157, 174, 0.40)' : '1px solid rgba(255, 255, 255, 0.1)',
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
          <i className="ti ti-search" style={{ color: isSearchActive ? '#C6A75E' : 'var(--text-muted)', fontSize: '18px', minWidth: '18px' }}></i>
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
              color: 'var(--text-main)',
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
              color: 'var(--text-muted)',
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
              <div style={{ position: 'absolute', top: '8px', right: '10px', width: '8px', height: '8px', background: '#C6A75E', borderRadius: '50%', boxShadow: '0 0 10px rgba(123,157,174,0.8)' }}></div>
            )}
          </button>
          
          {showNotifications && (
            <div className="dropdown-menu" style={{ position: 'absolute', top: '50px', right: '0', width: '300px', background: '#211816', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', boxShadow: '0 10px 40px rgba(232,220,200,0.6)', zIndex: 100, overflow: 'hidden' }}>
              <div style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Notifications</span>
                <span style={{ fontSize: '11px', background: 'rgba(123,157,174,0.12)', color: '#C6A75E', padding: '2px 8px', borderRadius: '100px' }}>{notifications.length} New</span>
              </div>
              
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {notifications.length === 0 ? (
                  <div style={{ padding: '30px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                    <i className="ti ti-bell-off" style={{ fontSize: '24px', opacity: 0.5, marginBottom: '8px', display: 'block' }}></i>
                    No recent deepfake alerts.
                  </div>
                ) : (
                  notifications.map((notif, idx) => (
                    <div style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', gap: '12px', background: 'rgba(122,46,50,0.06)' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#A63A3F', marginTop: '6px', flexShrink: 0, boxShadow: '0 0 8px rgba(122,46,50,0.7)' }}></div>
                      <div style={{ overflow: 'hidden' }}>
                        <div style={{ fontSize: '13px', color: 'var(--text-main)', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ fontWeight: 600 }}>Deepfake Detected</span>
                          <span style={{ color: '#A63A3F', fontWeight: 600 }}>{notif.confidence.toFixed(1)}%</span>
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
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
                  style={{ padding: '12px', textAlign: 'center', fontSize: '12px', color: '#C6A75E', cursor: 'pointer', background: 'rgba(123,157,174,0.05)', borderTop: '1px solid rgba(123,157,174,0.12)' }}
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
              color: 'var(--text-main)',
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
              <div style={{ width: '30px', height: '30px', borderRadius: '50%', background: 'linear-gradient(135deg, #C6A75E, #7A2E32)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', fontWeight: 800, fontFamily: '"Bebas Neue", sans-serif' }}>
                {initials}
              </div>
            )}
            <i className="ti ti-chevron-down" style={{ fontSize: '14px', color: 'var(--text-muted)' }}></i>
          </button>

          {showProfileMenu && (
            <div className="dropdown-menu" style={{ position: 'absolute', top: '50px', right: '0', width: '200px', background: '#211816', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', boxShadow: '0 10px 40px rgba(232,220,200,0.6)', zIndex: 100, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <Link to="/user" style={{ padding: '12px 16px', color: 'var(--text-main)', textDecoration: 'none', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' }} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                <i className="ti ti-user" style={{ fontSize: '16px', color: 'var(--text-muted)' }}></i> My Profile
              </Link>
              <Link to="/user" style={{ padding: '12px 16px', color: 'var(--text-main)', textDecoration: 'none', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' }} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                <i className="ti ti-settings" style={{ fontSize: '16px', color: 'var(--text-muted)' }}></i> Preferences
              </Link>
              <div style={{ width: '100%', height: '1px', background: 'rgba(255,255,255,0.1)' }}></div>
              <div 
                onClick={() => { logout(); navigate('/login'); }} 
                style={{ padding: '12px 16px', color: '#A63A3F', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' }} 
                onMouseOver={(e) => e.target.style.background = 'rgba(122,46,50,0.12)'} 
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