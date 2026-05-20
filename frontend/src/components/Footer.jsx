import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import VAIcon from './VAIcon';

const Footer = () => {
  const location = useLocation();

  const navLinks = [
    { label: 'Detector', path: '/' },
    { label: 'Live Monitor', path: '/live' },
    { label: 'Batch Upload', path: '/batch' },
    { label: 'History', path: '/history' },
    { label: 'Settings', path: '/user' },
  ];

  return (
    <footer style={{
      marginTop: '80px',
      borderTop: '1px solid rgba(255,255,255,0.06)',
      paddingTop: '30px',
      paddingBottom: '20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '20px',
      width: '100%',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }} onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
        <VAIcon size={22} style={{ borderRadius: '4px' }} />
        <span style={{ fontFamily: '"Space Grotesk", sans-serif', fontSize: '14px', letterSpacing: '1px', color: '#fff' }}>
          <span style={{ fontWeight: 800 }}>VOCAL</span>
          <span style={{ fontWeight: 300, color: '#1dcfcf' }}>ARMOR</span>
        </span>
        <span style={{ fontFamily: '"DM Mono", monospace', fontSize: '11px', color: '#7a8f94', marginLeft: '6px' }}>· MIT License</span>
      </div>

      <div style={{ display: 'flex', gap: '26px', flexWrap: 'wrap' }}>
        {navLinks.map((link) => {
          const isActive = location.pathname === link.path;
          return (
            <Link
              key={link.label}
              to={link.path}
              style={{
                fontFamily: "'Space Grotesk', sans-serif",
                fontSize: '13px',
                color: isActive ? '#1dcfcf' : '#7a8f94',
                textDecoration: 'none',
                cursor: 'pointer',
                fontWeight: isActive ? '600' : '400',
                transition: 'all .2s ease',
              }}
              onMouseOver={e => {
                if (!isActive) e.currentTarget.style.color = '#fff';
              }}
              onMouseOut={e => {
                if (!isActive) e.currentTarget.style.color = '#7a8f94';
              }}
            >
              {link.label}
            </Link>
          );
        })}
      </div>

      <span style={{ fontFamily: '"DM Mono", monospace', fontSize: '11px', color: 'rgba(29,207,207,0.45)', letterSpacing: '0.05em' }}>
        Built with TensorFlow · FastAPI
      </span>
    </footer>
  );
};

export default Footer;
