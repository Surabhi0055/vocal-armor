import React, { useEffect, useState } from 'react';

const CustomCursor = () => {
  const [position, setPosition] = useState({ x: -100, y: -100 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    const handleMouseOver = (e) => {
      if (e.target.closest('a, button, .icon-btn, .side-icon-box, input, .upload-zone, .file-pill')) {
        setIsHovering(true);
      }
    };

    const handleMouseOut = (e) => {
      if (e.target.closest('a, button, .icon-btn, .side-icon-box, input, .upload-zone, .file-pill')) {
        setIsHovering(false);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseout', handleMouseOut);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseout', handleMouseOut);
    };
  }, []);

  return (
    <div 
      style={{
        position: 'fixed',
        left: position.x,
        top: position.y,
        pointerEvents: 'none',
        zIndex: 9999,
        transform: 'translate(-50%, -50%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      <div style={{
        width: isHovering ? 64 : 32, 
        height: isHovering ? 64 : 32, 
        borderRadius: '50%', 
        border: isHovering ? '1px solid rgba(242, 92, 44, 0.8)' : '1px solid rgba(0, 209, 224, 0.4)',
        position: 'absolute',
        transition: 'all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
      }}></div>
      <div style={{
        width: 6, height: 6, borderRadius: '50%', backgroundColor: '#00f0ff',
        boxShadow: '0 0 10px #00f0ff'
      }}></div>
    </div>
  );
};

export default CustomCursor;
