import React from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';

function App() {
  return (
    <>
      {/* Background Ambient Orbs */}
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>
      <div className="orb orb-3"></div>
      
      <Sidebar />

      <div className="main-wrapper">
        <Navbar />
        
        <div className="content">
          <h1 className="hero-title">Detect AI Voices<br/>Before They Deceive</h1>
          <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
             Our frontend is successfully connected to React!
          </p>
        </div>
      </div>
    </>
  )
}

export default App;
