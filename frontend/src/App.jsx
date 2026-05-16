import React from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import WaveformBackground from './components/WaveformBackground';
import CustomCursor from './components/CustomCursor';

function App() {
  return (
    <>
      <CustomCursor />
      <WaveformBackground />

      <Sidebar />

      <div className="main-content">
        <Navbar />
        <Dashboard />
      </div>
    </>
  )
}

export default App;
