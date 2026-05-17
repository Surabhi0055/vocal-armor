import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { UserProvider } from './UserContext';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import HistoryPage from './components/HistoryPage';
import UserPage from './components/UserPage';
import LiveMonitorPage from './components/LiveMonitorPage';
import BatchUploadPage from './components/BatchUploadPage';
import WaveformBackground from './components/WaveformBackground';
import CustomCursor from './components/CustomCursor';

function App() {
  return (
    <UserProvider>
      <BrowserRouter>
        <CustomCursor />
      <WaveformBackground />

      <Sidebar />

      <div className="main-content">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/batch" element={<BatchUploadPage />} />
          <Route path="/live" element={<LiveMonitorPage />} />
          <Route path="/user" element={<UserPage />} />
        </Routes>
      </div>
    </BrowserRouter>
    </UserProvider>
  )
}

export default App;
