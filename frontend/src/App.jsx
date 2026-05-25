import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import AuthPage from './pages/AuthPage';
import AuthCallback from './pages/AuthCallback';
import LandingPage from './pages/LandingPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import ProtectedRoute from './components/ProtectedRoute';

// Existing layout core components
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import WaveformBackground from './components/WaveformBackground';
import CustomCursor from './components/CustomCursor';

// Existing views
import Dashboard from './components/Dashboard';
import HistoryPage from './components/HistoryPage';
import BatchUploadPage from './components/BatchUploadPage';
import LiveMonitorPage from './components/LiveMonitorPage';
import UserPage from './components/UserPage';

function AppLayout() {
  return (
    <>
      {/* Custom magnetic UI cursor tracker */}
      <CustomCursor />
      
      {/* Waveform dynamic background animation */}
      <WaveformBackground />
      
      {/* Expanding navigation drawer */}
      <Sidebar />
      
      {/* Main viewport frame with active route switching */}
      <div className="main-content">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/batch" element={<BatchUploadPage />} />
          <Route path="/live" element={<LiveMonitorPage />} />
          <Route path="/user" element={<UserPage />} />
          {/* Default fallback redirects to base Dashboard */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Open Authentication screens */}
        <Route path="/start" element={<LandingPage />} />
        <Route path="/login" element={<AuthPage />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
        
        {/* Secure dashboard workspace gating */}
        <Route path="/*" element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        } />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
