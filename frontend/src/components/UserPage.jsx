import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { getPrefs, savePrefs, getHistory } from '../utils/storage';
import Footer from './Footer';

const UserPage = () => {
  const { user, accessToken, updateUser } = useAuthStore();
  
  const fullName = user?.full_name || user?.username || 'User';
  const nameParts = fullName.trim().split(' ');
  const first = nameParts[0] || '';
  const last = nameParts.length > 1 ? nameParts[nameParts.length - 1] : '';

  const [avatarUploadLoading, setAvatarUploadLoading] = useState(false);

  const [editForm, setEditForm] = useState({
    firstName: first,
    lastName: last,
    email: user?.email || '',
    phone: ''
  });

  const [strictFilter, setStrictFilter] = useState(true);
  const [autoSave, setAutoSave] = useState(true);
  const [emailAlerts, setEmailAlerts] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [scanCount, setScanCount] = useState(0);

  // Load per-user prefs and scan count when user changes
  useEffect(() => {
    const prefs = getPrefs();
    setStrictFilter(prefs.strictFilter);
    setAutoSave(prefs.autoSave);
    setEmailAlerts(prefs.emailAlerts);
    setEditForm(prev => ({
      ...prev,
      firstName: first,
      lastName: last,
      email: user?.email || '',
      phone: prefs.phone || ''
    }));
    setScanCount(getHistory().length);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, user?.email]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSave = async () => {
    setIsSaving(true);
    
    // 1. Save preferences to local storage
    savePrefs({ strictFilter, autoSave, emailAlerts, phone: editForm.phone });
    
    // 2. Save profile data to backend
    try {
      const res = await fetch('http://localhost:8000/users/me', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          full_name: `${editForm.firstName} ${editForm.lastName}`.trim(),
          email: editForm.email,
          phone: editForm.phone
        })
      });
      if (res.ok) {
        const updatedUser = await res.json();
        updateUser(updatedUser);
      }
    } catch (e) {
      console.error("Failed to update profile", e);
    }
    
    setIsSaving(false);
  };

  const handleAvatarUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAvatarUploadLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/users/me/avatar', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        },
        body: formData
      });
      if (res.ok) {
        const updatedUser = await res.json();
        updateUser(updatedUser);
      }
    } catch (err) {
      console.error("Avatar upload failed", err);
    }
    setAvatarUploadLoading(false);
  };

  return (
    <div className="dashboard" style={{ paddingBottom: '100px' }}>
      
      <div style={{ marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontFamily: '"Bebas Neue", sans-serif', fontSize: '48px', fontWeight: 400, letterSpacing: '2px', lineHeight: 1, marginBottom: '12px', textTransform: 'uppercase' }}>
            USER <span style={{ color: '#00d4c8', textShadow: '0 0 40px rgba(0,212,200,0.4)' }}>PROFILE</span>
          </h1>
          <p style={{ fontSize: '14px', color: '#7ea8a4', lineHeight: 1.6 }}>
            Manage your account settings, personal info, and engine preferences.
          </p>
        </div>
        <Link to="/" style={{ background: 'transparent', color: '#7ea8a4', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', padding: '8px 16px', cursor: 'pointer', fontSize: '12px', letterSpacing: '1px', textDecoration: 'none', transition: 'all 0.2s' }} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
          RETURN HOME
        </Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
        {/* Left Column: Profile Overview */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '32px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            
            <label style={{ cursor: 'pointer', position: 'relative', display: 'block', marginBottom: '24px' }}>
              <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleAvatarUpload} disabled={avatarUploadLoading} />
              <div style={{ width: '120px', height: '120px', borderRadius: '50%', background: 'rgba(0, 212, 200, 0.1)', border: '2px solid rgba(0, 212, 200, 0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '48px', color: '#00d4c8', boxShadow: '0 0 40px rgba(0,212,200,0.2)', overflow: 'hidden', position: 'relative' }}>
                {user?.avatar_url ? (
                  <img src={user.avatar_url} alt="Profile" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <i className="ti ti-user-circle"></i>
                )}
                {/* Hover overlay */}
                <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: avatarUploadLoading ? 1 : 0, transition: 'opacity 0.2s' }} className="avatar-overlay">
                  {avatarUploadLoading ? <i className="ti ti-loader" style={{ animation: 'spin 1s linear infinite', color: '#00d4c8', fontSize: '24px' }}></i> : <i className="ti ti-camera" style={{ color: 'white', fontSize: '24px' }}></i>}
                </div>
              </div>
            </label>

            <div style={{ fontSize: '24px', fontWeight: 600, color: 'white', marginBottom: '8px' }}>{fullName}</div>
            <div style={{ color: '#7ea8a4', fontSize: '14px' }}>{user?.email || 'admin@vocalarmor.com'}</div>

            <div style={{ width: '100%', height: '1px', background: 'rgba(255,255,255,0.1)', margin: '32px 0' }}></div>

            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '16px' }}>
              <span style={{ color: '#7ea8a4', fontSize: '14px' }}>Total Scans</span>
              <span style={{ color: 'white', fontWeight: 600 }}>{scanCount.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <span style={{ color: '#7ea8a4', fontSize: '14px' }}>Detection Accuracy</span>
              <span style={{ color: '#00d4c8', fontWeight: 600 }}>99.8%</span>
            </div>
          </div>

          <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '24px' }}>
             <div style={{ fontSize: '14px', fontWeight: 600, color: 'white', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px' }}>Recent Activity</div>
             
             <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
               <div style={{ display: 'flex', gap: '12px' }}>
                 <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00d4c8', marginTop: '4px' }}></div>
                 <div>
                   <div style={{ fontSize: '12px', color: 'white', fontWeight: 500 }}>Batch Scan Completed</div>
                   <div style={{ fontSize: '10px', color: '#7ea8a4', marginTop: '2px' }}>Today, 14:23 PM</div>
                 </div>
               </div>
               
               <div style={{ display: 'flex', gap: '12px' }}>
                 <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#e8521e', marginTop: '4px' }}></div>
                 <div>
                   <div style={{ fontSize: '12px', color: 'white', fontWeight: 500 }}>Deepfake Detected (98%)</div>
                   <div style={{ fontSize: '10px', color: '#7ea8a4', marginTop: '2px' }}>Yesterday, 09:15 AM</div>
                 </div>
               </div>

               <div style={{ display: 'flex', gap: '12px' }}>
                 <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#7ea8a4', marginTop: '4px' }}></div>
                 <div>
                   <div style={{ fontSize: '12px', color: 'white', fontWeight: 500 }}>System Login</div>
                   <div style={{ fontSize: '10px', color: '#7ea8a4', marginTop: '2px' }}>May 15th, 08:00 AM</div>
                 </div>
               </div>
             </div>
          </div>

        </div>

        {/* Right Column: Settings */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Personal Details */}
          <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="ti ti-id" style={{ color: '#00d4c8' }}></i> Personal Details
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>First Name</label>
                  <input type="text" name="firstName" value={editForm.firstName} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
                </div>
                <div>
                  <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>Last Name</label>
                  <input type="text" name="lastName" value={editForm.lastName} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
                </div>
              </div>

              <div>
                <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>Email Address</label>
                <input type="email" name="email" value={editForm.email} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '8px', display: 'block' }}>Phone Number</label>
                <input type="text" name="phone" value={editForm.phone} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
              </div>
            </div>
          </div>

          {/* Engine Preferences */}
          <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="ti ti-settings" style={{ color: '#00d4c8' }}></i> Detection Preferences
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Strict False-Positive Filter</div>
                  <div style={{ fontSize: '12px', color: '#7ea8a4' }}>Increases threshold on v3 model to prevent flagging human voices.</div>
                </div>
                <div onClick={() => setStrictFilter(!strictFilter)} style={{ width: '44px', height: '24px', background: strictFilter ? '#00d4c8' : 'rgba(255,255,255,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}>
                  <div style={{ width: '20px', height: '20px', background: strictFilter ? '#0f2229' : '#7ea8a4', borderRadius: '50%', position: 'absolute', top: '2px', left: strictFilter ? '22px' : '2px', transition: 'left 0.3s' }}></div>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Auto-Save Analysis History</div>
                  <div style={{ fontSize: '12px', color: '#7ea8a4' }}>Keep a local record of all batch and live recordings in the History page.</div>
                </div>
                <div onClick={() => setAutoSave(!autoSave)} style={{ width: '44px', height: '24px', background: autoSave ? '#00d4c8' : 'rgba(255,255,255,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}>
                  <div style={{ width: '20px', height: '20px', background: autoSave ? '#0f2229' : '#7ea8a4', borderRadius: '50%', position: 'absolute', top: '2px', left: autoSave ? '22px' : '2px', transition: 'left 0.3s' }}></div>
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Email Alerts on High-Risk Audio</div>
                  <div style={{ fontSize: '12px', color: '#7ea8a4' }}>Get notified instantly if Batch Scanner detects 95%+ probability deepfakes.</div>
                </div>
                <div onClick={() => setEmailAlerts(!emailAlerts)} style={{ width: '44px', height: '24px', background: emailAlerts ? '#e8521e' : 'rgba(255,255,255,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}>
                  <div style={{ width: '20px', height: '20px', background: emailAlerts ? 'white' : '#7ea8a4', borderRadius: '50%', position: 'absolute', top: '2px', left: emailAlerts ? '22px' : '2px', transition: 'left 0.3s' }}></div>
                </div>
              </div>
            </div>
          </div>

          <button onClick={handleSave} style={{ background: isSaving ? '#0a2228' : '#00d4c8', color: isSaving ? '#00d4c8' : '#0f2229', border: isSaving ? '1px solid #00d4c8' : 'none', borderRadius: '8px', padding: '16px 24px', cursor: 'pointer', fontWeight: 700, letterSpacing: '1px', boxShadow: isSaving ? 'none' : '0 4px 16px rgba(0,212,200,0.3)', width: '100%', marginTop: 'auto', transition: 'all 0.2s', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }} disabled={isSaving}>
            {isSaving ? <><i className="ti ti-loader" style={{ animation: 'spin 1s linear infinite' }}></i> SAVING CHANGES...</> : 'SAVE ALL CHANGES'}
          </button>

        </div>
      </div>
      
      <Footer />
      
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        label:hover .avatar-overlay { opacity: 1 !important; }
      `}</style>
    </div>
  );
};

export default UserPage;
