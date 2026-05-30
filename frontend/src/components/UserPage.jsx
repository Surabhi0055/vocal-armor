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
  const [avatarPreview, setAvatarPreview] = useState(null);
  const [avatarMsg, setAvatarMsg] = useState(null); // { type: 'success'|'error', text: string }

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
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/users/me`, {
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

    // Show an immediate local preview regardless of backend status
    const localUrl = URL.createObjectURL(file);
    setAvatarPreview(localUrl);
    setAvatarMsg(null);
    setAvatarUploadLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/users/me/avatar`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        },
        body: formData
      });
      if (res.ok) {
        const updatedUser = await res.json();
        updateUser(updatedUser);
        // Replace local blob URL with the persisted server URL
        URL.revokeObjectURL(localUrl);
        setAvatarPreview(null);
        setAvatarMsg({ type: 'success', text: 'Profile photo updated!' });
      } else {
        const err = await res.json().catch(() => ({}));
        setAvatarMsg({ type: 'error', text: err.detail || 'Upload failed. Try again.' });
      }
    } catch (err) {
      console.error('Avatar upload failed', err);
      setAvatarMsg({ type: 'error', text: 'Could not reach server. Photo preview shown locally.' });
    }
    setAvatarUploadLoading(false);
    // Auto-clear the message after 4 s
    setTimeout(() => setAvatarMsg(null), 4000);
  };

  return (
    <div className="dashboard" style={{ paddingBottom: '20px' }}>
      


      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
        {/* Left Column: Profile Overview */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '20px', padding: '32px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            
            <label style={{ cursor: 'pointer', position: 'relative', display: 'block', marginBottom: avatarMsg ? '12px' : '24px' }}>
              <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleAvatarUpload} disabled={avatarUploadLoading} />
              <div style={{ width: '120px', height: '120px', borderRadius: '50%', background: 'rgba(123, 157, 174, 0.10)', border: '2px solid rgba(123, 157, 174, 0.30)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '48px', color: '#C6A75E', boxShadow: '0 0 40px rgba(123,157,174,0.18)', overflow: 'hidden', position: 'relative' }}>
                {(avatarPreview || user?.avatar_url) ? (
                  <img src={avatarPreview || user.avatar_url} alt="Profile" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <i className="ti ti-user-circle"></i>
                )}
                {/* Hover overlay */}
                <div style={{ position: 'absolute', inset: 0, background: 'rgba(232,220,200,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: avatarUploadLoading ? 1 : 0, transition: 'opacity 0.2s' }} className="avatar-overlay">
                  {avatarUploadLoading ? <i className="ti ti-loader" style={{ animation: 'spin 1s linear infinite', color: '#C6A75E', fontSize: '24px' }}></i> : <i className="ti ti-camera" style={{ color: 'var(--text-card)', fontSize: '24px' }}></i>}
                </div>
              </div>
            </label>
            {/* Upload feedback message */}
            {avatarMsg && (
              <div style={{
                marginBottom: '16px',
                padding: '8px 14px',
                borderRadius: '8px',
                fontSize: '12px',
                fontWeight: 500,
                background: avatarMsg.type === 'success' ? 'rgba(123,157,174,0.12)' : 'rgba(122,46,50,0.12)',
                border: `1px solid ${avatarMsg.type === 'success' ? 'rgba(123,157,174,0.30)' : 'rgba(122,46,50,0.30)'}`,
                color: avatarMsg.type === 'success' ? '#C6A75E' : '#A63A3F',
                display: 'flex', alignItems: 'center', gap: '6px'
              }}>
                <i className={`ti ${avatarMsg.type === 'success' ? 'ti-circle-check' : 'ti-alert-circle'}`}></i>
                {avatarMsg.text}
              </div>
            )}

            <div style={{ fontSize: '24px', fontWeight: 600, color: 'var(--text-card)', marginBottom: '8px' }}>{fullName}</div>
            <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>{user?.email || 'admin@vocalarmor.com'}</div>

            <div style={{ width: '100%', height: '1px', background: 'rgba(232,220,200,0.1)', margin: '32px 0' }}></div>

            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '16px' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Total Scans</span>
              <span style={{ color: 'var(--text-card)', fontWeight: 600 }}>{scanCount.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Detection Accuracy</span>
              <span style={{ color: '#C6A75E', fontWeight: 600 }}>99.8%</span>
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '20px', padding: '24px' }}>
             <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-card)', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px' }}>Recent Activity</div>
             
             <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
               <div style={{ display: 'flex', gap: '12px' }}>
                 <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#C6A75E', marginTop: '4px' }}></div>
                 <div>
                   <div style={{ fontSize: '12px', color: 'var(--text-card)', fontWeight: 500 }}>Batch Scan Completed</div>
                   <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>Today, 14:23 PM</div>
                 </div>
               </div>
               
               <div style={{ display: 'flex', gap: '12px' }}>
                 <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#A63A3F', marginTop: '4px' }}></div>
                 <div>
                   <div style={{ fontSize: '12px', color: 'var(--text-card)', fontWeight: 500 }}>Deepfake Detected (98%)</div>
                   <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>Yesterday, 09:15 AM</div>
                 </div>
               </div>

               <div style={{ display: 'flex', gap: '12px' }}>
                 <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--text-muted)', marginTop: '4px' }}></div>
                 <div>
                   <div style={{ fontSize: '12px', color: 'var(--text-card)', fontWeight: 500 }}>System Login</div>
                   <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>May 15th, 08:00 AM</div>
                 </div>
               </div>
             </div>
          </div>

        </div>

        {/* Right Column: Settings */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Personal Details */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="ti ti-id" style={{ color: '#C6A75E' }}></i> Personal Details
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px', display: 'block' }}>First Name</label>
                  <input type="text" name="firstName" value={editForm.firstName} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(232,220,200,0.03)', border: '1px solid rgba(232,220,200,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'var(--text-card)', outline: 'none' }} />
                </div>
                <div>
                  <label style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px', display: 'block' }}>Last Name</label>
                  <input type="text" name="lastName" value={editForm.lastName} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(232,220,200,0.03)', border: '1px solid rgba(232,220,200,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'var(--text-card)', outline: 'none' }} />
                </div>
              </div>

              <div>
                <label style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px', display: 'block' }}>Email Address</label>
                <input type="email" name="email" value={editForm.email} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(232,220,200,0.03)', border: '1px solid rgba(232,220,200,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'var(--text-card)', outline: 'none' }} />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px', display: 'block' }}>Phone Number</label>
                <input type="text" name="phone" value={editForm.phone} onChange={handleInputChange} style={{ width: '100%', background: 'rgba(232,220,200,0.03)', border: '1px solid rgba(232,220,200,0.1)', padding: '12px 16px', borderRadius: '8px', color: 'var(--text-card)', outline: 'none' }} />
              </div>
            </div>
          </div>

          {/* Engine Preferences */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="ti ti-settings" style={{ color: '#C6A75E' }}></i> Detection Preferences
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(232,220,200,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Strict False-Positive Filter</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Increases threshold on v3 model to prevent flagging human voices.</div>
                </div>
                <div onClick={() => setStrictFilter(!strictFilter)} style={{ width: '44px', height: '24px', background: strictFilter ? '#C6A75E' : 'rgba(232,220,200,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}>
                  <div style={{ width: '20px', height: '20px', background: strictFilter ? '#1E1310' : 'var(--text-muted)', borderRadius: '50%', position: 'absolute', top: '2px', left: strictFilter ? '22px' : '2px', transition: 'left 0.3s' }}></div>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(232,220,200,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Auto-Save Analysis History</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Keep a local record of all batch and live recordings in the History page.</div>
                </div>
                <div onClick={() => setAutoSave(!autoSave)} style={{ width: '44px', height: '24px', background: autoSave ? '#C6A75E' : 'rgba(232,220,200,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}>
                  <div style={{ width: '20px', height: '20px', background: autoSave ? '#1E1310' : 'var(--text-muted)', borderRadius: '50%', position: 'absolute', top: '2px', left: autoSave ? '22px' : '2px', transition: 'left 0.3s' }}></div>
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(232,220,200,0.02)', padding: '16px', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>Email Alerts on High-Risk Audio</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Get notified instantly if Batch Scanner detects 95%+ probability deepfakes.</div>
                </div>
                <div onClick={() => setEmailAlerts(!emailAlerts)} style={{ width: '44px', height: '24px', background: emailAlerts ? '#A63A3F' : 'rgba(232,220,200,0.1)', borderRadius: '12px', position: 'relative', cursor: 'pointer', transition: 'background 0.3s' }}>
                  <div style={{ width: '20px', height: '20px', background: emailAlerts ? 'white' : 'var(--text-muted)', borderRadius: '50%', position: 'absolute', top: '2px', left: emailAlerts ? '22px' : '2px', transition: 'left 0.3s' }}></div>
                </div>
              </div>
            </div>
          </div>

          <button onClick={handleSave} style={{ background: isSaving ? '#211816' : 'linear-gradient(90deg, #E8DCC8, #C6A75E)', color: isSaving ? '#C6A75E' : '#151412', border: isSaving ? '1px solid rgba(198,167,94,0.3)' : '1px solid rgba(198,167,94,0.5)', borderRadius: '50px', padding: '16px 24px', cursor: 'pointer', fontWeight: 700, letterSpacing: '1px', boxShadow: isSaving ? 'none' : '0 8px 32px rgba(0,0,0,0.3)', width: '100%', marginTop: 'auto', transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }} disabled={isSaving}>
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
