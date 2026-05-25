import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import WaveformBackground from '../components/WaveformBackground';

const EyeOpen = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
  </svg>
);
const EyeOff = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>
);

const StrengthBar = ({ password }) => {
  const score = (() => {
    let s = 0;
    if (password.length >= 8)  s++;
    if (/[A-Z]/.test(password)) s++;
    if (/[0-9]/.test(password)) s++;
    if (/[^A-Za-z0-9]/.test(password)) s++;
    return s;
  })();
  const labels = ['', 'Weak', 'Fair', 'Good', 'Strong'];
  const colors = ['', '#e84d1c', '#f5a623', '#1dcfcf', '#00e676'];
  if (!password) return null;
  return (
    <div style={{ marginTop: '8px' }}>
      <div style={{ display: 'flex', gap: '4px', marginBottom: '4px' }}>
        {[1,2,3,4].map(i => (
          <div key={i} style={{
            flex: 1, height: '3px', borderRadius: '2px',
            background: i <= score ? colors[score] : 'rgba(255,255,255,0.1)',
            transition: 'background 0.3s',
          }} />
        ))}
      </div>
      <span style={{ fontSize: '11px', color: colors[score] }}>{labels[score]}</span>
    </div>
  );
};

export default function ResetPasswordPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');

  const [password,        setPassword]        = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPass,        setShowPass]        = useState(false);
  const [isLoading,       setIsLoading]       = useState(false);
  const [error,           setError]           = useState('');
  const [success,         setSuccess]         = useState(false);

  useEffect(() => {
    if (!token) {
      setError('Invalid reset link. Please request a new one.');
    }
  }, [token]);

  const inp = {
    width: '100%', boxSizing: 'border-box',
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '12px', padding: '12px 44px 12px 14px',
    fontSize: '14px', color: '#dfe8e6', fontFamily: '"Syne", sans-serif',
    outline: 'none', transition: 'all 0.3s ease',
  };
  const fi = ev => {
    ev.target.style.background = 'rgba(255,255,255,0.08)';
    ev.target.style.borderColor = 'rgba(29,207,207,0.60)';
    ev.target.style.boxShadow = '0 0 15px rgba(29,207,207,0.15)';
  };
  const fo = ev => {
    ev.target.style.background = 'rgba(255,255,255,0.03)';
    ev.target.style.borderColor = 'rgba(255,255,255,0.15)';
    ev.target.style.boxShadow = 'none';
  };
  const lbl = {
    display: 'block', fontSize: '11px', fontWeight: 600,
    color: 'rgba(29,207,207,0.80)', letterSpacing: '0.09em',
    textTransform: 'uppercase', marginBottom: '6px',
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!token) return setError('Invalid reset link. Please request a new one.');
    if (password.length < 8) return setError('Password must be at least 8 characters');
    if (password !== confirmPassword) return setError('Passwords do not match');

    setIsLoading(true);
    try {
      const res = await fetch('http://localhost:8000/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, new_password: password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.detail || 'Something went wrong. Please try again.');
      } else {
        setSuccess(true);
        setTimeout(() => navigate('/login'), 3000);
      }
    } catch {
      setError('Network error. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  const styles = `
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus {
      -webkit-box-shadow: 0 0 0 30px rgba(5, 14, 16, 0.6) inset !important;
      -webkit-text-fill-color: #dfe8e6 !important;
    }
    .reset-btn {
      width: 100%; padding: 13px; cursor: pointer;
      font-family: "Syne", sans-serif; font-size: 14px;
      font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
      color: #ffffff; border-radius: 12px;
      background: linear-gradient(135deg, rgba(255,92,43,0.35) 0%, rgba(29,207,207,0.35) 100%);
      border: 1px solid rgba(255,255,255,0.3);
      backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .reset-btn:hover:not(:disabled) {
      background: linear-gradient(135deg, rgba(255,92,43,0.5) 0%, rgba(29,207,207,0.5) 100%);
      box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 0 20px rgba(255,255,255,0.1);
      transform: scale(1.02) translateY(-2px);
    }
    .reset-btn:active:not(:disabled) { transform: scale(0.97) translateY(1px); }
    .reset-btn:disabled { opacity: 0.6; cursor: not-allowed; }
  `;

  return (
    <>
      <style>{styles}</style>
      <div style={{
        width: '100vw', height: '100vh', background: '#050e10',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: '"Syne", sans-serif', overflow: 'hidden', position: 'relative',
      }}>
        <WaveformBackground />

        {/* Top cyan gradient */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: '260px',
          background: 'linear-gradient(180deg, rgba(29,207,207,0.10) 0%, transparent 100%)',
          pointerEvents: 'none', zIndex: 1,
        }} />
        {/* Bottom orange gradient */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0, height: '220px',
          background: 'linear-gradient(0deg, rgba(232,82,30,0.09) 0%, transparent 100%)',
          pointerEvents: 'none', zIndex: 1,
        }} />

        <div style={{
          position: 'relative', zIndex: 10,
          width: '100%', maxWidth: '440px', margin: '16px',
          background: 'rgba(255,255,255,0)',
          backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
          border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: '24px', padding: '44px 40px 40px',
          boxShadow: '0 15px 35px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.1)',
        }}>

          {/* Lock icon */}
          <div style={{ textAlign: 'center', marginBottom: '8px' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: '56px', height: '56px', borderRadius: '16px',
              background: 'rgba(29,207,207,0.10)',
              border: '1px solid rgba(29,207,207,0.25)',
              marginBottom: '16px',
            }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1dcfcf" strokeWidth="2" strokeLinecap="round">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
              </svg>
            </div>
          </div>

          <div style={{ textAlign: 'center', marginBottom: '28px' }}>
            <h1 style={{
              fontFamily: '"Bebas Neue", sans-serif', fontSize: '38px',
              letterSpacing: '0.05em', color: '#dfe8e6', margin: '0 0 6px',
              textShadow: '0 0 30px rgba(29,207,207,0.30)',
            }}>
              New Password
            </h1>
            <p style={{ fontSize: '13px', color: '#7ea8a4', margin: 0 }}>
              Choose a strong password for your account
            </p>
          </div>

          {/* ── Success state ── */}
          {success ? (
            <div style={{ textAlign: 'center' }}>
              <div style={{
                background: 'rgba(29,207,207,0.07)',
                border: '1px solid rgba(29,207,207,0.25)',
                borderRadius: '14px', padding: '28px 20px', marginBottom: '20px',
              }}>
                <div style={{ fontSize: '40px', marginBottom: '12px' }}>✓</div>
                <p style={{ color: '#1dcfcf', fontWeight: 600, margin: '0 0 6px' }}>
                  Password updated!
                </p>
                <p style={{ fontSize: '13px', color: '#7ea8a4', margin: 0 }}>
                  Redirecting you to sign in…
                </p>
              </div>
              <button
                onClick={() => navigate('/login')}
                className="reset-btn"
              >
                Go to Sign In →
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

              {error && (
                <div style={{
                  background: 'rgba(255,92,43,0.08)', border: '1px solid rgba(255,92,43,0.25)',
                  borderRadius: '10px', padding: '10px 14px',
                  color: 'rgba(255,138,80,0.90)', fontSize: '13px',
                }}>
                  {error}
                </div>
              )}

              {/* No token = dead link — show only error, no form */}
              {token && (
                <>
                  <div>
                    <label style={lbl}>New Password</label>
                    <div style={{ position: 'relative' }}>
                      <input
                        style={inp}
                        type={showPass ? 'text' : 'password'}
                        placeholder="Min 8 characters"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        onFocus={fi} onBlur={fo}
                        autoFocus
                      />
                      <button
                        type="button"
                        onClick={() => setShowPass(!showPass)}
                        style={{
                          position: 'absolute', right: '14px', top: '50%',
                          transform: 'translateY(-50%)',
                          background: 'none', border: 'none',
                          color: '#7ea8a4', cursor: 'pointer', display: 'flex',
                        }}
                      >
                        {showPass ? <EyeOff /> : <EyeOpen />}
                      </button>
                    </div>
                    <StrengthBar password={password} />
                  </div>

                  <div>
                    <label style={lbl}>Confirm Password</label>
                    <div style={{ position: 'relative' }}>
                      <input
                        style={{
                          ...inp,
                          borderColor: confirmPassword && confirmPassword !== password
                            ? 'rgba(255,92,43,0.5)' : undefined,
                        }}
                        type={showPass ? 'text' : 'password'}
                        placeholder="Repeat password"
                        value={confirmPassword}
                        onChange={e => setConfirmPassword(e.target.value)}
                        onFocus={fi} onBlur={fo}
                      />
                      {confirmPassword && confirmPassword === password && (
                        <span style={{
                          position: 'absolute', right: '14px', top: '50%',
                          transform: 'translateY(-50%)', color: '#1dcfcf', fontSize: '16px',
                        }}>✓</span>
                      )}
                    </div>
                  </div>

                  <button type="submit" disabled={isLoading} className="reset-btn">
                    {isLoading ? 'Updating password…' : 'Set New Password →'}
                  </button>
                </>
              )}

              <button
                type="button"
                onClick={() => navigate('/login')}
                style={{
                  background: 'none', border: 'none',
                  color: 'rgba(29,207,207,0.60)', cursor: 'pointer',
                  fontSize: '13px', fontFamily: 'inherit',
                  textAlign: 'center', padding: '4px 0',
                }}
              >
                ← Back to sign in
              </button>
            </form>
          )}
        </div>
      </div>
    </>
  );
}
