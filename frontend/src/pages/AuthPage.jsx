import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
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



export default function AuthPage() {
  const navigate = useNavigate();
  const { login, register, isLoading, error, clearError } = useAuthStore();

  const [mode,            setMode]            = useState('login');
  const [email,           setEmail]           = useState('');
  const [username,        setUsername]        = useState('');
  const [fullName,        setFullName]        = useState('');
  const [password,        setPassword]        = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [rememberMe,      setRememberMe]      = useState(false);
  const [showPass,        setShowPass]        = useState(false);
  const [formError,       setFormError]       = useState('');
  const [forgotSent,      setForgotSent]      = useState(false);
  const [oauthLoading,    setOauthLoading]    = useState(false);

  const cardRef = useRef(null);



  const handleLogin = async (e) => {
    e.preventDefault(); setFormError(''); clearError();
    if (!email || !password) return setFormError('Please fill all fields');
    const res = await login(email, password, rememberMe);
    if (res.success) navigate('/'); else setFormError(res.error);
  };

  const handleRegister = async (e) => {
    e.preventDefault(); setFormError(''); clearError();
    if (!email||!username||!password||!confirmPassword) return setFormError('Please fill all fields');
    if (password !== confirmPassword) return setFormError('Passwords do not match');
    if (password.length < 8) return setFormError('Minimum 8 characters');
    const res = await register(email, username, password, fullName);
    if (res.success) navigate('/'); else setFormError(res.error);
  };

  const handleForgot = async (e) => {
    e.preventDefault(); setFormError('');
    if (!email) return setFormError('Please enter your email');
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        return setFormError(data?.detail || 'Something went wrong.');
      }
      setForgotSent(true);
    } catch {
      setFormError('Network error. Please check your connection.');
    }
  };

  const handleGoogleLogin = () => {
    if (oauthLoading) return;
    setOauthLoading(true);
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    window.location.href = `${apiUrl}/auth/google`;
  };

  const switchMode = (m) => { setMode(m); setFormError(''); setForgotSent(false); clearError(); };

  const inp = {
    width:'100%', boxSizing:'border-box',
    background:'rgba(0,0,0,0.25)',
    border:'1px solid rgba(232,220,200,0.3)',
    borderRadius:'12px', padding:'12px 14px',
    fontSize:'14px', color:'#E8DCC8', fontFamily:'"Syne", sans-serif',
    outline:'none', transition:'all 0.3s ease',
    backdropFilter:'blur(10px)', WebkitBackdropFilter:'blur(10px)', 
  };
  
  const fi = ev => { 
    ev.target.style.background='rgba(0,0,0,0.4)'; 
    ev.target.style.borderColor='#C6A75E'; 
    ev.target.style.boxShadow='0 0 15px rgba(198,167,94,0.15)'; 
  };
  const fo = ev => { 
    ev.target.style.background='rgba(0,0,0,0.25)'; 
    ev.target.style.borderColor='rgba(232,220,200,0.3)'; 
    ev.target.style.boxShadow='none'; 
  };
  const lbl = { display:'block', fontSize:'11px', fontWeight:600, color:'#E8DCC8', letterSpacing:'0.09em', textTransform:'uppercase', marginBottom:'6px' };

  const glassBtn = {
    width:'100%', padding:'13px', cursor:'pointer', fontFamily:'"Syne", sans-serif',
    fontSize:'14px', fontWeight:700, letterSpacing:'0.06em', textTransform:'uppercase',
    color:'#E8DCC8', borderRadius:'50px',
    background:'transparent',
    border:'1px solid #C6A75E',
    backdropFilter:'blur(12px)', WebkitBackdropFilter:'blur(12px)',
    boxShadow:'0 8px 32px rgba(0,0,0,0.2)',
    opacity: isLoading ? 0.65 : 1,
    transition:'all 0.3s ease',
  };

  const googleBtn = {
    ...glassBtn,
    fontWeight:600, textTransform:'none', letterSpacing:'normal',
    display:'flex', alignItems:'center', justifyContent:'center', gap:'10px',
  };

  const stylesOverride = `
    input:-webkit-autofill,
    input:-webkit-autofill:hover, 
    input:-webkit-autofill:focus, 
    input:-webkit-autofill:active{
        -webkit-box-shadow: 0 0 0 30px #151412 inset !important;
        -webkit-text-fill-color: #E8DCC8 !important;
        transition: background-color 5000s ease-in-out 0s;
    }

    .bounce-btn {
      width: 100%;
      padding: 13px;
      cursor: pointer;
      font-family: "Syne", sans-serif;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #1E1D1B;
      border-radius: 50px;
      
      background: linear-gradient(90deg, #E8DCC8, #C6A75E);
      border: 1px solid rgba(198,167,94,0.5);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      
      opacity: ${isLoading ? 0.65 : 1};
      transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .bounce-btn:hover:not(:disabled) {
      background: linear-gradient(90deg, #C6A75E, #A38A4B);
      border-color: rgba(232,220,200,0.4);
      box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 0 20px rgba(232,220,200,0.06);
      transform: scale(1.03) translateY(-2px);
    }
    
    .bounce-btn:active:not(:disabled) {
      transform: scale(0.97) translateY(2px);
    }

    .slide-in-right {
      animation: slideInRight 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    @keyframes slideInRight {
      0% {
        transform: translateX(100vw);
        opacity: 0;
      }
      100% {
        transform: translateX(0);
        opacity: 1;
      }
    }
  `;

  const GoogleSVG = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" style={{flexShrink:0}}>
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/>
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
    </svg>
  );

  return (
    <>
      <style>{stylesOverride}</style>
      <div style={{ width:'100vw', height:'100vh', background:'var(--bg-main)',
        display:'flex', alignItems:'center', justifyContent:'center',
        fontFamily:'"Syne", sans-serif', overflow:'hidden', position:'relative' }}>

        {/* Same waveform background as home & landing page */}
        <WaveformBackground />

        {/* Top gradient — Gold glow */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: '260px',
          background: 'linear-gradient(180deg, rgba(198,167,94,0.09) 0%, transparent 100%)',
          pointerEvents: 'none', zIndex: 1,
        }} />

        {/* Bottom gradient — Maroon glow */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0, height: '220px',
          background: 'linear-gradient(0deg, rgba(166,58,63,0.10) 0%, transparent 100%)',
          pointerEvents: 'none', zIndex: 1,
        }} />

        <div ref={cardRef} className="slide-in-right" style={{
          position:'relative', zIndex:10,
          width:'100%', maxWidth:'460px', margin:'16px',
          background:'rgba(198,167,94,0.05)',
          backdropFilter:'blur(18px)', WebkitBackdropFilter:'blur(18px)',
          border:'1px solid rgba(198,167,94,0.3)', 
          borderRadius:'24px', padding:'44px 40px 40px',
          boxShadow:'0 20px 50px rgba(0,0,0,0.4)',
        }}>

          <div style={{ textAlign:'center', marginBottom:'32px' }}>
            <h1 style={{ fontFamily:'"Bebas Neue", sans-serif', fontSize:'44px',
              letterSpacing:'0.05em', color:'#E8DCC8', margin:0,
              textShadow:'0 0 30px rgba(198,167,94,0.25)' }}>
              {mode==='login'  && 'Welcome Back'}
              {mode==='signup' && 'Create Account'}
              {mode==='forgot' && 'Reset Password'}
            </h1>
          </div>

          {(formError||error) && (
            <div style={{ background:'rgba(122,46,50,0.10)', border:'1px solid rgba(166,58,63,0.28)',
              borderRadius:'10px', padding:'9px 13px', color:'rgba(200,100,105,0.92)',
              fontSize:'13px', marginBottom:'16px', backdropFilter:'blur(5px)' }}>
              {formError||error}
            </div>
          )}
          {/* FIX: success banner scoped inside forgot mode only to avoid
              it showing up on the login form after returning from forgot */}
          {forgotSent && mode === 'forgot' && (
            <div style={{ background:'rgba(123,157,174,0.08)', border:'1px solid rgba(123,157,174,0.25)',
              borderRadius:'10px', padding:'9px 13px', color:'#7B9DAE', fontSize:'13px', marginBottom:'16px', backdropFilter:'blur(5px)' }}>
              ✓ If that email is registered, a reset link has been sent.
            </div>
          )}

          {/* ── LOGIN ── */}
          {mode==='login' && (
            <form onSubmit={handleLogin} style={{ display:'flex', flexDirection:'column', gap:'16px' }}>
              <div>
                <label style={lbl}>Email</label>
                <input style={inp} type="email" placeholder="you@example.com"
                  value={email} onChange={e=>setEmail(e.target.value)} onFocus={fi} onBlur={fo}/>
              </div>
              <div>
                <label style={lbl}>Password</label>
                <div style={{ position:'relative' }}>
                  <input style={{...inp,paddingRight:'44px'}}
                    type={showPass?'text':'password'} placeholder="••••••••"
                    value={password} onChange={e=>setPassword(e.target.value)} onFocus={fi} onBlur={fo}/>
                  <button type="button" onClick={()=>setShowPass(!showPass)}
                    style={{ position:'absolute', right:'14px', top:'50%', transform:'translateY(-50%)',
                      background:'none', border:'none', color:'var(--text-muted)', cursor:'pointer', display:'flex' }}>
                    {showPass ? <EyeOff/> : <EyeOpen/>}
                  </button>
                </div>
              </div>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                {/* FIX: use a toggle function — clicking the label sets e.target to the <label>
                    element (not the checkbox), making e.target.checked undefined. */}
                <label
                  style={{ display:'flex', alignItems:'center', gap:'7px', cursor:'pointer', fontSize:'12px', color:'var(--text-muted)', userSelect:'none' }}
                  onClick={(e) => { e.preventDefault(); setRememberMe(v => !v); }}
                >
                  <div style={{
                    width: '16px', height: '16px', borderRadius: '4px', flexShrink: 0,
                    border: `1.5px solid ${rememberMe ? '#7B9DAE' : 'rgba(255,255,255,0.3)'}`,
                    background: rememberMe ? 'rgba(123,157,174,0.2)' : 'transparent',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    transition: 'all 0.2s',
                  }}>
                    {rememberMe && (
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <path d="M1.5 5L4 7.5L8.5 2.5" stroke="#7B9DAE" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                  </div>
                  Remember me
                </label>
                <button type="button" onClick={()=>switchMode('forgot')}
                  style={{ background:'none', border:'none', color:'rgba(123,157,174,0.85)', cursor:'pointer', fontSize:'12px', fontFamily:'inherit' }}>
                  Forgot password?
                </button>
              </div>

              <button type="submit" disabled={isLoading} className="bounce-btn">
                {isLoading ? 'Signing in…' : 'Sign In →'}
              </button>

              <div style={{ display:'flex', alignItems:'center', gap:'10px', color:'rgba(255,255,255,0.30)', fontSize:'11px' }}>
                <span style={{ flex:1, height:'1px', background:'rgba(255,255,255,0.1)'}}/>
                or continue with
                <span style={{ flex:1, height:'1px', background:'rgba(255,255,255,0.1)'}}/>
              </div>
              <button type="button"
                style={{ ...googleBtn, opacity: oauthLoading ? 0.65 : 1 }}
                onClick={handleGoogleLogin}
                disabled={oauthLoading}
                onMouseOver={e => { if (!oauthLoading) e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
                onMouseOut={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
                <GoogleSVG/> {oauthLoading ? 'Redirecting…' : 'Continue with Google'}
              </button>
            </form>
          )}

          {/* ── SIGNUP ── */}
          {mode==='signup' && (
            <form onSubmit={handleRegister} style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'10px' }}>
                <div>
                  <label style={lbl}>Full Name</label>
                  <input style={inp} type="text" placeholder="Your name"
                    value={fullName} onChange={e=>setFullName(e.target.value)} onFocus={fi} onBlur={fo}/>
                </div>
                <div>
                  <label style={lbl}>Username</label>
                  <input style={inp} type="text" placeholder="username"
                    value={username} onChange={e=>setUsername(e.target.value)} onFocus={fi} onBlur={fo}/>
                </div>
              </div>
              <div>
                <label style={lbl}>Email</label>
                <input style={inp} type="email" placeholder="you@example.com"
                  value={email} onChange={e=>setEmail(e.target.value)} onFocus={fi} onBlur={fo}/>
              </div>
              <div>
                <label style={lbl}>Password</label>
                <div style={{ position:'relative' }}>
                  <input style={{...inp,paddingRight:'44px'}} type={showPass?'text':'password'} placeholder="Min 8 characters"
                    value={password} onChange={e=>setPassword(e.target.value)} onFocus={fi} onBlur={fo}/>
                  <button type="button" onClick={()=>setShowPass(!showPass)}
                    style={{ position:'absolute', right:'14px', top:'50%', transform:'translateY(-50%)',
                      background:'none', border:'none', color:'var(--text-muted)', cursor:'pointer', display:'flex' }}>
                    {showPass ? <EyeOff/> : <EyeOpen/>}
                  </button>
                </div>
              </div>
              <div>
                <label style={lbl}>Confirm Password</label>
                <input style={inp} type={showPass?'text':'password'} placeholder="Repeat password"
                  value={confirmPassword} onChange={e=>setConfirmPassword(e.target.value)} onFocus={fi} onBlur={fo}/>
              </div>
              
              <button type="submit" disabled={isLoading} className="bounce-btn">
                {isLoading ? 'Creating…' : 'Create Account →'}
              </button>
              
              <div style={{ display:'flex', alignItems:'center', gap:'10px', color:'rgba(255,255,255,0.30)', fontSize:'11px' }}>
                <span style={{ flex:1, height:'1px', background:'rgba(255,255,255,0.1)'}}/>or
                <span style={{ flex:1, height:'1px', background:'rgba(255,255,255,0.1)'}}/>
              </div>
              <button type="button"
                style={{ ...googleBtn, opacity: oauthLoading ? 0.65 : 1 }}
                onClick={handleGoogleLogin}
                disabled={oauthLoading}
                onMouseOver={e => { if (!oauthLoading) e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
                onMouseOut={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
                <GoogleSVG/> {oauthLoading ? 'Redirecting…' : 'Sign up with Google'}
              </button>
            </form>
          )}

          {/* ── FORGOT ── */}
          {mode==='forgot' && !forgotSent && (
            <form onSubmit={handleForgot} style={{ display:'flex', flexDirection:'column', gap:'16px' }}>
              <div>
                <label style={lbl}>Email address</label>
                <input style={inp} type="email" placeholder="you@example.com"
                  value={email} onChange={e=>setEmail(e.target.value)} onFocus={fi} onBlur={fo}/>
              </div>
              <button type="submit" className="bounce-btn">Send Reset Link →</button>
              
              <button type="button" onClick={()=>switchMode('login')}
                style={{ background:'none', border:'none', color:'rgba(123,157,174,0.65)', cursor:'pointer',
                  fontSize:'13px', fontFamily:'inherit', textAlign:'center', padding:'4px 0' }}>
                ← Back to sign in
              </button>
            </form>
          )}
          {forgotSent && mode==='forgot' && (
            <button onClick={()=>switchMode('login')} className="bounce-btn">
              ← Back to sign in
            </button>
          )}

          {mode !== 'forgot' && (
            <p style={{ textAlign:'center', marginTop:'22px', fontSize:'13px', color:'rgba(255,255,255,0.4)' }}>
              {mode==='login' ? "Don't have an account?" : 'Already have an account?'}
              <button onClick={()=>switchMode(mode==='login'?'signup':'login')}
                style={{ background:'none', border:'none', color:'rgba(123,157,174,0.95)', fontWeight:600,
                  marginLeft:'5px', cursor:'pointer', fontFamily:'inherit', fontSize:'13px' }}>
                {mode==='login' ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          )}

        </div>
      </div>
    </>
  );
}