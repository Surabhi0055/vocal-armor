import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

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

const PALETTE = [
  [255, 92,  43],
  [  0, 212, 200],
  [255, 138,  0],
  [  0, 180, 255],
  [232,  82,  30],
];

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

  const canvasRef = useRef(null);
  const cardRef   = useRef(null);
  const mouseRef  = useRef({ x: -9999, y: -9999 });
  const tRef      = useRef(0);
  const rafRef    = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', e => { mouseRef.current = { x: e.clientX, y: e.clientY }; });

    const drawCluster = (cx, cy, innerR, gap, count, t, colOffset) => {
      const mdx    = mouseRef.current.x - cx;
      const mdy    = mouseRef.current.y - cy;
      const mDist  = Math.sqrt(mdx * mdx + mdy * mdy);
      const mBoost = Math.max(0, 1 - mDist / 220) * 45;

      for (let i = 0; i < count; i++) {
        const baseR  = innerR + i * gap;
        const expand = mBoost * (1 - i / count) * Math.sin(t * 0.04 - i * 0.3 + 1);
        const R      = Math.max(1, baseR + expand);
        const [r,g,b]= PALETTE[(i + colOffset) % PALETTE.length];
        const alpha  = Math.max(0.05, 0.72 - i * (0.62 / count));
        const lw     = Math.max(0.4, 1.9 - i * (1.3 / count));

        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth   = lw;
        ctx.shadowColor = `rgba(${r},${g},${b},${alpha * 0.55})`;
        ctx.shadowBlur  = lw * 9;
        ctx.stroke();
        ctx.shadowBlur  = 0;
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const t = tRef.current;

      let tlX = canvas.width * 0.22, tlY = canvas.height * 0.18;
      let brX = canvas.width * 0.78, brY = canvas.height * 0.82;

      if (cardRef.current) {
        const rect = cardRef.current.getBoundingClientRect();
        tlX = rect.left; tlY = rect.top;
        brX = rect.right; brY = rect.bottom;
      }

      drawCluster(tlX, tlY, 8,  16, 16, t, 0);
      drawCluster(brX, brY, 12, 20, 20, t, 1);

      tRef.current++;
      rafRef.current = requestAnimationFrame(draw);
    };
    draw();

    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(rafRef.current); };
  }, []);

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
      await fetch('http://localhost:8000/auth/forgot-password', {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({email}),
      });
      setForgotSent(true);
    } catch { setFormError('Something went wrong.'); }
  };

  const switchMode = (m) => { setMode(m); setFormError(''); setForgotSent(false); clearError(); };

  const inp = {
    width:'100%', boxSizing:'border-box',
    background:'rgba(255,255,255,0.03)',
    border:'1px solid rgba(255,255,255,0.15)',
    borderRadius:'12px', padding:'12px 14px',
    fontSize:'14px', color:'#dfe8e6', fontFamily:'"Syne", sans-serif',
    outline:'none', transition:'all 0.3s ease',
    backdropFilter:'blur(10px)', WebkitBackdropFilter:'blur(10px)', 
  };
  
  const fi = ev => { 
    ev.target.style.background='rgba(255,255,255,0.08)'; 
    ev.target.style.borderColor='rgba(0,212,200,0.60)'; 
    ev.target.style.boxShadow='0 0 15px rgba(0,212,200,0.15)'; 
  };
  const fo = ev => { 
    ev.target.style.background='rgba(255,255,255,0.03)'; 
    ev.target.style.borderColor='rgba(255,255,255,0.15)'; 
    ev.target.style.boxShadow='none'; 
  };
  const lbl = { display:'block', fontSize:'11px', fontWeight:600, color:'rgba(0,212,200,0.80)', letterSpacing:'0.09em', textTransform:'uppercase', marginBottom:'6px' };

  const glassBtn = {
    width:'100%', padding:'13px', cursor:'pointer', fontFamily:'"Syne", sans-serif',
    fontSize:'14px', fontWeight:700, letterSpacing:'0.06em', textTransform:'uppercase',
    color:'#dfe8e6', borderRadius:'12px',
    background:'rgba(255,255,255,0.05)',
    border:'1px solid rgba(255,255,255,0.25)',
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
        -webkit-box-shadow: 0 0 0 30px rgba(5, 14, 16, 0.6) inset !important;
        -webkit-text-fill-color: #dfe8e6 !important;
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
      color: #ffffff;
      border-radius: 12px;
      
      background: linear-gradient(135deg, rgba(255,92,43,0.35) 0%, rgba(0,212,200,0.35) 100%);
      border: 1px solid rgba(255,255,255,0.3);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      
      opacity: ${isLoading ? 0.65 : 1};
      transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .bounce-btn:hover:not(:disabled) {
      background: linear-gradient(135deg, rgba(255,92,43,0.5) 0%, rgba(0,212,200,0.5) 100%);
      border-color: rgba(255,255,255,0.5);
      box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 0 20px rgba(255,255,255,0.1);
      transform: scale(1.04) translateY(-2px);
    }
    
    .bounce-btn:active:not(:disabled) {
      transform: scale(0.96) translateY(2px);
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
      <div style={{ width:'100vw', height:'100vh', background:'#050e10',
        display:'flex', alignItems:'center', justifyContent:'center',
        fontFamily:'"Syne", sans-serif', overflow:'hidden', position:'relative' }}>

        <canvas ref={canvasRef} style={{ position:'absolute', inset:0, width:'100%', height:'100%', pointerEvents:'none' }}/>

        <div style={{ position:'absolute', left:'4%', top:'4%', width:'320px', height:'320px', borderRadius:'50%',
          background:'radial-gradient(circle, rgba(255,92,43,0.22) 0%, transparent 65%)',
          filter:'blur(40px)', pointerEvents:'none', zIndex:1 }}/>
        <div style={{ position:'absolute', right:'4%', bottom:'4%', width:'420px', height:'420px', borderRadius:'50%',
          background:'radial-gradient(circle, rgba(0,180,255,0.18) 0%, transparent 65%)',
          filter:'blur(50px)', pointerEvents:'none', zIndex:1 }}/>

        <div ref={cardRef} style={{
          position:'relative', zIndex:10,
          width:'100%', maxWidth:'460px', margin:'16px',
          background:'rgba(255,255,255,0)',       // 100% transparent base
          backdropFilter:'blur(6px)', WebkitBackdropFilter:'blur(6px)', // Very light blur so waves are clear
          border:'1px solid rgba(255,255,255,0.15)', 
          borderRadius:'24px', padding:'44px 40px 40px',
          boxShadow:'0 15px 35px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.1)',
        }}>

          <div style={{ textAlign:'center', marginBottom:'32px' }}>
            <h1 style={{ fontFamily:'"Bebas Neue", sans-serif', fontSize:'44px',
              letterSpacing:'0.05em', color:'#dfe8e6', margin:0,
              textShadow:'0 0 30px rgba(0,212,200,0.30)' }}>
              {mode==='login'  && 'Welcome Back'}
              {mode==='signup' && 'Create Account'}
              {mode==='forgot' && 'Reset Password'}
            </h1>
          </div>

          {(formError||error) && (
            <div style={{ background:'rgba(255,92,43,0.08)', border:'1px solid rgba(255,92,43,0.25)',
              borderRadius:'10px', padding:'9px 13px', color:'rgba(255,138,80,0.90)',
              fontSize:'13px', marginBottom:'16px', backdropFilter:'blur(5px)' }}>
              {formError||error}
            </div>
          )}
          {forgotSent && (
            <div style={{ background:'rgba(0,212,200,0.07)', border:'1px solid rgba(0,212,200,0.22)',
              borderRadius:'10px', padding:'9px 13px', color:'#00d4c8', fontSize:'13px', marginBottom:'16px', backdropFilter:'blur(5px)' }}>
              ✓ Reset link sent! Check your inbox.
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
                      background:'none', border:'none', color:'#7ea8a4', cursor:'pointer', display:'flex' }}>
                    {showPass ? <EyeOff/> : <EyeOpen/>}
                  </button>
                </div>
              </div>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                <label style={{ display:'flex', alignItems:'center', gap:'7px', cursor:'pointer', fontSize:'12px', color:'#7ea8a4' }}>
                  <input type="checkbox" checked={rememberMe} onChange={e=>setRememberMe(e.target.checked)} style={{ accentColor:'#00d4c8' }}/>
                  Remember me
                </label>
                <button type="button" onClick={()=>switchMode('forgot')}
                  style={{ background:'none', border:'none', color:'rgba(0,212,200,0.80)', cursor:'pointer', fontSize:'12px', fontFamily:'inherit' }}>
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
              <button type="button" style={googleBtn}
                onClick={()=>window.location.href='http://localhost:8000/auth/google'}
                onMouseOver={e => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                onMouseOut={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
                <GoogleSVG/> Continue with Google
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
                      background:'none', border:'none', color:'#7ea8a4', cursor:'pointer', display:'flex' }}>
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
              <button type="button" style={googleBtn}
                onClick={()=>window.location.href='http://localhost:8000/auth/google'}
                onMouseOver={e => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                onMouseOut={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
                <GoogleSVG/> Sign up with Google
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
                style={{ background:'none', border:'none', color:'rgba(0,212,200,0.60)', cursor:'pointer',
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
                style={{ background:'none', border:'none', color:'rgba(0,212,200,0.9)', fontWeight:600,
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