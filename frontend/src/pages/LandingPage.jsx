import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import WaveformBackground from '../components/WaveformBackground';
import VAIcon from '../components/VAIcon';

const G = `
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700;800&display=swap');
:root{--bg:#1F2A44;--marie:#A63A3F;--nordic:#C6A75E;--cream:#E8DCC8;--muted:rgba(232,220,200,0.6);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg);font-family:'Space Grotesk',sans-serif;color:var(--cream);overflow-x:hidden;}
.fu{opacity:0;transform:translateY(24px);transition:opacity .6s cubic-bezier(.4,0,.2,1),transform .6s cubic-bezier(.4,0,.2,1);}
.fu.v{opacity:1;transform:none;}
.fl{opacity:0;transform:translateX(-20px);transition:opacity .65s cubic-bezier(.4,0,.2,1),transform .65s cubic-bezier(.4,0,.2,1);}
.fl.v{opacity:1;transform:none;}
.fr{opacity:0;transform:translateX(20px);transition:opacity .65s cubic-bezier(.4,0,.2,1),transform .65s cubic-bezier(.4,0,.2,1);}
.fr.v{opacity:1;transform:none;}
.gl{background:rgba(26,18,16,.70);backdrop-filter:blur(16px) saturate(1.3);border:1px solid rgba(255,255,255,.07);border-radius:20px;transition:border-color .3s;}
.gl:hover{border-color:rgba(123,157,174,.28);}
.nl{font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;color:var(--muted);cursor:pointer;letter-spacing:.04em;transition:color .2s;}
.nl:hover{color:#fff;}
.bp{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;padding:12px 28px;border-radius:50px;border:1px solid rgba(232,220,200,0.5);cursor:pointer;background:linear-gradient(90deg, #C6A75E, #1F2A44);color:#E8DCC8;letter-spacing:.04em;transition:opacity .2s,transform .2s;}
.bp:hover{opacity:.85;transform:scale(1.03);}
.bg{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;padding:12px 28px;border-radius:50px;border:1px solid #C6A75E;cursor:pointer;background:transparent;color:#E8DCC8;letter-spacing:.04em;transition:background .2s;}
.bg:hover{background:rgba(198,167,94,.10);}

/* Bento Box Styles */
.bento-dark {
  background: transparent; backdrop-filter: blur(14px); border: 1px solid rgba(232,220,200,0.3); border-radius: 20px;
  padding: 40px; color: #E8DCC8; transition: transform 0.3s, border-color 0.3s;
}
.bento-dark:hover { border-color: rgba(198,167,94,0.3); transform: translateY(-4px); }

.bento-light {
  background: transparent; border: 1px solid rgba(232,220,200,0.3); border-radius: 20px;
  padding: 40px; color: #E8DCC8; transition: transform 0.3s, border-color 0.3s;
}
.bento-light:hover { border-color: rgba(166,58,63,0.3); transform: translateY(-4px); }

.num-circle-dark {
  width: 36px; height: 36px; border-radius: 50%; background: rgba(198,167,94,0.15); color: #E8DCC8;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 13px; margin-bottom: 40px;
  border: 1px solid rgba(198,167,94,0.2);
}
.num-circle-light {
  width: 36px; height: 36px; border-radius: 50%; background: transparent; color: #E8DCC8;
  border: 1px solid rgba(232,220,200,0.3);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 13px; margin-bottom: 40px;
}
`;

const scrollTo = (id) => {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

export default function LandingPage() {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  
  // Background/Cursor refs
  const dot = useRef(null);
  const ring = useRef(null);
  const mouse = useRef({ x: -999, y: -999 });
  const rPos = useRef({ x: -999, y: -999 });
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const prev = { ov: document.body.style.overflow, h: document.body.style.height };
    document.body.style.overflow = 'auto'; document.body.style.height = 'auto';
    const root = document.getElementById('root');
    const pr = root ? { d: root.style.display, h: root.style.height } : null;
    if (root) { root.style.display = 'block'; root.style.height = 'auto'; }
    return () => {
      document.body.style.overflow = prev.ov; document.body.style.height = prev.h;
      if (root && pr) { root.style.display = pr.d; root.style.height = pr.h; }
    };
  }, []);

  useEffect(() => {
    const mv = e => { mouse.current = { x: e.clientX, y: e.clientY }; };
    window.addEventListener('mousemove', mv);
    let raf;
    const tick = () => {
      const { x, y } = mouse.current;
      if (dot.current) { dot.current.style.left = x + 'px'; dot.current.style.top = y + 'px'; }
      rPos.current.x += (x - rPos.current.x) * 0.12; rPos.current.y += (y - rPos.current.y) * 0.12;
      if (ring.current) { ring.current.style.left = rPos.current.x + 'px'; ring.current.style.top = rPos.current.y + 'px'; }
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => { window.removeEventListener('mousemove', mv); cancelAnimationFrame(raf); };
  }, []);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', fn); return () => window.removeEventListener('scroll', fn);
  }, []);

  useEffect(() => {
    const io = new IntersectionObserver(entries => {
      entries.forEach((e, i) => {
        if (e.isIntersecting) { setTimeout(() => e.target.classList.add('v'), i * 80); io.unobserve(e.target); }
      });
    }, { threshold: 0.1 });
    document.querySelectorAll('.fu,.fl,.fr').forEach(el => io.observe(el));
    return () => io.disconnect();
  }, []);


  const S = {
    syne:  { fontFamily: "'Syne', sans-serif" },          // headings & feature titles
    epi:   { fontFamily: "'Syne', sans-serif" },           // descriptive / body text
    mono:  { fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '0.05em' }, // section labels & tags
    bebas: { fontFamily: "'Bebas Neue', sans-serif" },    // hero title
  };

  const navLinks = [
    { label: 'How It Works', id: 'how-it-works' },
    { label: 'Features', id: 'features' }
  ];

  const footerLinks = [
    { label: 'How It Works', id: 'how-it-works' },
    { label: 'Features', id: 'features' },
    { label: 'GitHub', url: 'https://github.com' },
    { label: 'Privacy', id: null }
  ];

  return (
    <>
      <style>{G}</style>
      <WaveformBackground />

      <div style={{ width: '100vw', minHeight: '100vh', background: 'transparent', position: 'relative', zIndex: 10, overflowX: 'hidden' }}>

        {/* ── NAV ── */}
        <nav style={{
          position: 'sticky', top: 0, zIndex: 100, display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', padding: '0 5%', height: '64px',
          background: 'rgba(23,32,54,0.92)', backdropFilter: 'blur(24px)',
          borderBottom: `1px solid ${scrolled ? 'rgba(198,167,94,0.50)' : 'rgba(198,167,94,0.1)'}`,
          transition: 'border-color 0.4s',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }} onClick={() => scrollTo('hero')}>
          <VAIcon size={32} style={{ borderRadius: '6px' }} />
            <span style={{ fontFamily: '"Space Grotesk", sans-serif', fontSize: '17px', letterSpacing: '1px', color: '#E8DCC8' }}><span style={{ fontWeight: 800 }}>VOCAL</span><span style={{ fontWeight: 300, color: '#C6A75E' }}>ARMOR</span></span>
          </div>
          <div style={{ display: 'flex', gap: '32px' }}>
            {navLinks.map(l => (
              <span key={l.label} className="nl"
                onClick={() => l.url ? window.open(l.url, '_blank') : l.id ? scrollTo(l.id) : null}>
                {l.label}
              </span>
            ))}
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            {user ? (
              <button className="bp" onClick={() => navigate('/')}>Dashboard</button>
            ) : (
              <>
                <button className="bg" onClick={() => navigate('/login')}>Log In</button>
                <button className="bp" onClick={() => navigate('/login')}>Sign Up</button>
              </>
            )}
          </div>
        </nav>

        <section id="hero" style={{ minHeight: '90vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '80px 6% 60px' }}>
          <h1 className="hero-title">
            DETECT AI <span style={{ color: '#A63A3F' }}>VOICES</span>
            <br />
            BEFORE THEY{" "}
            <span style={{ color: '#C6A75E', textShadow: "0 0 60px rgba(198,167,94,0.5)" }}>
              DECEIVE
            </span>
          </h1>
          <p className="hero-subtitle" style={{ fontSize: '12px', marginBottom: '32px' }}>
            Real-time deepfake voice detection via spectrogram analysis.
            <br />
            Upload any audio — get a verdict in under 2 seconds.
          </p>
          <button className="bp" style={{ padding: '13px 32px', fontSize: '15px' }} onClick={() => navigate(user ? '/' : '/login')}>Get Started Free →</button>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section id="how-it-works" className="fu" style={{ padding: '60px 6%', maxWidth: '1400px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px', marginBottom: '70px' }}>
            <div style={{ height: '1px', width: '80px', background: 'rgba(255,255,255,0.1)' }} />
            <h2 style={{ ...S.mono, fontSize: '12px', fontWeight: 500, color: 'var(--text-muted)', letterSpacing: '0.2em' }}>HOW IT WORKS</h2>
            <div style={{ height: '1px', width: '80px', background: 'rgba(255,255,255,0.1)' }} />
          </div>

          <div className="hiw-grid" style={{ gap: '30px', margin: 0 }}>
            <div className="hiw-card">
              <div className="hiw-step-number">1</div>
              <div className="hiw-title">Audio Ingestion</div>
              <div className="hiw-desc">
                Loads audio, forces mono channel, and resamples to 22.05 kHz for
                uniform analysis input across all supported formats.
              </div>
            </div>
            <div className="hiw-card">
              <div className="hiw-step-number">2</div>
              <div className="hiw-title">Mel Spectrogram</div>
              <div className="hiw-desc">
                Converts the loudest 2-second window into a 128×128 mel spectrogram
                image for deep visual pattern recognition.
              </div>
            </div>
            <div className="hiw-card">
              <div className="hiw-step-number">3</div>
              <div className="hiw-title">CNN Inference</div>
              <div className="hiw-desc">
                VocalArmor's proprietary CNN model classifies the spectrogram as a
                real human voice or an AI-generated deepfake.
              </div>
            </div>
          </div>
        </section>

        {/* ── FEATURES BENTO BOX (Image 2 Bottom Style) ── */}
        <section id="features" className="fu" style={{ padding: '40px 6% 100px', maxWidth: '1400px', margin: '0 auto' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Top Row */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px' }}>
              
              <div className="bento-dark" style={{ flex: '4 1 300px', minHeight: '320px' }}>
                <div className="num-circle-dark">01</div>
                <h3 style={{ ...S.syne, fontSize: '18px', fontWeight: 700, letterSpacing: '0.02em', marginBottom: '16px', textTransform: 'uppercase' }}>Live Monitor Engine</h3>
                <p style={{ ...S.epi, fontSize: '14px', color: '#a0a0a0', lineHeight: 1.7 }}>
                  Our highly optimized live monitor analyzes inbound microphone audio streams in real-time, delivering actionable classifications within two seconds to catch deepfakes instantly.
                </p>
              </div>

              <div className="bento-dark" style={{ flex: '6 1 450px', minHeight: '320px' }}>
                <div className="num-circle-dark">02</div>
                <h3 style={{ ...S.syne, fontSize: '18px', fontWeight: 700, letterSpacing: '0.02em', marginBottom: '16px', textTransform: 'uppercase' }}>History & Session Analytics</h3>
                <p style={{ ...S.epi, fontSize: '14px', color: '#a0a0a0', lineHeight: 1.7 }}>
                  Track your entire detection footprint. Every scan is logged with detailed confidence metrics, spectrogram visuals, and timestamps, allowing you to audit your history and analyze past sessions.
                </p>
              </div>

            </div>

            {/* Bottom Row */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px' }}>
              
              <div className="bento-dark" style={{ flex: '5 1 350px', minHeight: '320px', background: 'linear-gradient(135deg, rgba(198,167,94,0.9), rgba(31,42,68,0.9))', border: 'none', color: '#E8DCC8', boxShadow: '0 15px 40px rgba(0,0,0,0.4)' }}>
                <div className="num-circle-dark" style={{ color: '#E8DCC8' }}>03</div>
                <h3 style={{ ...S.syne, fontSize: '18px', fontWeight: 700, letterSpacing: '0.02em', marginBottom: '16px', textTransform: 'uppercase', color: '#E8DCC8' }}>Batch Processing Pipeline</h3>
                <p style={{ ...S.epi, fontSize: '14px', color: '#E8DCC8', lineHeight: 1.7, fontWeight: 500 }}>
                  From verifying massive datasets to sweeping historical archives, our robust batch upload architecture breathes efficiency into your moderation processes, scaling alongside your enterprise needs.
                </p>
              </div>

              <div className="bento-dark" style={{ flex: '5 1 350px', minHeight: '320px', position: 'relative', overflow: 'hidden' }}>
                <div className="num-circle-dark">04</div>
                <h3 style={{ ...S.syne, fontSize: '18px', fontWeight: 700, letterSpacing: '0.02em', marginBottom: '16px', position: 'relative', zIndex: 2, textTransform: 'uppercase' }}>Audio Detector</h3>
                <p style={{ fontFamily: '"Space Grotesk", sans-serif', fontSize: '12px', color: '#a0a0a0', lineHeight: 1.7, position: 'relative', zIndex: 2 }}>
                  Real-time deepfake voice detection via spectrogram analysis. Upload any audio — get a verdict in under 2 seconds.
                </p>
                {/* Decorative blob similar to the image */}
                <div style={{ position: 'absolute', right: '-40px', bottom: '-40px', width: '200px', height: '200px', background: 'radial-gradient(circle, rgba(198,167,94,0.22) 0%, transparent 70%)', filter: 'blur(30px)', pointerEvents: 'none', zIndex: 1 }} />
              </div>

            </div>
          </div>
        </section>
        {/* ── CTA ── */}
        <section className="fu" style={{ padding: '0 6% 100px', display: 'flex', justifyContent: 'center' }}>
          <button className="bp" style={{ padding: '16px 48px', fontSize: '18px', boxShadow: '0 10px 30px rgba(29,207,207,0.2)' }} onClick={() => navigate(user ? '/' : '/login')}>
            Get Started Free →
          </button>
        </section>

        {/* ── FOOTER ── */}
        <footer style={{ borderTop: '1px solid rgba(255,255,255,0.06)', padding: '30px 6%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '9px', cursor: 'pointer' }} onClick={() => scrollTo('hero')}>
            <VAIcon size={22} style={{ borderRadius: '4px' }} />
            <span style={{ fontFamily: '"Space Grotesk", sans-serif', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '1px' }}><span style={{ fontWeight: 800 }}>VOCAL</span><span style={{ fontWeight: 300, color: '#C6A75E' }}>ARMOR</span></span><span style={{ fontFamily: '"DM Mono", monospace', fontSize: '12px', color: 'var(--text-muted)' }}> · MIT License</span>
          </div>
          <div style={{ display: 'flex', gap: '26px' }}>
            {footerLinks.map(l => (
              <span key={l.label}
                style={{ ...S.epi, fontSize: '13px', color: 'var(--text-muted)', cursor: 'pointer', transition: 'color .2s' }}
                onClick={() => l.url ? window.open(l.url, '_blank') : l.id ? scrollTo(l.id) : null}
                onMouseOver={e => e.currentTarget.style.color = '#fff'}
                onMouseOut={e => e.currentTarget.style.color = '#7a8f94'}>
                {l.label}
              </span>
            ))}
          </div>
          <span style={{ ...S.mono, fontSize: '11px', color: 'rgba(123,157,174,0.55)', letterSpacing: '0.05em' }}>Built with TensorFlow · FastAPI</span>
        </footer>

      </div>
    </>
  );
}
