import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import FakeRateChart from "./FakeRateChart";
import ConfidenceHistogram from "./ConfidenceHistogram";
import HistoryTable from "./HistoryTable";
import { saveAnalysis, getHistory } from "../utils/storage";

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const reqIdRef = useRef(null);

  // Quick stats state
  const [quickStats, setQuickStats] = useState({
    maxFake: null,
    maxReal: null,
    streak: 0,
    todayCount: 0
  });

  const loadQuickStats = useCallback(() => {
    const history = getHistory();
    let maxF = null, maxR = null;
    let currentStreak = 0, maxStreak = 0;
    let prevWasFake = false;
    let todayC = 0;
    
    const today = new Date().toLocaleDateString();

    history.forEach(item => {
      // most confident fake/real
      if (item.prediction === 'FAKE') {
        if (!maxF || item.confidence > maxF.confidence) maxF = item;
      } else {
        if (!maxR || item.confidence > maxR.confidence) maxR = item;
      }
      
      // today's count
      if (item.date === today) todayC++;
    });

    // Calculate longest session streak of FAKE detections
    // Since history is newest first, let's reverse to process chronologically
    [...history].reverse().forEach(item => {
      if (item.prediction === 'FAKE') {
        if (prevWasFake) currentStreak++;
        else currentStreak = 1;
        prevWasFake = true;
        if (currentStreak > maxStreak) maxStreak = currentStreak;
      } else {
        prevWasFake = false;
        currentStreak = 0;
      }
    });

    setQuickStats({ maxFake: maxF, maxReal: maxR, streak: maxStreak, todayCount: todayC });
  }, []);

  useEffect(() => {
    loadQuickStats();
    window.addEventListener('va_history_updated', loadQuickStats);
    return () => window.removeEventListener('va_history_updated', loadQuickStats);
  }, [loadQuickStats]);

  // ── Spectrogram animator ──────────────────────────────────────────────────
  const stopVisualizer = useCallback(() => {
    if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
    if (sourceRef.current) sourceRef.current.disconnect();
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  }, []);

  const drawIdle = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const bars = 100;
    const bw = W / bars;
    for (let i = 0; i < bars; i++) {
      ctx.fillStyle = "rgba(0, 209, 224, 0.1)";
      ctx.beginPath();
      ctx.roundRect(i * bw, H / 2 - 1, bw - 1, 2, 1);
      ctx.fill();
    }
  }, []);

  const startVisualizer = (fileOrBlob) => {
    stopVisualizer();
    if (!canvasRef.current) return;

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = audioCtx;
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyserRef.current = analyser;

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const audioBuffer = await audioCtx.decodeAudioData(e.target.result);
        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
        sourceRef.current = source;
        source.start(0);
      } catch (err) {
        console.error("Audio decode error:", err);
      }
    };
    reader.readAsArrayBuffer(fileOrBlob);

    const canvas = canvasRef.current;
    const W = canvas.width, H = canvas.height;
    const freqData = new Uint8Array(analyser.frequencyBinCount);
    const bars = 100;
    const bw = W / bars;

    const draw = () => {
      if (!analyserRef.current) return;
      reqIdRef.current = requestAnimationFrame(draw);
      analyserRef.current.getByteFrequencyData(freqData);
      const canvasCtx = canvas.getContext("2d");
      canvasCtx.clearRect(0, 0, W, H);

      freqData.forEach((val, i) => {
        const ratio = val / 255;
        const h = Math.max(3, ratio * H * 0.9);
        const r = Math.round(0 + 242 * ratio);
        const g = Math.round(209 - 117 * ratio);
        const b = Math.round(224 - 132 * ratio);
        canvasCtx.fillStyle = `rgba(${r},${g},${b},${0.5 + ratio * 0.5})`;
        canvasCtx.beginPath();
        canvasCtx.roundRect(i * (bw + 1), H - h, bw, h, 2);
        canvasCtx.fill();
      });
    };
    draw();
  };

  // Draw idle bars on mount
  useEffect(() => { drawIdle(); return () => stopVisualizer(); }, [drawIdle, stopVisualizer]);

  const [dragActive, setDragActive] = useState(false);
  const [activeTab, setActiveTab] = useState("file"); // "file" | "url"
  const [url, setUrl] = useState("");

  // ── File handlers ─────────────────────────────────────────────────────────
  const handleBrowseClick = () => fileInputRef.current.click();

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setResult(null);
      stopVisualizer();
      drawIdle();
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (activeTab !== "file") return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
      stopVisualizer();
      drawIdle();
    }
  };

  // ── Analyze ───────────────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (activeTab === "file" && !file) return;
    if (activeTab === "url" && !url.trim()) return;
    
    setIsAnalyzing(true);
    setResult(null);

    try {
      let response;
      if (activeTab === "file") {
        startVisualizer(file);
        const formData = new FormData();
        formData.append("file", file);
        response = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          body: formData,
        });
      } else {
        stopVisualizer();
        drawIdle();
        response = await fetch(`http://127.0.0.1:8000/predict-url?url=${encodeURIComponent(url)}`, {
          method: "POST",
        });
      }

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      
      setResult(data);
      
      saveAnalysis({
        ...data,
        filename: activeTab === 'file' ? file.name : null
      });

    } catch (err) {
      console.error(err);
      alert("Error analyzing audio: " + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Verdict helpers ───────────────────────────────────────────────────────
  const isReal = result?.prediction === "REAL";
  const verdictColor = isReal ? "#00d1e0" : "#f25c2c";
  const verdictGlow = isReal
    ? "0 0 32px rgba(0,209,224,0.35)"
    : "0 0 32px rgba(242,92,44,0.35)";
  const verdictBg = isReal
    ? "rgba(0,209,224,0.06)"
    : "rgba(242,92,44,0.06)";
  const verdictBorder = isReal
    ? "1px solid rgba(0,209,224,0.25)"
    : "1px solid rgba(242,92,44,0.25)";

  return (
    <div className="dashboard">
      {/* ── Hero ── */}
      <div className="hero-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(0,212,200,0.07)', border: '1px solid rgba(0,212,200,0.22)', borderRadius: '100px', padding: '8px 20px', fontSize: '10px', letterSpacing: '0.14em', color: '#00d4c8', fontWeight: 600, textTransform: 'uppercase', marginBottom: '40px' }}>
          <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#00d4c8', boxShadow: '0 0 8px #00d4c8', animation: 'livePulse 2s infinite' }}></div>
          THREAT INTELLIGENCE DASHBOARD
        </div>
        <h1 className="hero-title">
          DETECT AI <span className="text-orange">VOICES</span>
          <br />
          BEFORE THEY{" "}
          <span className="val-cyan" style={{ textShadow: "0 0 60px rgba(0,212,200,0.5)" }}>
            DECEIVE
          </span>
        </h1>
        <p className="hero-subtitle">
          Real-time deepfake voice detection powered by CNN
          <br />
          spectrogram analysis. Upload any audio — get a verdict in
          <br />
          under two seconds.
        </p>
      </div>

      {/* ── Upload Container ── */}
      <div className="upload-container">
        
        {/* Tabs */}
        <div style={{ display: "flex", gap: "16px", marginBottom: "20px", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "12px" }}>
          <button 
            onClick={() => setActiveTab("file")}
            style={{
              background: "transparent", border: "none", color: activeTab === "file" ? "#00d1e0" : "rgba(255,255,255,0.5)",
              fontSize: "14px", fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: "8px",
              borderBottom: activeTab === "file" ? "2px solid #00d1e0" : "2px solid transparent", paddingBottom: "14px", marginBottom: "-14px"
            }}
          >
            <i className="ti ti-file-upload"></i> UPLOAD FILE
          </button>
          <button 
            onClick={() => setActiveTab("url")}
            style={{
              background: "transparent", border: "none", color: activeTab === "url" ? "#00d1e0" : "rgba(255,255,255,0.5)",
              fontSize: "14px", fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: "8px",
              borderBottom: activeTab === "url" ? "2px solid #00d1e0" : "2px solid transparent", paddingBottom: "14px", marginBottom: "-14px"
            }}
          >
            <i className="ti ti-link"></i> PASTE URL
          </button>
        </div>

        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          accept="audio/*"
          onChange={handleFileChange}
        />

        {activeTab === "file" ? (
          <>
            <div
              className={`upload-zone ${dragActive ? "drag-active" : ""}`}
              onClick={handleBrowseClick}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              style={{ cursor: "pointer" }}
            >
              <div className="upload-icon-wrapper">
                <i className="ti ti-cloud-upload"></i>
              </div>
              <p>
                Drop audio file here or{" "}
                <span style={{ color: "var(--accent-orange)" }}>browse files</span>
              </p>
              <div className="upload-formats">
                WAV • MP3 • FLAC • OGG • M4A • OPUS • up to 25 MB
              </div>
            </div>

            {file && (
              <div className="file-row">
                <div className="file-pill">
                  <i className="ti ti-file-music" style={{ fontSize: 18 }}></i>
                  {file.name}
                </div>
                <button
                  className="btn-primary"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    "ANALYZING..."
                  ) : (
                    <>ANALYZE <i className="ti ti-arrow-right"></i></>
                  )}
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="url-row" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ position: "relative" }}>
              <i className="ti ti-link" style={{ position: "absolute", left: "16px", top: "50%", transform: "translateY(-50%)", color: "var(--accent-cyan)", fontSize: "20px" }}></i>
              <input 
                type="text" 
                placeholder="Paste YouTube, SoundCloud, or direct audio link..." 
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                style={{
                  width: "100%", padding: "16px 16px 16px 48px", borderRadius: "12px",
                  background: "rgba(0, 209, 224, 0.05)", border: "1px solid rgba(0, 209, 224, 0.2)",
                  color: "white", fontSize: "14px", outline: "none", fontFamily: "inherit"
                }}
              />
            </div>
            <button
              className="btn-primary"
              onClick={handleAnalyze}
              disabled={isAnalyzing || !url.trim()}
              style={{ width: "100%", justifyContent: "center" }}
            >
              {isAnalyzing ? (
                "DOWNLOADING & ANALYZING..."
              ) : (
                <>ANALYZE URL <i className="ti ti-arrow-right"></i></>
              )}
            </button>
          </div>
        )}

        {/* ── Live Spectrogram Visualizer ── */}
        <div style={{
          marginTop: "20px",
          background: "rgba(0,0,0,0.25)",
          borderRadius: "12px",
          border: "1px solid rgba(0,209,224,0.12)",
          padding: "12px 16px",
          position: "relative",
          overflow: "hidden",
        }}>
          <div style={{
            fontSize: "10px",
            letterSpacing: "2px",
            color: "rgba(0,209,224,0.5)",
            marginBottom: "8px",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: isAnalyzing ? "#00d1e0" : "rgba(0,209,224,0.3)",
              boxShadow: isAnalyzing ? "0 0 8px #00d1e0" : "none",
              display: "inline-block",
              animation: isAnalyzing ? "pulse 1s ease-in-out infinite" : "none",
            }} />
            LIVE SPECTROGRAM {isAnalyzing ? "— ANALYZING" : ""}
          </div>
          <canvas
            ref={canvasRef}
            width={600}
            height={80}
            style={{ width: "100%", height: "80px", display: "block" }}
          />
        </div>

        {/* ── Result Card ── */}
        {result && (
          <div style={{
            marginTop: "16px",
            padding: "24px",
            background: verdictBg,
            borderRadius: "16px",
            border: verdictBorder,
            boxShadow: verdictGlow,
            display: "flex",
            alignItems: "center",
            gap: "24px",
            flexWrap: "wrap",
          }}>
            <div style={{
              width: 64,
              height: 64,
              borderRadius: "50%",
              background: verdictBg,
              border: verdictBorder,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              fontSize: 28,
              color: verdictColor,
            }}>
              {isReal ? (
                <i className="ti ti-shield-check"></i>
              ) : (
                <i className="ti ti-alert-triangle"></i>
              )}
            </div>

            <div style={{ flex: 1 }}>
              <div style={{
                fontSize: "11px",
                letterSpacing: "2px",
                color: "rgba(255,255,255,0.4)",
                marginBottom: "4px",
              }}>
                VERDICT
              </div>
              <div style={{
                fontSize: "28px",
                fontWeight: 800,
                letterSpacing: "3px",
                color: verdictColor,
                textShadow: verdictGlow,
                marginBottom: "6px",
              }}>
                {result.prediction === "REAL" ? "✓ HUMAN VOICE" : "⚠ AI DEEPFAKE"}
              </div>
              <div style={{ fontSize: "13px", color: "rgba(255,255,255,0.6)" }}>
                {result.message}
              </div>
            </div>

            <div style={{
              textAlign: "center",
              flexShrink: 0,
            }}>
              <div style={{
                fontSize: "32px",
                fontWeight: 800,
                color: verdictColor,
                lineHeight: 1,
              }}>
                {Number(result.confidence).toFixed(1)}%
              </div>
              <div style={{
                fontSize: "10px",
                letterSpacing: "1.5px",
                color: "rgba(255,255,255,0.4)",
                marginTop: "4px",
              }}>
                CONFIDENCE
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="section-divider">
        <span className="section-divider-text" style={{ fontSize: '12px', letterSpacing: '0.3em', color: '#3d6e6a' }}>SESSION ANALYTICS</span>
      </div>

      <FakeRateChart />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        <ConfidenceHistogram />
        
        {/* Quick Stats Panel */}
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '24px', backdropFilter: 'blur(16px)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ fontSize: '10px', letterSpacing: '0.2em', color: '#3d6e6a', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 700 }}>
            QUICK STATS
          </div>

          <div style={{ background: 'rgba(232,82,30,0.05)', border: '1px solid rgba(232,82,30,0.1)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', color: '#7ea8a4', marginBottom: '4px' }}>Most Confident FAKE</div>
            {quickStats.maxFake ? (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#dfe8e6', fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                  {quickStats.maxFake.filename}
                </span>
                <span style={{ color: '#e8521e', fontWeight: 600 }}>{quickStats.maxFake.confidence.toFixed(1)}%</span>
              </div>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.3)', fontSize: '13px' }}>No fakes detected yet</div>
            )}
          </div>

          <div style={{ background: 'rgba(0,212,200,0.05)', border: '1px solid rgba(0,212,200,0.1)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', color: '#7ea8a4', marginBottom: '4px' }}>Most Confident REAL</div>
            {quickStats.maxReal ? (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#dfe8e6', fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
                  {quickStats.maxReal.filename}
                </span>
                <span style={{ color: '#00d4c8', fontWeight: 600 }}>{quickStats.maxReal.confidence.toFixed(1)}%</span>
              </div>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.3)', fontSize: '13px' }}>No real voices detected yet</div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '16px' }}>
            <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#f0a429', marginBottom: '4px' }}>{quickStats.streak}</div>
              <div style={{ fontSize: '11px', color: '#7ea8a4', lineHeight: 1.2 }}>Longest Fake Streak</div>
            </div>
            <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#00d4c8', marginBottom: '4px' }}>{quickStats.todayCount}</div>
              <div style={{ fontSize: '11px', color: '#7ea8a4', lineHeight: 1.2 }}>Analyses Today</div>
            </div>
          </div>

        </div>
      </div>

      <HistoryTable />

    </div>
  );
};

export default Dashboard;
