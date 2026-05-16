import React, { useState, useRef, useEffect, useCallback } from "react";

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);

  // ── Spectrogram animator ──────────────────────────────────────────────────
  const stopVisualizer = useCallback(() => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (sourceRef.current) { try { sourceRef.current.stop(); } catch (_) {} }
    if (audioCtxRef.current) { try { audioCtxRef.current.close(); } catch (_) {} }
    animFrameRef.current = null;
    sourceRef.current = null;
    audioCtxRef.current = null;
    analyserRef.current = null;
  }, []);

  const drawIdle = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const bars = 64;
    const bw = W / bars - 2;
    for (let i = 0; i < bars; i++) {
      const h = 4 + Math.random() * 8;
      ctx.fillStyle = "rgba(0,209,224,0.18)";
      ctx.beginPath();
      ctx.roundRect(i * (bw + 2), H / 2 - h / 2, bw, h, 2);
      ctx.fill();
    }
  }, []);

  const startVisualizer = useCallback(async (audioFile) => {
    stopVisualizer();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const arrayBuffer = await audioFile.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = ctx;

    let buffer;
    try {
      buffer = await ctx.decodeAudioData(arrayBuffer);
    } catch {
      drawIdle();
      return;
    }

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 128;
    analyserRef.current = analyser;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(analyser);
    analyser.connect(ctx.destination);
    sourceRef.current = source;
    source.start(0);
    source.onended = stopVisualizer;

    const freqData = new Uint8Array(analyser.frequencyBinCount);
    const W = canvas.width, H = canvas.height;
    const bars = freqData.length;
    const bw = W / bars - 1;

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(freqData);
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
  }, [stopVisualizer, drawIdle]);

  // Draw idle bars on mount
  useEffect(() => { drawIdle(); return () => stopVisualizer(); }, [drawIdle, stopVisualizer]);

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

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
      stopVisualizer();
      drawIdle();
    }
  };

  // ── Analyze ───────────────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setResult(null);

    // Start live visualizer
    startVisualizer(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing audio: " + err.message);
    } finally {
      setIsAnalyzing(false);
      // Let visualizer keep playing; it'll stop naturally when audio ends
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
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          accept="audio/*"
          onChange={handleFileChange}
        />

        {/* Drop zone */}
        <div
          className="upload-zone"
          onClick={handleBrowseClick}
          onDragOver={(e) => e.preventDefault()}
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

        {/* File row */}
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
          }}>
            {/* Big verdict icon */}
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

            {/* Confidence ring */}
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

      {/* ── Stats ── */}
      <div className="stats-row">
        <div className="stat-pill"><strong>31K+</strong> voices analyzed</div>
        <div className="stat-pill"><strong>98.1%</strong> val accuracy</div>
        <div className="stat-pill"><strong>&lt;2s</strong> detection time</div>
        <div className="stat-pill"><strong>7</strong> audio formats</div>
        <div className="stat-pill"><strong>0.3%</strong> false negatives</div>
      </div>

      <div className="section-divider">
        <span className="section-divider-text">HOW IT WORKS</span>
      </div>

      <div className="hiw-grid">
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

      <div className="section-divider">
        <span className="section-divider-text">MODEL ACCURACY</span>
      </div>

      <div className="accuracy-grid">
        <div className="accuracy-card">
          <div className="accuracy-value val-cyan">98.1%</div>
          <div className="accuracy-desc">Validation accuracy on held-out dataset of 6,200 samples</div>
        </div>
        <div className="accuracy-card">
          <div className="accuracy-value val-orange">0.3%</div>
          <div className="accuracy-desc">False negative rate — real voice incorrectly flagged as fake</div>
        </div>
        <div className="accuracy-card">
          <div className="accuracy-value val-white">1.6%</div>
          <div className="accuracy-desc">False positive rate — deepfake voice slipping through as real</div>
        </div>
        <div className="accuracy-card">
          <div
            className="accuracy-value val-orange"
            style={{ color: "#ffc107", textShadow: "0 0 40px rgba(255,193,7,0.4)" }}
          >
            31K+
          </div>
          <div className="accuracy-desc">Total voice samples analyzed since public launch</div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
