import React, { useState, useRef, useEffect, useCallback } from "react";
import { saveAnalysis } from "../utils/storage";
import ModelSelector from "./ModelSelector";
import Footer from "./Footer";

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedModel, setSelectedModel] = useState("best");

  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const reqIdRef = useRef(null);

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
    const W = canvas.width,
      H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const bars = 100;
    const bw = W / bars;
    for (let i = 0; i < bars; i++) {
      ctx.fillStyle = "rgba(78, 125, 150, 0.12)";
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

        source.onended = () => {
          if (reqIdRef.current) {
            cancelAnimationFrame(reqIdRef.current);
            reqIdRef.current = null;
          }
        };
      } catch (err) {
        console.error("Audio decode error:", err);
      }
    };
    reader.readAsArrayBuffer(fileOrBlob);

    const canvas = canvasRef.current;
    const W = canvas.width,
      H = canvas.height;
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
        // Nordic Lake → Marie gradient based on intensity
        const r = Math.round(78 + (166 - 78) * ratio);
        const g = Math.round(125 - (125 - 58) * ratio);
        const b = Math.round(150 - (150 - 63) * ratio);
        canvasCtx.fillStyle = `rgba(${r},${g},${b},${0.5 + ratio * 0.5})`;
        canvasCtx.beginPath();
        canvasCtx.roundRect(i * (bw + 1), H - h, bw, h, 2);
        canvasCtx.fill();
      });
    };
    draw();
  };

  // Draw idle bars on mount
  useEffect(() => {
    drawIdle();
    return () => stopVisualizer();
  }, [drawIdle, stopVisualizer]);

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

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    try {
      let response;
      if (activeTab === "file") {
        startVisualizer(file);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", selectedModel);
        response = await fetch(`${apiUrl}/predict`, {
          method: "POST",
          body: formData,
        });
      } else {
        stopVisualizer();
        drawIdle();
        response = await fetch(
          `${apiUrl}/predict-url?url=${encodeURIComponent(url)}&model=${selectedModel}`,
          {
            method: "POST",
          },
        );
      }

      const data = await response.json().catch(() => ({}));
      if (!response.ok)
        throw new Error(data.detail || data.error || "Analysis failed");

      setResult(data);

      saveAnalysis({
        ...data,
        filename: activeTab === "file" ? file.name : null,
      });
    } catch (err) {
      console.error(err);
      alert("Error analyzing audio: " + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Nordic Lake = real/human; Marie = fake/deepfake
  const isReal = result?.prediction === "REAL";
  const verdictColor  = isReal ? "#C6A75E" : "#A63A3F";
  const verdictGlow   = isReal ? "0 0 32px rgba(123,157,174,0.35)" : "0 0 32px rgba(122,46,50,0.45)";
  const verdictBg     = isReal ? "rgba(123,157,174,0.07)" : "rgba(122,46,50,0.10)";
  const verdictBorder = isReal ? "1px solid rgba(123,157,174,0.25)" : "1px solid rgba(122,46,50,0.30)";

  return (
    <div className="dashboard">
      {/* ── Hero ── */}
      <div className="hero-section">
        <h1 className="hero-title">
          DETECT AI <span className="text-marie" style={{ WebkitTextStroke: '1px #3d6e6a' }}>VOICES</span>
          <br />
          BEFORE THEY{" "}
          <span
            className="val-cyan"
            style={{ textShadow: "0 0 60px rgba(123,157,174,0.5)", WebkitTextStroke: '1px #3d6e6a' }}
          >
            DECEIVE
          </span>
        </h1>
        <p className="hero-subtitle">
          Real-time deepfake voice detection via spectrogram analysis.
          <br />
          Upload any audio — get a verdict in under 2 seconds.
        </p>
      </div>

      {/* ── Upload Container ── */}
      <div className="upload-container" style={{ width: "100%", maxWidth: "980px", padding: "36px", boxSizing: "border-box" }}>
        <ModelSelector
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: "16px",
            marginBottom: "20px",
            borderBottom: "1px solid rgba(232,220,200,0.1)",
            paddingBottom: "12px",
          }}
        >
          <button
            onClick={() => setActiveTab("file")}
            style={{
              background: "transparent",
              border: "none",
          color: activeTab === "file" ? "var(--accent-nordic)" : "rgba(26,18,16,0.40)",
              fontSize: "14px",
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              borderBottom:
                activeTab === "file"
                  ? "2px solid var(--accent-nordic)"
                  : "2px solid transparent",
              paddingBottom: "14px",
              marginBottom: "-14px",
            }}
          >
            <i className="ti ti-file-upload"></i> UPLOAD FILE
          </button>
          <button
            onClick={() => setActiveTab("url")}
            style={{
              background: "transparent",
              border: "none",
              color: activeTab === "url" ? "var(--accent-nordic)" : "rgba(26,18,16,0.40)",
              fontSize: "14px",
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              borderBottom:
                activeTab === "url"
                  ? "2px solid var(--accent-nordic)"
                  : "2px solid transparent",
              paddingBottom: "14px",
              marginBottom: "-14px",
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
              style={{ cursor: "pointer", padding: "85px 20px" }}
            >
              <div className="upload-icon-wrapper">
                <i className="ti ti-cloud-upload"></i>
              </div>
              <p>
                Drop audio file here or{" "}
                <span style={{ color: "var(--accent-orange)" }}>
                  browse files
                </span>
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
                    <>
                      ANALYZE <i className="ti ti-arrow-right"></i>
                    </>
                  )}
                </button>
              </div>
            )}
          </>
        ) : (
          <div
            className="url-row"
            style={{ display: "flex", flexDirection: "column", gap: "16px" }}
          >
            <div style={{ position: "relative" }}>
              <i
                className="ti ti-link"
                style={{
                  position: "absolute",
                  left: "16px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--accent-nordic)",
                  fontSize: "20px",
                }}
              ></i>
              <input
                type="text"
                placeholder="Paste YouTube, SoundCloud, or direct audio link..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                style={{
                  width: "100%",
                  padding: "16px 16px 16px 48px",
                  borderRadius: "12px",
                  background: "var(--nordic-dim)",
                  border: "1px solid var(--nordic-border)",
                  color: "var(--text-card)",
                  fontSize: "14px",
                  outline: "none",
                  fontFamily: "inherit",
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
                <>
                  ANALYZE URL <i className="ti ti-arrow-right"></i>
                </>
              )}
            </button>
          </div>
        )}

        {/* ── Live Spectrogram Visualizer ── */}
        <div
          style={{
            display: (isAnalyzing || result) ? "block" : "none",
            marginTop: "20px",
            background: "rgba(232,220,200,0.25)",
            borderRadius: "12px",
            border: "1px solid var(--nordic-border)",
            padding: "12px 16px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <canvas
            ref={canvasRef}
            width={600}
            height={80}
            style={{ width: "100%", height: "80px", display: "block" }}
          />
        </div>

        {/* ── Result Card ── */}
        {result && (
          <div
            style={{
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
            }}
          >
            <div
              style={{
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
              }}
            >
              {isReal ? (
                <i className="ti ti-shield-check"></i>
              ) : (
                <i className="ti ti-alert-triangle"></i>
              )}
            </div>

            <div style={{ flex: 1 }}>
              <div
                style={{
                  fontSize: "11px",
                  letterSpacing: "2px",
                  color: "rgba(232,220,200,0.4)",
                  marginBottom: "4px",
                }}
              >
                VERDICT
              </div>
              <div
                style={{
                  fontSize: "28px",
                  fontWeight: 800,
                  letterSpacing: "3px",
                  color: verdictColor,
                  textShadow: verdictGlow,
                  marginBottom: "6px",
                }}
              >
                {result.prediction === "REAL" ? "HUMAN VOICE" : "AI DEEPFAKE"}
              </div>
              <div style={{ fontSize: "13px", color: "rgba(232,220,200,0.6)" }}>
                {result.message}
              </div>
            </div>

            <div
              style={{
                textAlign: "center",
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  fontSize: "32px",
                  fontWeight: 800,
                  color: verdictColor,
                  lineHeight: 1,
                }}
              >
                {Number(result.confidence).toFixed(1)}%
              </div>
              <div
                style={{
                  fontSize: "10px",
                  letterSpacing: "1.5px",
                  color: "rgba(232,220,200,0.4)",
                  marginTop: "4px",
                }}
              >
                CONFIDENCE
              </div>
            </div>

            {/* AI EXPLAINABILITY HEATMAP - FULL WIDTH ROW */}
            {result.heatmap && (
              <div
                style={{
                  width: "100%",
                  marginTop: "16px",
                  paddingTop: "16px",
                  borderTop: "1px solid rgba(232,220,200,0.1)",
                  animation: "fadeIn 0.5s ease-out",
                }}
              >
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--text-muted)",
                    marginBottom: "16px",
                    textTransform: "uppercase",
                    letterSpacing: "1px",
                    textAlign: "center",
                    fontWeight: 600
                  }}
                >
                  CNN ACTIVATION HEATMAP
                </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <img
                    src={result.heatmap}
                    alt="Neural Attention Heatmap"
                    style={{
                      width: "100%",
                      maxWidth: "280px",
                      borderRadius: "12px",
                      border: "1px solid rgba(232,220,200,0.1)",
                      objectFit: "contain",
                      boxShadow: "0 8px 32px rgba(232,220,200,0.4)"
                    }}
                  />
                </div>
                <p
                  style={{
                    fontSize: "12px",
                    color: "rgba(232,220,200,0.6)",
                    marginTop: "16px",
                    lineHeight: 1.5,
                    textAlign: "center",
                    maxWidth: "400px",
                    margin: "16px auto 0"
                  }}
                >
                   The <span style={{ color: "var(--accent-nordic)", fontWeight: "bold" }}>warm regions</span>{" "}
                  highlight the specific audio frequencies that triggered the AI's detection model.
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── FOOTER ── */}
      <Footer />
    </div>
  );
};

export default Dashboard;
