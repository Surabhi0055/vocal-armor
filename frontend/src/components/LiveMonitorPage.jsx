import React, { useState, useRef, useEffect, useCallback } from 'react';
import { saveAnalysis } from '../utils/storage';
import ModelSelector from './ModelSelector';
import Footer from './Footer';

// ── WAV encoder (pure JS, no libraries needed) ─────────────────────────────
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) 
      view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}

const formatTime = (seconds) => {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = (seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
};

const LiveMonitorPage = () => {
  const [status, setStatus] = useState('idle'); // idle | recording | analyzing | done | error
  const [result, setResult] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0); // seconds
  const [audioLevel, setAudioLevel] = useState(0); // 0-100 for volume meter
  const [errorMsg, setErrorMsg] = useState('');
  const [totalSamplesRecorded, setTotalSamplesRecorded] = useState(0);
  const [selectedModel, setSelectedModel] = useState('best');

  // Refs — don't need re-render on change
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const samplesRef = useRef(new Float32Array(0)); // accumulate ALL samples
  const actualSampleRateRef = useRef(44100);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const timerRef = useRef(null);
  const audioLevelRef = useRef(null);

  // ── Waveform visualizer ──────────────────────────────────────────────────
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    const bufLen = analyser.frequencyBinCount;
    const dataArr = new Uint8Array(bufLen);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArr);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#e8521e';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#e8521e';
      ctx.beginPath();

      const sliceWidth = canvas.width / bufLen;
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const v = dataArr[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  }, []);

  const drawIdleWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }, []);

  // ── Start Recording ─────────────────────────────────────────────────────
  const startRecording = async () => {
    setErrorMsg('');
    setResult(null);
    setRecordingTime(0);
    setStatus('recording');
    samplesRef.current = new Float32Array(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        }
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      actualSampleRateRef.current = audioCtx.sampleRate;
      audioCtxRef.current = audioCtx;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      analyser.connect(processor);
      processor.connect(audioCtx.destination);

      // Accumulate ALL samples — no chunking
      processor.onaudioprocess = (event) => {
        const chunk = event.inputBuffer.getChannelData(0);
        const prev = samplesRef.current;
        const combined = new Float32Array(prev.length + chunk.length);
        combined.set(prev);
        combined.set(chunk, prev.length);
        samplesRef.current = combined;
      };

      // Recording timer
      timerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1);
      }, 1000);

      // Volume meter
      const updateLevel = () => {
        audioLevelRef.current = requestAnimationFrame(updateLevel);
        const arr = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(arr);
        const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
        setAudioLevel(Math.round((avg / 255) * 100));
      };
      updateLevel();

      // Start waveform
      drawWaveform();

    } catch (err) {
      setStatus('error');
      if (err.name === 'NotAllowedError') {
        setErrorMsg('Microphone access denied. Allow mic permissions and try again.');
      } else if (err.name === 'NotFoundError') {
        setErrorMsg('No microphone found. Please connect a microphone.');
      } else {
        setErrorMsg(`Error: ${err.message}`);
      }
    }
  };

  // ── Stop and Analyze ──────────────────────────────────────────────────────
  const stopAndAnalyze = async () => {
    setStatus('analyzing');
    clearInterval(timerRef.current);
    cancelAnimationFrame(animFrameRef.current);
    cancelAnimationFrame(audioLevelRef.current);

    // Stop mic
    if (processorRef.current) { 
      processorRef.current.disconnect(); 
      processorRef.current = null; 
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) { 
      audioCtxRef.current.close(); 
      audioCtxRef.current = null; 
    }

    drawIdleWaveform();

    const samples = samplesRef.current;
    const sampleRate = actualSampleRateRef.current;
    setTotalSamplesRecorded(samples.length);

    // Need at least 1 second of audio
    if (samples.length < sampleRate) {
      setStatus('error');
      setErrorMsg('Recording too short. Please record at least 2 seconds of audio.');
      return;
    }

    try {
      // Encode full session as WAV
      const wavBuffer = encodeWAV(samples, sampleRate);
      const blob = new Blob([wavBuffer], { type: 'audio/wav' });
      const formData = new FormData();
      formData.append('file', blob, 'live_session.wav');
      formData.append('model', selectedModel);

      console.log(
        `Sending full session — samples: ${samples.length}, ` +
        `rate: ${sampleRate}, ` +
        `duration: ${(samples.length / sampleRate).toFixed(1)}s, ` +
        `size: ${(wavBuffer.byteLength / 1024 / 1024).toFixed(2)}MB`
      );

      // Send to regular POST /predict endpoint — no WebSocket needed
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await response.json();
      console.log('Final result:', data);

      setResult(data);
      setStatus('done');

      // Save to history
      saveAnalysis({
        ...data,
        filename: `live_session_${recordingTime}s.wav`,
        source_url: null,
      });

    } catch (err) {
      setStatus('error');
      setErrorMsg(`Analysis failed: ${err.message}`);
    }
  };

  const resetRecording = () => {
    setStatus('idle');
    setResult(null);
    setRecordingTime(0);
    setAudioLevel(0);
    setErrorMsg('');
    samplesRef.current = new Float32Array(0);
    drawIdleWaveform();
  };

  // Draw idle flat line on mount
  useEffect(() => {
    drawIdleWaveform();
  }, [drawIdleWaveform]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      cancelAnimationFrame(animFrameRef.current);
      cancelAnimationFrame(audioLevelRef.current);
      if (processorRef.current) processorRef.current.disconnect();
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      if (audioCtxRef.current) audioCtxRef.current.close();
    };
  }, []);

  const isRecording = status === 'recording';
  const verdictColor = result?.is_deepfake ? '#e8521e' : '#00d4c8';
  const verdictGlow = result?.is_deepfake
    ? '0 0 32px rgba(232,82,30,0.5)'
    : '0 0 32px rgba(0,212,200,0.5)';

  return (
    <div style={{ padding: '40px', maxWidth: '1000px', margin: '0 auto', width: '100%', zIndex: 2, position: 'relative', paddingBottom: '100px' }}>

      {/* Page Header */}
      <div style={{ marginBottom: '40px' }}>

        <h1 style={{ fontFamily: '"Bebas Neue", sans-serif', fontSize: '48px', fontWeight: 400, letterSpacing: '2px', lineHeight: 1, marginBottom: '12px', textTransform: 'uppercase' }}>
          REAL-TIME <span style={{ color: '#e8521e', textShadow: '0 0 40px rgba(232,82,30,0.4)' }}>STREAM ANALYSIS</span>
        </h1>
        <p style={{ fontSize: '14px', color: '#7ea8a4', lineHeight: 1.6 }}>
          Speak into your microphone. VocalArmor analyzes every 2-second window for deepfake patterns.
        </p>
      </div>

      {/* Main Card */}
      <div style={{ background: '#0f2229', border: `1px solid ${isRecording ? 'rgba(232,82,30,0.3)' : 'rgba(255,255,255,0.08)'}`, borderRadius: '20px', padding: '32px', marginBottom: '24px', transition: 'border-color 0.3s', boxShadow: isRecording ? '0 0 40px rgba(232,82,30,0.1)' : 'none' }}>

        <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />

        {/* Waveform Canvas */}
        <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '8px', marginBottom: '28px', position: 'relative', overflow: 'hidden' }}>
          <canvas
            ref={canvasRef}
            width={900}
            height={120}
            style={{ width: '100%', height: '120px', display: 'block' }}
          />
          {!isRecording && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
              <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: '13px', letterSpacing: '2px', textTransform: 'uppercase' }}>
                Waveform will appear here
              </span>
            </div>
          )}
        </div>

        {/* Controls Row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>

          {status === 'idle' && (
            <>
              <button
                onClick={startRecording}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'linear-gradient(135deg, #e8521e, #ff8a00)', color: 'white', border: 'none', borderRadius: '10px', padding: '14px 28px', fontSize: '14px', fontWeight: 700, cursor: 'pointer', letterSpacing: '1px', boxShadow: '0 4px 20px rgba(232,82,30,0.4)', transition: 'all 0.2s' }}>
                <i className="ti ti-player-record" style={{ fontSize: '18px' }}></i>
                START RECORDING
              </button>
              <div style={{ fontSize: '13px', color: '#7ea8a4' }}>
                Click to start recording your voice session
              </div>
            </>
          )}

          {status === 'recording' && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#e8521e', animation: 'livePulse 2s infinite' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#e8521e' }}></div>
                  <span style={{ fontSize: '12px', fontWeight: 700, letterSpacing: '1px' }}>REC</span>
                </div>
                <div style={{ fontFamily: 'monospace', fontSize: '24px', color: '#e8521e', fontWeight: 700 }}>
                  {formatTime(recordingTime)}
                </div>
              </div>
              <button
                onClick={stopAndAnalyze}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'transparent', color: '#e8521e', border: '1px solid rgba(232,82,30,0.4)', borderRadius: '10px', padding: '14px 28px', fontSize: '14px', fontWeight: 700, cursor: 'pointer', letterSpacing: '1px' }}>
                <i className="ti ti-player-stop" style={{ fontSize: '18px' }}></i>
                STOP &amp; ANALYZE
              </button>
              <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: '150px' }}>
                <div style={{ fontSize: '13px', color: '#7ea8a4' }}>Recording in progress — speak now</div>
                <div style={{ height: '3px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden', marginTop: '12px' }}>
                  <div style={{
                    width: `${audioLevel}%`, height: '100%',
                    background: audioLevel > 70 ? '#e8521e' : audioLevel > 30 ? '#f0a429' : '#00d4c8',
                    transition: 'width 0.1s, background 0.2s', borderRadius: '2px'
                  }} />
                </div>
                <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.3)', marginTop: '4px', letterSpacing: '1px' }}>MIC LEVEL</div>
              </div>
            </>
          )}

          {status === 'analyzing' && (
            <>
              <button disabled style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(255,255,255,0.05)', color: '#7ea8a4', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', padding: '14px 28px', fontSize: '14px', fontWeight: 700, cursor: 'not-allowed', letterSpacing: '1px' }}>
                <i className="ti ti-loader ti-spin" style={{ fontSize: '18px' }}></i>
                ANALYZING...
              </button>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ fontSize: '13px', color: '#7ea8a4', fontWeight: 600 }}>Analyzing full session...</div>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.3)' }}>Running CNN inference on recorded audio</div>
              </div>
            </>
          )}
          
          {status === 'done' && (
            <>
              <button
                onClick={resetRecording}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'transparent', color: '#00d4c8', border: '1px solid rgba(0,212,200,0.4)', borderRadius: '10px', padding: '14px 28px', fontSize: '14px', fontWeight: 700, cursor: 'pointer', letterSpacing: '1px' }}>
                <i className="ti ti-refresh" style={{ fontSize: '18px' }}></i>
                ANALYZE ANOTHER
              </button>
            </>
          )}
          
          {status === 'error' && (
            <>
              <button
                onClick={resetRecording}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'transparent', color: '#7ea8a4', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '10px', padding: '14px 28px', fontSize: '14px', fontWeight: 700, cursor: 'pointer', letterSpacing: '1px' }}>
                <i className="ti ti-refresh" style={{ fontSize: '18px' }}></i>
                TRY AGAIN
              </button>
            </>
          )}
        </div>

        {/* Error Message */}
        {errorMsg && (
          <div style={{ marginTop: '20px', background: 'rgba(232,82,30,0.08)', border: '1px solid rgba(232,82,30,0.3)', borderRadius: '10px', padding: '14px 16px', color: '#e8521e', fontSize: '13px', display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
            <i className="ti ti-alert-circle" style={{ fontSize: '18px', flexShrink: 0, marginTop: '1px' }}></i>
            {errorMsg}
          </div>
        )}
      </div>

      {/* Result Card */}
      {status === 'done' && result && (
        <div style={{ background: '#0f2229', border: `1px solid ${result.is_deepfake ? 'rgba(232,82,30,0.3)' : 'rgba(0,212,200,0.3)'}`, borderRadius: '20px', padding: '32px', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '28px', boxShadow: verdictGlow }}>
          <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: result.is_deepfake ? 'rgba(232,82,30,0.12)' : 'rgba(0,212,200,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '28px', color: verdictColor, flexShrink: 0 }}>
            <i className={result.is_deepfake ? 'ti ti-alert-triangle' : 'ti ti-shield-check'}></i>
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '10px', letterSpacing: '2px', color: '#7ea8a4', marginBottom: '6px' }}>LATEST ANALYSIS</div>
            <div style={{ fontSize: '28px', fontWeight: 800, color: verdictColor, letterSpacing: '2px', marginBottom: '4px', textShadow: verdictGlow }}>
              {result.is_deepfake ? '⚠ AI DEEPFAKE DETECTED' : '✓ HUMAN VOICE VERIFIED'}
            </div>
            <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>
              Analyzed {recordingTime}s of audio
            </div>
            <div style={{ fontSize: '11px', color: '#7ea8a4', marginTop: '8px' }}>
              Session duration: {formatTime(recordingTime)} · Samples: {totalSamplesRecorded.toLocaleString()} · Rate: {actualSampleRateRef.current}Hz
            </div>
          </div>
          <div style={{ textAlign: 'center', flexShrink: 0 }}>
            <div style={{ fontSize: '40px', fontWeight: 800, color: verdictColor, lineHeight: 1 }}>
              {Number(result.confidence).toFixed(1)}%
            </div>
            <div style={{ fontSize: '10px', color: '#7ea8a4', marginTop: '6px', letterSpacing: '1px' }}>CONFIDENCE</div>
          </div>
        </div>
      )}

      {/* Empty state — no result yet */}
      {status === 'idle' && (
        <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '20px', padding: '60px 20px', textAlign: 'center' }}>
          <i className="ti ti-waveform" style={{ fontSize: '56px', color: 'rgba(255,255,255,0.1)', marginBottom: '20px', display: 'block' }}></i>
          <div style={{ fontSize: '16px', color: 'rgba(255,255,255,0.4)', marginBottom: '8px' }}>No analysis yet</div>
          <div style={{ fontSize: '13px', color: '#7ea8a4' }}>Click START RECORDING above to begin</div>
        </div>
      )}

      <Footer />
    </div>
  );
};

export default LiveMonitorPage;
