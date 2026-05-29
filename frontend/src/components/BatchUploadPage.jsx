import React, { useState, useRef } from 'react';
import { saveAnalysis } from '../utils/storage';
import ModelSelector from './ModelSelector';
import Footer from './Footer';

const BatchUploadPage = () => {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('idle'); // idle | analyzing | done
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [selectedModel, setSelectedModel] = useState('best');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files);
    // Filter for audio only
    const audioFiles = selected.filter(f => 
      f.type.startsWith('audio/') || 
      f.name.toLowerCase().match(/\.(wav|mp3|flac|ogg|m4a|aac|wma)$/)
    );
    setFiles(prev => [...prev, ...audioFiles]);
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

  const traverseFileTree = (item, path = '') => {
    return new Promise((resolve) => {
      if (item.isFile) {
        item.file((file) => resolve([file]));
      } else if (item.isDirectory) {
        const dirReader = item.createReader();
        const entries = [];
        const readEntries = () => {
          dirReader.readEntries(async (results) => {
            if (!results.length) {
              let files = [];
              for (const entry of entries) {
                const subFiles = await traverseFileTree(entry, path + item.name + '/');
                files = [...files, ...subFiles];
              }
              resolve(files);
            } else {
              entries.push(...results);
              readEntries();
            }
          });
        };
        readEntries();
      } else {
        resolve([]);
      }
    });
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    let allFiles = [];
    if (e.dataTransfer.items) {
      for (let i = 0; i < e.dataTransfer.items.length; i++) {
        const item = e.dataTransfer.items[i].webkitGetAsEntry();
        if (item) {
          const files = await traverseFileTree(item);
          allFiles = [...allFiles, ...files];
        }
      }
    } else {
      allFiles = Array.from(e.dataTransfer.files);
    }

    const audioFiles = allFiles.filter(f => 
      f.type.startsWith('audio/') || 
      f.name.toLowerCase().match(/\.(wav|mp3|flac|ogg|m4a|aac|wma)$/)
    );
    setFiles(prev => [...prev, ...audioFiles]);
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const startBatchAnalysis = async () => {
    if (files.length === 0) return;
    setStatus('analyzing');
    setProgress(0);
    setResults([]);

    const newResults = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', selectedModel);

      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || errorData.error || 'Analysis failed');
        }

        const data = await response.json();
        const resultItem = { ...data, filename: file.name, id: Date.now() + i };
        newResults.push(resultItem);
        
        saveAnalysis({ ...resultItem, source_url: null });
      } catch (err) {
        newResults.push({
          error: true,
          filename: file.name,
          message: err.message,
          id: Date.now() + i
        });
      }

      setProgress(i + 1);
    }

    setResults(newResults);
    setStatus('done');
  };

  const clearBatch = () => {
    setFiles([]);
    setResults([]);
    setProgress(0);
    setStatus('idle');
  };

  const fakeCount = results.filter(r => r.is_deepfake).length;
  const realCount = results.filter(r => !r.is_deepfake && !r.error).length;
  const errorCount = results.filter(r => r.error).length;

  return (
    <div className="dashboard" style={{ paddingBottom: '100px' }}>
      


      <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />

      {status === 'idle' && (
        <div 
          className={`upload-zone ${dragActive ? "drag-active" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          style={{ cursor: "pointer", marginBottom: "24px" }}
        >
          <div className="upload-icon-wrapper">
            <i className="ti ti-cloud-upload"></i>
          </div>
          <p>
            Drop multiple audio files here or{" "}
            <span style={{ color: "var(--accent-orange)" }}>browse files</span>
          </p>
          <div className="upload-formats">
            WAV • MP3 • FLAC • OGG • M4A • OPUS supported
          </div>
          
          <input 
            type="file" 
            multiple 
            accept="audio/*" 
            ref={fileInputRef} 
            onChange={handleFileSelect}
            style={{ display: 'none' }} 
          />
        </div>
      )}

      {files.length > 0 && status === 'idle' && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '20px', padding: '32px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div style={{ fontSize: '16px', fontWeight: 600 }}>{files.length} Files Queued</div>
            <button 
              className="btn-primary"
              onClick={startBatchAnalysis}>
              START BATCH ANALYSIS
            </button>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '300px', overflowY: 'auto' }}>
            {files.map((f, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(232,220,200,0.03)', padding: '12px 16px', borderRadius: '8px' }}>
                <span style={{ fontSize: '14px' }}>{f.name}</span>
                <i className="ti ti-x" style={{ cursor: 'pointer', color: 'var(--text-muted)' }} onClick={() => removeFile(i)}></i>
              </div>
            ))}
          </div>
        </div>
      )}

      {status === 'analyzing' && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(123,157,174,0.3)', borderRadius: '20px', padding: '40px', textAlign: 'center', boxShadow: '0 0 30px rgba(123,157,174,0.1)' }}>
          <i className="ti ti-loader ti-spin" style={{ fontSize: '48px', color: '#C6A75E', marginBottom: '20px', display: 'block' }}></i>
          <div style={{ fontSize: '24px', fontWeight: 600, marginBottom: '12px' }}>Analyzing Batch...</div>
          <div style={{ fontSize: '16px', color: 'var(--text-muted)', marginBottom: '24px' }}>Processing file {progress} of {files.length}</div>
          
          <div style={{ height: '8px', background: 'rgba(232,220,200,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: `${(progress / files.length) * 100}%`, height: '100%', background: '#C6A75E', transition: 'width 0.3s' }}></div>
          </div>
        </div>
      )}

      {status === 'done' && (
        <>
          <div style={{ display: 'flex', gap: '24px', marginBottom: '24px' }}>
            <div style={{ flex: 1, background: 'var(--bg-card)', border: '1px solid rgba(122,46,50,0.3)', borderRadius: '20px', padding: '32px', textAlign: 'center' }}>
              <div style={{ fontSize: '48px', fontWeight: 800, color: '#A63A3F', marginBottom: '8px' }}>{fakeCount}</div>
              <div style={{ fontSize: '12px', letterSpacing: '2px', color: 'var(--text-muted)' }}>AI DEEPFAKE</div>
            </div>
            <div style={{ flex: 1, background: 'var(--bg-card)', border: '1px solid rgba(123,157,174,0.3)', borderRadius: '20px', padding: '32px', textAlign: 'center' }}>
              <div style={{ fontSize: '48px', fontWeight: 800, color: '#C6A75E', marginBottom: '8px' }}>{realCount}</div>
              <div style={{ fontSize: '12px', letterSpacing: '2px', color: 'var(--text-muted)' }}>HUMAN VOICE</div>
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', border: '1px solid rgba(232,220,200,0.08)', borderRadius: '20px', padding: '32px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <div style={{ fontSize: '18px', fontWeight: 600 }}>Batch Results</div>
              <button 
                onClick={clearBatch}
                style={{ background: 'transparent', color: 'var(--text-muted)', border: '1px solid rgba(232,220,200,0.2)', borderRadius: '8px', padding: '8px 16px', cursor: 'pointer', fontSize: '12px', letterSpacing: '1px' }}>
                NEW BATCH
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {results.map((r, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', background: 'rgba(232,220,200,0.03)', padding: '16px', borderRadius: '12px', gap: '16px' }}>
                  <div style={{ flex: 1, fontSize: '14px' }}>{r.filename}</div>
                  {r.error ? (
                    <div style={{ color: '#A63A3F', fontSize: '13px' }}>Error: {r.message}</div>
                  ) : (
                    <>
                      <div style={{ padding: '4px 12px', borderRadius: '100px', fontSize: '11px', fontWeight: 700, letterSpacing: '1px', background: r.is_deepfake ? 'rgba(122,46,50,0.15)' : 'rgba(123,157,174,0.15)', color: r.is_deepfake ? '#A63A3F' : '#C6A75E' }}>
                        {r.prediction}
                      </div>
                      <div style={{ fontWeight: 600, width: '60px', textAlign: 'right' }}>
                        {r.confidence.toFixed(1)}%
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <Footer />
    </div>
  );
};

export default BatchUploadPage;
