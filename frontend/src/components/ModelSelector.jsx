import React from 'react';

const ModelSelector = ({ selectedModel, onModelChange }) => {
  return (
    <div style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
      <label style={{ fontSize: '14px', color: '#7ea8a4', fontWeight: 600 }}>Select Engine Model:</label>
      <select 
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        style={{
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(0, 212, 200, 0.3)',
          color: '#00d4c8',
          padding: '8px 16px',
          borderRadius: '8px',
          outline: 'none',
          cursor: 'pointer',
          fontWeight: 600,
          fontFamily: 'inherit'
        }}
      >
        <option value="best" style={{ background: '#0f2229' }}>vocal_armor_best (Standard)</option>
        <option value="v2" style={{ background: '#0f2229' }}>vocal_armor_v2 (Intermediate)</option>
        <option value="v3" style={{ background: '#0f2229' }}>vocal_armor_v3 (Modern Deepfake Voice)</option>
      </select>
    </div>
  );
};

export default ModelSelector;
