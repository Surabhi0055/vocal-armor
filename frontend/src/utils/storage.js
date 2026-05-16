export const saveAnalysis = (result) => {
  const entry = {
    id: Date.now(),
    filename: result.filename || result.source_url || 'live_mic',
    source: result.source_url ? 'url' : 'upload',
    prediction: result.prediction,
    confidence: result.confidence,
    raw_score: result.raw_score,
    is_deepfake: result.is_deepfake,
    timestamp: new Date().toISOString(),
    date: new Date().toLocaleDateString(),
    time: new Date().toLocaleTimeString(),
  };

  const existing = JSON.parse(localStorage.getItem('va_history') || '[]');
  const updated = [entry, ...existing].slice(0, 500); // keep last 500
  localStorage.setItem('va_history', JSON.stringify(updated));
  
  // Dispatch custom event to tell components to re-render
  window.dispatchEvent(new Event('va_history_updated'));
};

export const getHistory = () => {
  return JSON.parse(localStorage.getItem('va_history') || '[]');
};

export const clearHistory = () => {
  localStorage.removeItem('va_history');
  window.dispatchEvent(new Event('va_history_updated'));
};

export const deleteAnalysis = (id) => {
  const existing = getHistory();
  const updated = existing.filter(item => item.id !== id);
  localStorage.setItem('va_history', JSON.stringify(updated));
  window.dispatchEvent(new Event('va_history_updated'));
};

export const exportCSV = (data) => {
  const headers = ['ID', 'Filename', 'Source', 'Verdict', 'Confidence', 'Raw Score', 'Date', 'Time'];
  const rows = data.map(d => [
    d.id, 
    `"${d.filename}"`, // Quote to handle commas in filename
    d.source, 
    d.prediction,
    d.confidence, 
    d.raw_score, 
    d.date, 
    d.time
  ]);
  const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'vocalarmor_history.csv';
  a.click();
};
