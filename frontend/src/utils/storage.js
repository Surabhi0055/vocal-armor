import { useAuthStore } from '../store/authStore';

// ── Key helpers (scoped per user) ──────────────────────────────────────────
const historyKey  = (uid) => `va_history_${uid}`;
const prefsKey    = (uid) => `va_prefs_${uid}`;

/** Get the current user's UID (falls back to 'guest' for unauthenticated use) */
const getUid = () => {
  const { user } = useAuthStore.getState();
  return user?.id || user?.email || 'guest';
};

// ── History ────────────────────────────────────────────────────────────────
export const saveAnalysis = (result) => {
  const uid = getUid();
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

  const existing = JSON.parse(localStorage.getItem(historyKey(uid)) || '[]');
  const updated  = [entry, ...existing].slice(0, 500);
  localStorage.setItem(historyKey(uid), JSON.stringify(updated));

  window.dispatchEvent(new Event('va_history_updated'));
};

export const getHistory = () => {
  const uid = getUid();
  return JSON.parse(localStorage.getItem(historyKey(uid)) || '[]');
};

export const clearHistory = () => {
  const uid = getUid();
  localStorage.removeItem(historyKey(uid));
  window.dispatchEvent(new Event('va_history_updated'));
};

export const deleteAnalysis = (id) => {
  const uid     = getUid();
  const existing = getHistory();
  const updated  = existing.filter(item => item.id !== id);
  localStorage.setItem(historyKey(uid), JSON.stringify(updated));
  window.dispatchEvent(new Event('va_history_updated'));
};

// ── User Preferences ───────────────────────────────────────────────────────
export const savePrefs = (prefs) => {
  const uid = getUid();
  localStorage.setItem(prefsKey(uid), JSON.stringify(prefs));
};

export const getPrefs = () => {
  const uid = getUid();
  const defaults = { strictFilter: true, autoSave: true, emailAlerts: false, phone: '' };
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(prefsKey(uid)) || '{}') };
  } catch {
    return defaults;
  }
};

// ── CSV Export ─────────────────────────────────────────────────────────────
export const exportCSV = (data) => {
  const headers = ['ID', 'Filename', 'Source', 'Verdict', 'Confidence', 'Raw Score', 'Date', 'Time'];
  const rows    = data.map(d => [
    d.id,
    `"${d.filename}"`,
    d.source,
    d.prediction,
    d.confidence,
    d.raw_score,
    d.date,
    d.time
  ]);
  const csv  = [headers, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'vocalarmor_history.csv';
  a.click();
};
