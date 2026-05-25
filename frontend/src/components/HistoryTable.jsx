import React, { useState, useEffect, useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { getHistory, deleteAnalysis, clearHistory, exportCSV } from '../utils/storage';

const HistoryTable = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const location = useLocation();
  
  // Filters
  const [verdictFilter, setVerdictFilter] = useState('All');
  const [sourceFilter, setSourceFilter] = useState('All');
  const [minConfidence, setMinConfidence] = useState(0);
  const [dateFilter, setDateFilter] = useState('All time');
  const [search, setSearch] = useState('');
  
  // Sort
  const [sortCol, setSortCol] = useState('Date'); // 'Date', 'Confidence', 'Verdict'
  const [sortDesc, setSortDesc] = useState(true);
  
  // Pagination
  const [page, setPage] = useState(1);
  const rowsPerPage = 20;

  const loadData = () => {
    setLoading(true);
    // simulate slight read delay for skeleton
    setTimeout(() => {
      setHistory(getHistory());
      setLoading(false);
    }, 300);
  };

  useEffect(() => {
    loadData();
    window.addEventListener('va_history_updated', loadData);
    return () => window.removeEventListener('va_history_updated', loadData);
  }, []);

  useEffect(() => {
    if (location.state && location.state.globalSearch !== undefined) {
      setSearch(location.state.globalSearch);
      // Clear the state so a page refresh doesn't keep forcing the search
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const filteredData = useMemo(() => {
    let data = [...history];
    
    // Search
    if (search) {
      const s = search.toLowerCase();
      data = data.filter(item => item.filename.toLowerCase().includes(s));
    }
    
    // Verdict Filter
    if (verdictFilter !== 'All') {
      data = data.filter(item => item.prediction === verdictFilter);
    }
    
    // Source Filter
    if (sourceFilter !== 'All') {
      const map = { 'Upload': 'upload', 'URL': 'url', 'Live Mic': 'mic' };
      data = data.filter(item => item.source === map[sourceFilter]);
    }
    
    // Confidence Filter
    if (minConfidence > 0) {
      data = data.filter(item => item.confidence >= minConfidence);
    }
    
    // Date Filter
    if (dateFilter !== 'All time') {
      const now = new Date();
      const cutoff = new Date();
      if (dateFilter === 'Today') {
        cutoff.setHours(0,0,0,0);
      } else if (dateFilter === 'Last 7 days') {
        cutoff.setDate(now.getDate() - 7);
      } else if (dateFilter === 'Last 30 days') {
        cutoff.setDate(now.getDate() - 30);
      }
      data = data.filter(item => new Date(item.timestamp) >= cutoff);
    }
    
    // Sort
    data.sort((a, b) => {
      let valA, valB;
      if (sortCol === 'Date') {
        valA = new Date(a.timestamp).getTime();
        valB = new Date(b.timestamp).getTime();
      } else if (sortCol === 'Confidence') {
        valA = a.confidence;
        valB = b.confidence;
      } else if (sortCol === 'Verdict') {
        valA = a.prediction;
        valB = b.prediction;
      }
      
      if (valA < valB) return sortDesc ? 1 : -1;
      if (valA > valB) return sortDesc ? -1 : 1;
      return 0;
    });
    
    return data;
  }, [history, search, verdictFilter, sourceFilter, minConfidence, dateFilter, sortCol, sortDesc]);

  // Pagination
  const totalPages = Math.ceil(filteredData.length / rowsPerPage) || 1;
  const currentData = filteredData.slice((page - 1) * rowsPerPage, page * rowsPerPage);

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [totalPages, page]);

  const handleClear = () => {
    if (window.confirm("Are you sure you want to clear all history? This cannot be undone.")) {
      clearHistory();
    }
  };

  const getSourceIcon = (source) => {
    if (source === 'url') return <i className="ti ti-link" title="URL"></i>;
    if (source === 'mic') return <i className="ti ti-microphone" title="Live Mic"></i>;
    return <i className="ti ti-file-upload" title="Upload"></i>;
  };

  const toggleSort = (col) => {
    if (sortCol === col) {
      setSortDesc(!sortDesc);
    } else {
      setSortCol(col);
      setSortDesc(true);
    }
  };

  const SortIcon = ({ col }) => {
    if (sortCol !== col) return <i className="ti ti-arrows-sort" style={{ opacity: 0.3, marginLeft: 4 }}></i>;
    return <i className={`ti ti-sort-${sortDesc ? 'descending' : 'ascending'}`} style={{ color: '#00d4c8', marginLeft: 4 }}></i>;
  };

  return (
    <div style={{ background: '#0f2229', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '16px', padding: '24px', backdropFilter: 'blur(16px)', width: '100%', marginBottom: '40px' }}>
      
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ fontSize: '18px', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <i className="ti ti-history"></i> ANALYSIS HISTORY
        </h2>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button onClick={() => exportCSV(filteredData)} style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '8px 16px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <i className="ti ti-download"></i> Export CSV
          </button>
          <button onClick={handleClear} style={{ background: 'rgba(232,82,30,0.1)', border: '1px solid rgba(232,82,30,0.2)', color: '#e8521e', padding: '8px 16px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <i className="ti ti-trash"></i> Clear All
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', marginBottom: '24px', background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '12px' }}>
        
        {/* Search */}
        <div style={{ flex: '1 1 200px' }}>
          <div style={{ fontSize: '10px', color: '#7ea8a4', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '1px' }}>Search</div>
          <div style={{ position: 'relative' }}>
            <i className="ti ti-search" style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#7ea8a4' }}></i>
            <input 
              type="text" 
              placeholder="Search filename or URL..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ width: '100%', padding: '8px 12px 8px 36px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', color: 'white', fontSize: '13px', outline: 'none' }}
            />
          </div>
        </div>

        {/* Verdict */}
        <div>
          <div style={{ fontSize: '10px', color: '#7ea8a4', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '1px' }}>Verdict</div>
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', borderRadius: '6px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
            {['All', 'FAKE', 'REAL'].map(v => (
              <button 
                key={v}
                onClick={() => setVerdictFilter(v)}
                style={{ 
                  padding: '8px 16px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                  background: verdictFilter === v ? (v === 'FAKE' ? '#e8521e' : v === 'REAL' ? '#00d4c8' : 'rgba(255,255,255,0.1)') : 'transparent',
                  color: verdictFilter === v ? '#000' : '#7ea8a4'
                }}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* Source */}
        <div>
          <div style={{ fontSize: '10px', color: '#7ea8a4', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '1px' }}>Source</div>
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', borderRadius: '6px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
            {['All', 'Upload', 'URL', 'Live Mic'].map(s => (
              <button 
                key={s}
                onClick={() => setSourceFilter(s)}
                style={{ 
                  padding: '8px 16px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                  background: sourceFilter === s ? 'rgba(255,255,255,0.1)' : 'transparent',
                  color: sourceFilter === s ? 'white' : '#7ea8a4'
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Date Filter */}
        <div>
          <div style={{ fontSize: '10px', color: '#7ea8a4', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '1px' }}>Timeframe</div>
          <select 
            value={dateFilter} 
            onChange={(e) => setDateFilter(e.target.value)}
            style={{ padding: '8px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', color: 'white', fontSize: '13px', outline: 'none', cursor: 'pointer' }}
          >
            {['All time', 'Today', 'Last 7 days', 'Last 30 days'].map(d => (
              <option key={d} value={d} style={{ background: '#0f2229' }}>{d}</option>
            ))}
          </select>
        </div>

        {/* Confidence Slider */}
        <div style={{ flex: '1 1 200px', minWidth: '200px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#7ea8a4', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '1px' }}>
            <span>Min Confidence</span>
            <span style={{ color: 'white' }}>{minConfidence}%</span>
          </div>
          <input 
            type="range" min="0" max="100" 
            value={minConfidence} onChange={(e) => setMinConfidence(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d4c8' }}
          />
        </div>
      </div>

      <div style={{ fontSize: '12px', color: '#7ea8a4', marginBottom: '16px' }}>
        Showing {filteredData.length} of {history.length} results
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
          <thead>
            <tr style={{ background: 'rgba(0,0,0,0.3)', color: '#7ea8a4', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <th style={{ padding: '12px 16px', fontWeight: 600 }}>#</th>
              <th style={{ padding: '12px 16px', fontWeight: 600 }}>Filename / URL</th>
              <th style={{ padding: '12px 16px', fontWeight: 600, textAlign: 'center' }}>Source</th>
              <th onClick={() => toggleSort('Verdict')} style={{ padding: '12px 16px', fontWeight: 600, cursor: 'pointer', userSelect: 'none' }}>
                Verdict <SortIcon col="Verdict" />
              </th>
              <th onClick={() => toggleSort('Confidence')} style={{ padding: '12px 16px', fontWeight: 600, cursor: 'pointer', userSelect: 'none', width: '180px' }}>
                Confidence <SortIcon col="Confidence" />
              </th>
              <th style={{ padding: '12px 16px', fontWeight: 600 }}>Raw Score</th>
              <th onClick={() => toggleSort('Date')} style={{ padding: '12px 16px', fontWeight: 600, cursor: 'pointer', userSelect: 'none' }}>
                Date & Time <SortIcon col="Date" />
              </th>
              <th style={{ padding: '12px 16px', fontWeight: 600, textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              // Skeleton rows
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={`skel-${i}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td colSpan={8} style={{ padding: '16px' }}>
                    <div style={{ height: '20px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', animation: 'pulse 1.5s infinite' }}></div>
                  </td>
                </tr>
              ))
            ) : currentData.length === 0 ? (
              // Empty State
              <tr>
                <td colSpan={8} style={{ padding: '60px 20px', textAlign: 'center', color: '#7ea8a4' }}>
                  <i className="ti ti-waveform" style={{ fontSize: '48px', opacity: 0.3, marginBottom: '16px', display: 'block' }}></i>
                  <div style={{ fontSize: '16px', color: 'white', marginBottom: '8px' }}>No analyses found.</div>
                  <div>Adjust your filters or upload a voice file to get started.</div>
                </td>
              </tr>
            ) : (
              currentData.map((row, i) => {
                const isFake = row.prediction === 'FAKE';
                const badgeStyle = isFake 
                  ? { background: 'rgba(232,82,30,0.15)', border: '1px solid rgba(232,82,30,0.4)', color: '#e8521e' }
                  : { background: 'rgba(0,212,200,0.15)', border: '1px solid rgba(0,212,200,0.4)', color: '#00d4c8' };
                  
                const displayFile = row.filename.length > 30 ? row.filename.substring(0, 30) + '...' : row.filename;
                
                return (
                  <tr key={row.id} className="history-row" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)', transition: 'background 0.2s' }}>
                    <td style={{ padding: '12px 16px', color: '#7ea8a4' }}>{(page - 1) * rowsPerPage + i + 1}</td>
                    <td style={{ padding: '12px 16px' }} title={row.filename}>{displayFile}</td>
                    <td style={{ padding: '12px 16px', textAlign: 'center', color: '#7ea8a4', fontSize: '16px' }}>{getSourceIcon(row.source)}</td>
                    <td style={{ padding: '12px 16px' }}>
                      <span style={{ padding: '4px 10px', borderRadius: '100px', fontSize: '11px', fontWeight: 700, letterSpacing: '1px', ...badgeStyle }}>
                        {row.prediction}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                          <div style={{ width: `${row.confidence}%`, height: '100%', background: isFake ? '#e8521e' : '#00d4c8', borderRadius: '2px' }}></div>
                        </div>
                        <span style={{ color: isFake ? '#e8521e' : '#00d4c8', fontWeight: 600, width: '45px', textAlign: 'right' }}>
                          {row.confidence.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: '#7ea8a4' }}>{row.raw_score.toFixed(4)}</td>
                    <td style={{ padding: '12px 16px', color: '#7ea8a4', fontSize: '12px' }}>
                      <div style={{ color: 'white' }}>{row.date}</div>
                      <div>{row.time}</div>
                    </td>
                    <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                      <button onClick={() => deleteAnalysis(row.id)} title="Delete" style={{ background: 'transparent', border: 'none', color: '#7ea8a4', cursor: 'pointer', padding: '6px', borderRadius: '6px' }} className="hover-red">
                        <i className="ti ti-trash"></i>
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {!loading && currentData.length > 0 && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
          <div style={{ fontSize: '12px', color: '#7ea8a4' }}>
            Page {page} of {totalPages}
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={() => setPage(p => Math.max(1, p - 1))} 
              disabled={page === 1}
              style={{ padding: '6px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: page === 1 ? 'rgba(255,255,255,0.2)' : 'white', borderRadius: '6px', cursor: page === 1 ? 'default' : 'pointer' }}
            >
              Previous
            </button>
            <button 
              onClick={() => setPage(p => Math.min(totalPages, p + 1))} 
              disabled={page === totalPages}
              style={{ padding: '6px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: page === totalPages ? 'rgba(255,255,255,0.2)' : 'white', borderRadius: '6px', cursor: page === totalPages ? 'default' : 'pointer' }}
            >
              Next
            </button>
          </div>
        </div>
      )}

      <style>{`
        .history-row:hover {
          background: rgba(0, 212, 200, 0.05) !important;
          box-shadow: inset 3px 0 0 #00d4c8;
        }
        .hover-red:hover {
          color: #ff4757 !important;
          background: rgba(255, 71, 87, 0.1) !important;
        }
      `}</style>
    </div>
  );
};

export default HistoryTable;
