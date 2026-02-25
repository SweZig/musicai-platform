import { useState, useEffect, useRef } from 'react'
import { uploadTrack, listTracks, getSimilar, pollJob } from './api.js'

// ── Styles ────────────────────────────────────────────────────────────────
const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080810;
    --surface:  #10101e;
    --surface2: #181828;
    --border:   #ffffff12;
    --amber:    #f59e0b;
    --amber-dim:#f59e0b55;
    --cyan:     #06b6d4;
    --cyan-dim: #06b6d422;
    --red:      #ef4444;
    --text:     #e8e8f0;
    --muted:    #6b6b8a;
    --font-head:'Syne', sans-serif;
    --font-mono:'JetBrains Mono', monospace;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); }
  body { font-family: var(--font-mono); font-size: 13px; line-height: 1.6; }

  #root { min-height: 100vh; display: flex; flex-direction: column; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* Layout */
  .app { display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }
  
  /* Header */
  .header {
    grid-column: 1 / -1;
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--surface);
  }
  .logo {
    font-family: var(--font-head);
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--amber);
  }
  .logo span { color: var(--text); }
  .header-sub {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-left: auto;
  }

  /* Sidebar */
  .sidebar {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    background: var(--surface);
    overflow: hidden;
  }
  .sidebar-section {
    padding: 20px;
    border-bottom: 1px solid var(--border);
  }
  .section-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* Upload zone */
  .upload-zone {
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 28px 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }
  .upload-zone::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--amber-dim);
    opacity: 0;
    transition: opacity 0.2s;
  }
  .upload-zone:hover::before, .upload-zone.drag-over::before { opacity: 1; }
  .upload-zone:hover { border-color: var(--amber); }
  .upload-zone.drag-over { border-color: var(--amber); border-style: solid; }
  .upload-icon {
    font-size: 28px;
    margin-bottom: 8px;
    display: block;
  }
  .upload-text { color: var(--muted); font-size: 11px; }
  .upload-text strong { color: var(--amber); }
  input[type=file] { display: none; }

  /* Progress */
  .progress-bar {
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    margin-top: 12px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--amber);
    border-radius: 1px;
    transition: width 0.3s;
    animation: pulse-bar 1.5s ease-in-out infinite;
  }
  @keyframes pulse-bar {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  /* Track list */
  .track-list { overflow-y: auto; flex: 1; }
  .track-item {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.15s;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .track-item:hover { background: var(--surface2); }
  .track-item.active { background: var(--surface2); border-left: 2px solid var(--amber); }
  .track-name {
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track-meta { display: flex; gap: 8px; align-items: center; }
  .genre-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 2px;
    background: var(--cyan-dim);
    color: var(--cyan);
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .status-dot.analyzed { background: #22c55e; }
  .status-dot.processing { background: var(--amber); animation: blink 1s infinite; }
  .status-dot.error { background: var(--red); }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* Main panel */
  .main { overflow-y: auto; background: var(--bg); }
  .empty-state {
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    gap: 12px;
  }
  .empty-icon { font-size: 48px; opacity: 0.3; }
  .empty-text { font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }

  /* Track detail */
  .detail { padding: 32px; max-width: 900px; }
  .detail-header {
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .detail-title {
    font-family: var(--font-head);
    font-size: 28px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
    margin-bottom: 8px;
    word-break: break-all;
  }
  .detail-subtitle { color: var(--muted); font-size: 11px; letter-spacing: 1px; }

  /* Stat grid */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 24px;
  }
  .stat-cell {
    background: var(--surface);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .stat-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
  }
  .stat-value {
    font-size: 22px;
    font-weight: 500;
    color: var(--amber);
    letter-spacing: -0.5px;
  }
  .stat-value.small { font-size: 16px; }
  .stat-unit { font-size: 11px; color: var(--muted); }

  /* Section heading */
  .section-heading {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    margin-top: 28px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Genre scores */
  .genre-bars { display: flex; flex-direction: column; gap: 8px; }
  .genre-row { display: grid; grid-template-columns: 100px 1fr 52px; align-items: center; gap: 10px; }
  .genre-name { font-size: 11px; color: var(--muted); text-align: right; }
  .bar-track {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--cyan);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .bar-fill.top { background: var(--amber); }
  .bar-pct { font-size: 11px; color: var(--muted); text-align: right; }

  /* Similar tracks */
  .similar-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px;
  }
  .similar-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .similar-card:hover {
    border-color: var(--amber);
    background: var(--surface2);
  }
  .similar-score {
    font-size: 22px;
    font-weight: 500;
    color: var(--amber);
    letter-spacing: -0.5px;
    margin-bottom: 4px;
  }
  .similar-title {
    font-size: 11px;
    color: var(--text);
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .similar-genre { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

  /* Structure segments */
  .segments-row {
    display: flex;
    height: 32px;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .segment {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    transition: opacity 0.2s;
    border-right: 1px solid var(--border);
  }
  .segment:last-child { border-right: none; }
  .segment.intro  { background: #1a1a2e; }
  .segment.verse  { background: #16213e; }
  .segment.chorus { background: #0f3460; color: var(--cyan); }
  .segment.outro  { background: #1a1a2e; }

  /* Error */
  .error-msg {
    color: var(--red);
    font-size: 11px;
    padding: 10px 12px;
    border: 1px solid #ef444433;
    border-radius: 4px;
    background: #ef444411;
    margin-top: 10px;
  }

  /* Spinner */
  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--amber);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Fade in */
  .fade-in { animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  /* Confidence badge */
  .confidence {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }
  .confidence-val { color: var(--text); }

  .model-tag {
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 2px;
    background: var(--surface2);
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid var(--border);
  }
  .model-tag.rf { color: var(--amber); border-color: var(--amber-dim); }
`

// ── Helpers ───────────────────────────────────────────────────────────────
function fmt(v, dec = 1) {
  if (v == null) return '—'
  return typeof v === 'number' ? v.toFixed(dec) : v
}

function fmtDur(sec) {
  if (!sec) return '—'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function shortName(title = '') {
  return title.replace(/\.[^/.]+$/, '')
}

// ── Components ────────────────────────────────────────────────────────────

function GenreScores({ scores, topGenre }) {
  if (!scores) return null
  const entries = Object.entries(scores)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => b - a)
  const max = entries[0]?.[1] || 1

  return (
    <div className="genre-bars">
      {entries.map(([genre, score]) => (
        <div className="genre-row" key={genre}>
          <div className="genre-name">{genre}</div>
          <div className="bar-track">
            <div
              className={`bar-fill ${genre === topGenre ? 'top' : ''}`}
              style={{ width: `${(score / max) * 100}%` }}
            />
          </div>
          <div className="bar-pct">{(score * 100).toFixed(0)}%</div>
        </div>
      ))}
    </div>
  )
}

function Segments({ segments }) {
  if (!segments?.length) return null
  const total = segments.reduce((s, seg) => s + seg.duration, 0)
  return (
    <div className="segments-row">
      {segments.map((seg, i) => (
        <div
          key={i}
          className={`segment ${seg.label}`}
          style={{ flex: seg.duration / total }}
          title={`${seg.label} ${fmtDur(seg.start)}–${fmtDur(seg.end)}`}
        >
          {seg.label}
        </div>
      ))}
    </div>
  )
}

function SimilarCard({ track, onClick }) {
  return (
    <div className="similar-card" onClick={() => onClick(track.track_id)}>
      <div className="similar-score">{(track.similarity_score * 100).toFixed(1)}%</div>
      <div className="similar-title" title={track.title}>{shortName(track.title)}</div>
      <div className="similar-genre">{track.genre || 'unknown'}</div>
    </div>
  )
}

function TrackDetail({ track, onSelectTrack }) {
  const [similar, setSimilar] = useState(null)
  const [loadingSimilar, setLoadingSimilar] = useState(false)

  useEffect(() => {
    if (track?.status !== 'analyzed') return
    setLoadingSimilar(true)
    getSimilar(track.id, 6)
      .then(setSimilar)
      .catch(() => setSimilar([]))
      .finally(() => setLoadingSimilar(false))
  }, [track?.id])

  if (!track) return null

  const f = track.features || {}
  const c = track.classification || {}

  return (
    <div className="detail fade-in">
      <div className="detail-header">
        <div className="detail-title">{shortName(track.title || track.original_filename)}</div>
        <div className="detail-subtitle">
          {track.original_filename} &nbsp;·&nbsp; {fmtDur(track.duration_sec)} &nbsp;·&nbsp;
          <span style={{ color: track.status === 'analyzed' ? '#22c55e' : track.status === 'error' ? 'var(--red)' : 'var(--amber)' }}>
            {track.status}
          </span>
        </div>
      </div>

      {/* Core stats */}
      <div className="stat-grid">
        <div className="stat-cell">
          <div className="stat-label">BPM</div>
          <div className="stat-value">{fmt(f.bpm, 1)}</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Key</div>
          <div className="stat-value small">{f.key || '—'} {f.scale || ''}</div>
          <div className="stat-unit">{f.camelot || ''}</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Energy</div>
          <div className="stat-value">{f.energy != null ? (f.energy * 100).toFixed(1) : '—'}</div>
          <div className="stat-unit">rms ×100</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Loudness</div>
          <div className="stat-value small">{fmt(f.loudness_lufs, 1)}</div>
          <div className="stat-unit">LUFS</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Dance</div>
          <div className="stat-value">{f.danceability != null ? (f.danceability * 100).toFixed(0) : '—'}</div>
          <div className="stat-unit">%</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Duration</div>
          <div className="stat-value small">{fmtDur(track.duration_sec)}</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Groove</div>
          <div className="stat-value small">{f.groove_feel || '—'}</div>
          <div className="stat-unit">swing {fmt(f.swing_ratio, 2)}</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Chord</div>
          <div className="stat-value small" style={{ fontSize: 13 }}>{f.chord_progression || '—'}</div>
        </div>
      </div>

      {/* Genre */}
      {c.genre && (
        <>
          <div className="section-heading">Genreklassificering</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: 32, fontWeight: 800, color: 'var(--amber)' }}>
              {c.genre}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div className="confidence">
                Confidence: <span className="confidence-val">{c.confidence != null ? (c.confidence * 100).toFixed(1) + '%' : '—'}</span>
              </div>
              <span className={`model-tag ${c.model_version === 'random_forest' ? 'rf' : ''}`}>
                {c.model_version || 'heuristic'}
              </span>
            </div>
          </div>
          <GenreScores scores={c.genre_scores} topGenre={c.genre} />
        </>
      )}

      {/* Structure */}
      {f.segments?.length > 0 && (
        <>
          <div className="section-heading">Låtstruktur</div>
          <Segments segments={f.segments} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
            {f.section_count} sektioner &nbsp;·&nbsp; {f.beat_count} beats &nbsp;·&nbsp; regularitet {fmt(f.beat_regularity, 2)}
          </div>
        </>
      )}

      {/* Similar */}
      <div className="section-heading">Liknande låtar</div>
      {loadingSimilar ? (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--muted)', fontSize: 11 }}>
          <div className="spinner" /> Söker liknande...
        </div>
      ) : similar?.length > 0 ? (
        <div className="similar-grid">
          {similar.map(t => (
            <SimilarCard key={t.track_id} track={t} onClick={onSelectTrack} />
          ))}
        </div>
      ) : (
        <div style={{ color: 'var(--muted)', fontSize: 11 }}>
          Inga liknande låtar hittades — ladda upp fler spår.
        </div>
      )}

      {/* Spectral details */}
      <div className="section-heading">Spektrala features</div>
      <div className="stat-grid">
        <div className="stat-cell">
          <div className="stat-label">Centroid</div>
          <div className="stat-value small">{fmt(f.spectral_centroid_mean, 0)}</div>
          <div className="stat-unit">Hz</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Rolloff</div>
          <div className="stat-value small">{fmt(f.spectral_rolloff_mean, 0)}</div>
          <div className="stat-unit">Hz</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Bandwidth</div>
          <div className="stat-value small">{fmt(f.spectral_bandwidth_mean, 0)}</div>
          <div className="stat-unit">Hz</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">ZCR</div>
          <div className="stat-value small">{fmt(f.zero_crossing_rate_mean, 4)}</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Dyn Range</div>
          <div className="stat-value small">{fmt(f.dynamic_range, 1)}</div>
          <div className="stat-unit">dB</div>
        </div>
        <div className="stat-cell">
          <div className="stat-label">Beat Str</div>
          <div className="stat-value small">{fmt(f.beat_strength, 2)}</div>
        </div>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [tracks, setTracks] = useState([])
  const [selected, setSelected] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef()

  // Load track list on mount
  useEffect(() => {
    listTracks().then(setTracks).catch(() => {})
  }, [])

  // Refresh track in list when analysis completes
  function updateTrack(t) {
    setTracks(prev => {
      const idx = prev.findIndex(x => x.id === t.id)
      if (idx === -1) return [t, ...prev]
      const next = [...prev]
      next[idx] = t
      return next
    })
    if (selected?.id === t.id) setSelected(t)
  }

  async function handleFiles(files) {
    const file = files[0]
    if (!file) return
    setError('')
    setUploading(true)
    setUploadStatus('Laddar upp...')
    try {
      const res = await uploadTrack(file)
      setUploadStatus('Analyserar...')
      const done = await pollJob(res.track_id, track => {
        updateTrack(track)
        setSelected(track)
      })
      updateTrack(done)
      setSelected(done)
      setUploadStatus('')
    } catch (e) {
      setError(e.message)
      setUploadStatus('')
    } finally {
      setUploading(false)
    }
  }

  async function selectTrack(id) {
    const t = tracks.find(x => x.id === id) || null
    setSelected(t)
    // If from similar cards, track might not be in list yet — fetch it
    if (!t) {
      try {
        const { getTrack } = await import('./api.js')
        const full = await getTrack(id)
        setSelected(full)
        updateTrack(full)
      } catch {}
    }
  }

  return (
    <>
      <style>{css}</style>
      <div className="header">
        <div className="logo">Music<span>AI</span></div>
        <div className="header-sub">Audio Intelligence Platform</div>
      </div>
      <div className="app">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="section-label">Upload</div>
            <div
              className={`upload-zone${dragOver ? ' drag-over' : ''}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
            >
              <span className="upload-icon">🎵</span>
              <div className="upload-text">
                <strong>Klicka eller dra</strong> hit en ljudfil<br />
                WAV · MP3 · FLAC · AIFF · OGG
              </div>
              <input
                ref={fileRef}
                type="file"
                accept=".wav,.mp3,.flac,.aiff,.aif,.ogg"
                onChange={e => handleFiles(e.target.files)}
              />
            </div>
            {uploading && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10, fontSize: 11, color: 'var(--muted)' }}>
                  <div className="spinner" />
                  {uploadStatus}
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: '60%' }} />
                </div>
              </div>
            )}
            {error && <div className="error-msg">{error}</div>}
          </div>

          <div className="sidebar-section" style={{ padding: '12px 20px 8px' }}>
            <div className="section-label">Bibliotek — {tracks.length} spår</div>
          </div>

          <div className="track-list">
            {tracks.map(t => (
              <div
                key={t.id}
                className={`track-item${selected?.id === t.id ? ' active' : ''}`}
                onClick={() => setSelected(t)}
              >
                <div className="track-name" title={t.original_filename}>
                  {shortName(t.title || t.original_filename)}
                </div>
                <div className="track-meta">
                  <div className={`status-dot ${t.status}`} />
                  {t.classification?.genre && (
                    <span className="genre-badge">{t.classification.genre}</span>
                  )}
                  <span style={{ color: 'var(--muted)', fontSize: 10, marginLeft: 'auto' }}>
                    {fmtDur(t.duration_sec)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Main */}
        <main className="main">
          {selected ? (
            <TrackDetail track={selected} onSelectTrack={selectTrack} />
          ) : (
            <div className="empty-state">
              <div className="empty-icon">◎</div>
              <div className="empty-text">Välj ett spår eller ladda upp en ny fil</div>
            </div>
          )}
        </main>
      </div>
    </>
  )
}
