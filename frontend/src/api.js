const BASE = 'https://musicai-platform-production.up.railway.app/api/v1'

export async function uploadTrack(file, force = false) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/tracks/upload?force=${force}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getTrack(id) {
  const res = await fetch(`${BASE}/tracks/${id}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function listTracks() {
  const res = await fetch(`${BASE}/tracks/?limit=50`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getSimilar(id, limit = 5) {
  const res = await fetch(`${BASE}/tracks/${id}/similar?limit=${limit}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function pollJob(id, onStatus, maxWait = 120000) {
  const start = Date.now()
  while (Date.now() - start < maxWait) {
    const track = await getTrack(id)
    onStatus(track)
    if (track.status === 'analyzed' || track.status === 'error') return track
    await new Promise(r => setTimeout(r, 2000))
  }
  throw new Error('Timeout — analysen tog för lång tid')
}
