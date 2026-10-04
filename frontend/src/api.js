const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  let data = null
  const text = await res.text()
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = text
    }
  }

  if (!res.ok) {
    const detail =
      (data && data.detail) ||
      (typeof data === 'string' ? data : null) ||
      `HTTP ${res.status}`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }

  return data
}

export const MODES = {
  CREATIVE: 'Kreativní parťák',
  VAULT: 'Striktní Trezor (NotebookLM)',
}

export async function sendChat({ userId, message, mode, model }) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      user_id: userId,
      message,
      mode,
      model,
    }),
  })
}

export async function fetchMemories(userId) {
  return request(`/api/memories?user_id=${encodeURIComponent(userId)}`)
}

export async function deleteMemory(memoryId, userId = '') {
  const q = userId ? `?user_id=${encodeURIComponent(userId)}` : ''
  return request(`/api/memories/${encodeURIComponent(memoryId)}${q}`, {
    method: 'DELETE',
  })
}

export async function downloadExport(userId) {
  const res = await fetch(
    `${API_BASE}/api/export?user_id=${encodeURIComponent(userId)}`,
  )
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${userId}_zrcadlo_memory.json`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
