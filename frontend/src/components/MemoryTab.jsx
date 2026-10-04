import { useCallback, useEffect, useState } from 'react'
import { deleteMemory, fetchMemories } from '../api'

export default function MemoryTab({ userId, active }) {
  const [memories, setMemories] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [deletingId, setDeletingId] = useState(null)

  const load = useCallback(async () => {
    const uid = userId.trim().toLowerCase() || 'lukas'
    setLoading(true)
    setError('')
    try {
      const data = await fetchMemories(uid)
      setMemories(data.memories || [])
    } catch (err) {
      setError(err.message || 'Načtení paměti selhalo.')
      setMemories([])
    } finally {
      setLoading(false)
    }
  }, [userId])

  useEffect(() => {
    if (active) {
      load()
    }
  }, [active, load])

  async function handleDelete(id) {
    if (!window.confirm('Opravdu smazat tuto vzpomínku?')) return
    setDeletingId(id)
    setError('')
    try {
      await deleteMemory(id, userId.trim().toLowerCase() || 'lukas')
      setMemories((prev) => prev.filter((m) => m.id !== id))
    } catch (err) {
      setError(err.message || 'Smazání selhalo.')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-3 flex items-center justify-between gap-2">
        <p className="text-sm text-muted">
          Vzpomínky uživatele{' '}
          <span className="font-semibold text-ink">
            {userId.trim().toLowerCase() || 'lukas'}
          </span>
        </p>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className="rounded-lg border border-line bg-panel px-3 py-1.5 text-xs font-semibold text-ink hover:bg-accent-soft disabled:opacity-50"
        >
          Obnovit
        </button>
      </div>

      {error && (
        <p className="mb-3 text-sm text-warn" role="alert">
          {error}
        </p>
      )}

      {loading && (
        <p className="py-8 text-center text-sm text-muted">Načítám paměť…</p>
      )}

      {!loading && memories.length === 0 && (
        <p className="rounded-xl border border-dashed border-line bg-panel/60 px-4 py-8 text-center text-sm text-muted">
          Pro tohoto uživatele zatím nejsou žádné vzpomínky.
        </p>
      )}

      <ul className="min-h-0 flex-1 space-y-3 overflow-y-auto">
        {memories.map((m) => (
          <li
            key={m.id}
            className="flex items-start justify-between gap-3 rounded-xl border border-line bg-surface p-3 shadow-sm"
          >
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold uppercase tracking-wide text-accent">
                {m.timestamp || 'bez data'}
              </p>
              <p className="mt-1 text-sm leading-relaxed text-ink whitespace-pre-wrap">
                {m.text}
              </p>
            </div>
            <button
              type="button"
              onClick={() => handleDelete(m.id)}
              disabled={deletingId === m.id}
              className="shrink-0 rounded-lg border border-line px-2.5 py-1.5 text-xs font-semibold text-muted transition hover:border-red-300 hover:bg-red-50 hover:text-red-700 disabled:opacity-50"
            >
              {deletingId === m.id ? '…' : 'Smazat'}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
