import { useEffect, useRef, useState } from 'react'
import { sendChat } from '../api'

export default function ChatTab({
  userId,
  mode,
  modelName,
  messages,
  setMessages,
}) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSubmit(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    const uid = userId.trim().toLowerCase() || 'lukas'
    setInput('')
    setError('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const data = await sendChat({
        userId: uid,
        message: text,
        mode,
        model: modelName,
      })
      setMessages((prev) => {
        const next = [...prev]
        if (data.welcome && next.length === 1) {
          next.unshift({ role: 'assistant', content: data.welcome })
        }
        next.push({ role: 'assistant', content: data.reply })
        return next
      })
    } catch (err) {
      setError(err.message || 'Odeslání selhalo.')
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: err.message || 'Něco se pokazilo.',
          isError: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2">
        {messages.length === 0 && !loading && (
          <p className="rounded-xl border border-dashed border-line bg-panel/60 px-4 py-8 text-center text-sm text-muted">
            Napiš první zprávu — Zrcadlo načte paměť a naváže na ni.
          </p>
        )}

        {messages.map((msg, i) => (
          <div
            key={`${msg.role}-${i}`}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-accent text-white'
                  : msg.isError
                    ? 'border border-amber-200 bg-amber-50 text-warn'
                    : 'border border-line bg-surface text-ink shadow-sm'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-line bg-surface px-4 py-2.5 text-sm text-muted shadow-sm">
              Zrcadlo přemýšlí…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && (
        <p className="mb-2 text-xs text-warn" role="alert">
          {error}
        </p>
      )}

      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-line pt-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          placeholder="Napiš zprávu pro Zrcadlo…"
          className="min-w-0 flex-1 rounded-xl border border-line bg-panel px-3 py-2.5 text-sm outline-none ring-accent/30 focus:ring-2 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-accent-deep disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? '…' : 'Odeslat'}
        </button>
      </form>
    </div>
  )
}
