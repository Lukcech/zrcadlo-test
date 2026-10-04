import { useEffect, useRef, useState } from 'react'
import { MODES } from './api'
import Sidebar from './components/Sidebar'
import ChatTab from './components/ChatTab'
import MemoryTab from './components/MemoryTab'

const TABS = [
  { id: 'chat', label: '💬 Chat' },
  { id: 'memory', label: '🧠 Správa paměti' },
]

const CHAT_STORAGE_KEY = 'chatMessages'
const DEFAULT_MODEL = 'gemini-3.5-flash-lite'

function loadMessages() {
  try {
    const saved = sessionStorage.getItem(CHAT_STORAGE_KEY)
    return saved ? JSON.parse(saved) : []
  } catch {
    return []
  }
}

export default function App() {
  const [userId, setUserId] = useState(
    () => localStorage.getItem('userId') || 'lukas',
  )
  const [mode, setMode] = useState(MODES.CREATIVE)
  const [modelName, setModelName] = useState(DEFAULT_MODEL)
  const [tab, setTab] = useState('chat')
  const [messages, setMessages] = useState(loadMessages)
  const prevUserRef = useRef(userId)

  useEffect(() => {
    sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    if (prevUserRef.current !== userId) {
      setMessages([])
      sessionStorage.removeItem(CHAT_STORAGE_KEY)
      prevUserRef.current = userId
    }
  }, [userId])

  function handleUserIdChange(value) {
    setUserId(value)
    localStorage.setItem('userId', value)
  }

  return (
    <div className="flex h-full min-h-0 flex-col md:flex-row">
      <Sidebar
        userId={userId}
        onUserIdChange={handleUserIdChange}
        mode={mode}
        onModeChange={setMode}
        modelName={modelName}
        onModelChange={setModelName}
      />

      <main className="flex min-h-0 flex-1 flex-col p-4 md:p-6">
        <header className="mb-4">
          <h1 className="font-display text-2xl font-semibold tracking-tight text-ink md:text-3xl">
            Konverzace
          </h1>
          <p className="mt-1 text-sm text-muted">
            Uživatel <span className="font-medium text-ink">{userId || 'lukas'}</span>
            {' · '}
            {mode === MODES.VAULT ? 'Striktní Trezor' : 'Kreativní parťák'}
            {' · '}
            <span className="font-medium text-ink">{modelName}</span>
          </p>
        </header>

        <div className="mb-4 flex gap-1 rounded-xl border border-line bg-panel p-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold transition ${
                tab === t.id
                  ? 'bg-surface text-ink shadow-sm'
                  : 'text-muted hover:text-ink'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <section className="min-h-0 flex-1 rounded-2xl border border-line bg-surface/80 p-4 shadow-sm backdrop-blur-sm">
          {tab === 'chat' ? (
            <ChatTab
              userId={userId}
              mode={mode}
              modelName={modelName}
              messages={messages}
              setMessages={setMessages}
            />
          ) : (
            <MemoryTab userId={userId} active={tab === 'memory'} />
          )}
        </section>
      </main>
    </div>
  )
}
