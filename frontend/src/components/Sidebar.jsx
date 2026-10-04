import { MODES, downloadExport } from '../api'

export default function Sidebar({
  userId,
  onUserIdChange,
  mode,
  onModeChange,
  modelName,
  onModelChange,
}) {
  async function handleExport() {
    try {
      await downloadExport(userId.trim().toLowerCase() || 'lukas')
    } catch (err) {
      alert(err.message || 'Export se nezdařil.')
    }
  }

  return (
    <aside className="flex h-full w-full flex-col gap-6 border-r border-line bg-surface/90 p-5 backdrop-blur-sm md:w-72 md:shrink-0">
      <div>
        <p className="font-display text-2xl font-semibold tracking-tight text-ink">
          Zrcadlo
        </p>
        <p className="mt-1 text-sm text-muted">Paměť a konverzace</p>
      </div>

      <label className="block space-y-1.5">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted">
          User ID
        </span>
        <input
          value={userId}
          onChange={(e) => onUserIdChange(e.target.value)}
          className="w-full rounded-lg border border-line bg-panel px-3 py-2 text-sm outline-none ring-accent/30 transition focus:ring-2"
          placeholder="lukas"
        />
      </label>

      <fieldset className="space-y-2">
        <legend className="text-xs font-semibold uppercase tracking-wide text-muted">
          Režim odpovědí
        </legend>
        <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-line bg-panel px-3 py-2.5 has-[:checked]:border-accent has-[:checked]:bg-accent-soft">
          <input
            type="radio"
            name="mode"
            className="mt-1 accent-[var(--color-accent)]"
            checked={mode === MODES.CREATIVE}
            onChange={() => onModeChange(MODES.CREATIVE)}
          />
          <span>
            <span className="block text-sm font-medium text-ink">
              Kreativní parťák
            </span>
            <span className="text-xs text-muted">temperature 0.6</span>
          </span>
        </label>
        <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-line bg-panel px-3 py-2.5 has-[:checked]:border-accent has-[:checked]:bg-accent-soft">
          <input
            type="radio"
            name="mode"
            className="mt-1 accent-[var(--color-accent)]"
            checked={mode === MODES.VAULT}
            onChange={() => onModeChange(MODES.VAULT)}
          />
          <span>
            <span className="block text-sm font-medium text-ink">
              Striktní Trezor
            </span>
            <span className="text-xs text-muted">jen fakta z paměti · 0.0</span>
          </span>
        </label>
      </fieldset>

      <label className="block space-y-1.5">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted">
          Gemini model
        </span>
        <select
          value={modelName}
          onChange={(e) => onModelChange(e.target.value)}
          className="w-full rounded-lg border border-line bg-panel px-3 py-2 text-sm outline-none ring-accent/30 transition focus:ring-2"
        >
          <option value="gemini-3.5-flash-lite">
            gemini-3.5-flash-lite — Výchozí
          </option>
          <option value="gemini-3.5-flash">
            gemini-3.5-flash — Záložní
          </option>
          <option value="gemini-flash-latest">
            gemini-flash-latest — Alias
          </option>
        </select>
      </label>

      <button
        type="button"
        onClick={handleExport}
        className="mt-auto rounded-lg bg-accent px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-accent-deep"
      >
        Stáhnout mé vzpomínky (JSON)
      </button>
    </aside>
  )
}

