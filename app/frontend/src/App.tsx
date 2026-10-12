import { useEffect, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { api } from './api'
import type { CatalogResponse, CompareRun, Item, MethodId, Query, SearchResponse, SystemStatus, View } from './types'

const methodInfo: Record<MethodId, { name: string; label: string; description: string }> = {
  bm25: { name: 'BM25', label: 'Keyword', description: 'Matches the words in your query.' },
  e5: { name: 'Multilingual E5', label: 'Semantic', description: 'Finds items with similar meaning.' },
  hybrid: { name: 'Hybrid · RRF', label: 'Combined', description: 'Combines keyword and semantic ranks.' },
  harrier: { name: 'Harrier', label: 'Semantic+', description: 'Newer multilingual embeddings with a task instruction.' },
  full: { name: 'Best system', label: 'LLM + hybrid', description: 'An LLM reads the intent, then hybrid keyword and semantic retrieval.' },
}
const methods: MethodId[] = ['bm25', 'e5', 'hybrid', 'harrier', 'full']
const compareMethods: MethodId[] = ['bm25', 'harrier', 'full']
const methodIcon = (id: MethodId) => id === 'bm25' ? 'search' : id === 'e5' || id === 'harrier' ? 'spark' : 'layers'
const number = new Intl.NumberFormat('en-US')
const priceFormat = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const priceLabel = (price: number) => `Price · ${priceFormat.format(price)}`
const ms = (value: number) => value < 1 ? '< 1' : number.format(Math.round(value))

function Icon({ name, className = '' }: { name: string; className?: string }) {
  const paths: Record<string, ReactNode> = {
    search: <><circle cx="10.8" cy="10.8" r="6.3" /><path d="m16 16 4.5 4.5" /></>,
    compare: <><rect x="3" y="5" width="7" height="14" rx="1.5" /><rect x="14" y="5" width="7" height="14" rx="1.5" /><path d="M6 9h1m-1 3h1m10-3h1m-1 3h1" /></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    arrow: <><path d="M4 12h16m-6-6 6 6-6 6" /></>,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3 2" /></>,
    leaf: <><path d="M5 18C0 10 8 4 20 4c0 12-6 20-14 14Z" /><path d="M4 21 15 10" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2" /></>,
    chevron: <path d="m9 5 7 7-7 7" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    spark: <><path d="m12 3 2.7 6.3L21 12l-6.3 2.7L12 21l-2.7-6.3L3 12l6.3-2.7Z" /></>,
    layers: <><path d="m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6 7a7 7 0 0 1 12-1l2 6M4 12l2 6a7 7 0 0 0 12-1" /></>,
    filter: <><path d="M4 6h16M7 12h10M10 18h4" /></>,
    image: <><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8" cy="9" r="1.5" /><path d="m4 18 6-6 4 4 3-3 4 4" /></>,
    file: <><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6Z" /><path d="M14 3v6h6M8 13h8m-8 4h5" /></>,
  }
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">{paths[name] ?? paths.search}</svg>
}

function FoodArt({ item, large = false }: { item: Item; large?: boolean }) {
  const text = `${item.category} ${item.taxonomy_l0} ${item.name}`.toLocaleLowerCase('pt-BR')
  const kind = /beleza|higiene|saude|saúde|casa|decora|eletro|bebe|pet|jardim|jogos|moda|papelaria/.test(item.taxonomy_l0.toLowerCase()) ? 'package' : /pizza/.test(text) ? 'pizza' : /bebida|suco|café|cafe|refrigerante|água|cerveja/.test(text) ? 'drink' : /doce|sobremesa|bolo|chocolate|açaí|sorvete/.test(text) ? 'sweet' : /burger|hamb[uú]rguer|sandu[ií]che|lanche/.test(text) ? 'burger' : /salada|vegan|vegetar/.test(text) ? 'leaf' : 'bowl'
  return <div className={`food-art art-${kind} ${large ? 'art-large' : ''}`}>
    <span className="art-dot dot-one" /><span className="art-dot dot-two" /><span className="art-line" />
    <svg className="food-drawing" viewBox="0 0 160 130" fill="none" aria-hidden="true">
      {kind === 'pizza' && <><ellipse cx="82" cy="111" rx="45" ry="6" fill="currentColor" opacity=".08" /><path d="m78 24 46 80H33L78 24Z" fill="#f3c680" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" /><path d="M34 104c20-10 67-10 90 0" stroke="#c78142" strokeWidth="9" strokeLinecap="round" /><circle cx="75" cy="53" r="6" fill="#c2674c" /><circle cx="57" cy="79" r="7" fill="#c2674c" /><circle cx="93" cy="79" r="7" fill="#c2674c" /><path d="m83 66 6-4m-17 26 6-5m-8-17-4-4" stroke="#5e8060" strokeWidth="4" strokeLinecap="round" /></>}
      {kind === 'drink' && <><ellipse cx="80" cy="113" rx="30" ry="5" fill="currentColor" opacity=".08" /><path d="M56 48h48l-5 55a9 9 0 0 1-9 8H70a9 9 0 0 1-9-8l-5-55Z" fill="#f7ead7" stroke="currentColor" strokeWidth="2.5" /><path d="M61 77h38l-3 26H64l-3-26Z" fill="#d99363" /><path d="m83 87 6-59 14-6" stroke="currentColor" strokeWidth="3" strokeLinecap="round" /><circle cx="58" cy="45" r="14" fill="#f3c680" stroke="currentColor" strokeWidth="2" /><path d="m58 32 1 26m-13-13 26-1" stroke="currentColor" opacity=".5" strokeWidth="1.5" /><path d="M55 48h49" stroke="currentColor" strokeWidth="2.5" /></>}
      {kind === 'sweet' && <><ellipse cx="80" cy="112" rx="42" ry="5" fill="currentColor" opacity=".08" /><path d="M51 74h57l-6 33H58l-7-33Z" fill="#e3bb9f" stroke="currentColor" strokeWidth="2.5" /><path d="M45 73c0-9 5-15 13-17-2-11 6-22 19-22 16-1 24 9 24 19 10 1 16 8 16 18 0 8-14 8-20 3-7 6-14 5-21 0-8 5-17 6-22 0-5 4-10 3-9-1Z" fill="#f5dfcf" stroke="currentColor" strokeWidth="2.5" /><path d="m64 81 4 20m12-20v20m16-20-4 20" stroke="currentColor" opacity=".35" strokeWidth="2" /><circle cx="80" cy="30" r="7" fill="#b76653" /><path d="m81 23 3-7" stroke="#52785d" strokeWidth="2.5" strokeLinecap="round" /></>}
      {kind === 'burger' && <><ellipse cx="80" cy="114" rx="47" ry="5" fill="currentColor" opacity=".08" /><path d="M37 66c0-25 20-38 43-38s43 13 43 38H37Z" fill="#dfaa67" stroke="currentColor" strokeWidth="2.5" /><path d="m44 74 12 4 12-4 12 4 13-4 12 4 13-4" stroke="#71966c" strokeWidth="7" strokeLinecap="round" /><rect x="38" y="81" width="85" height="13" rx="6" fill="#98634c" stroke="currentColor" strokeWidth="2" /><path d="m47 81 20 15 21-15" fill="#ebc267" /><path d="M39 99h83a13 13 0 0 1-13 13H52a13 13 0 0 1-13-13Z" fill="#dfaa67" stroke="currentColor" strokeWidth="2.5" /><path d="m62 45 3-2m13-3 3 1m15 5 3-2m-18 11 3 1" stroke="#fff4d8" strokeWidth="3" strokeLinecap="round" /></>}
      {kind === 'leaf' && <><ellipse cx="80" cy="111" rx="40" ry="5" fill="currentColor" opacity=".08" /><path d="M80 102C39 108 31 73 44 40c28 2 43 18 38 46" fill="#a5b895" stroke="currentColor" strokeWidth="2.5" /><path d="M81 103c-10-36 3-66 36-79 21 35 8 74-36 79Z" fill="#c2cfb0" stroke="currentColor" strokeWidth="2.5" /><path d="m81 110 19-55m-19 46L54 61m35 20 20-6m-23 12L74 75" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></>}
      {kind === 'package' && <><ellipse cx="80" cy="113" rx="35" ry="5" fill="currentColor" opacity=".08" /><path d="m49 45 31-16 31 16v54l-31 17-31-17V45Z" fill="#e8e1cd" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" /><path d="m49 45 31 17 31-17M80 62v54m-16-78 31 17v20" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" /><path d="m57 84 13 7m-13 0 8 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".55" /></>}
      {kind === 'bowl' && <><ellipse cx="80" cy="113" rx="44" ry="5" fill="currentColor" opacity=".08" /><path d="M35 70h90c-3 25-18 39-45 39S38 95 35 70Z" fill="#e8c4a4" stroke="currentColor" strokeWidth="2.5" /><path d="M40 64c5-18 14-26 22-20 10-21 26-12 26 0 15-9 30 0 32 20" fill="#b9c99c" stroke="currentColor" strokeWidth="2.5" /><path d="m59 47 6 16m22-15-5 17m16-13 6 12" stroke="#69865c" strokeWidth="2.5" strokeLinecap="round" /><path d="m106 58 22-35m-14 40 23-36" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" /><path d="M62 109h36" stroke="currentColor" strokeWidth="3" strokeLinecap="round" /></>}
    </svg>
    <span className="art-caption">{item.taxonomy_l0 || 'From the catalog'}</span>
  </div>
}

function ItemVisual({ item, photos, large = false }: { item: Item; photos: boolean; large?: boolean }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => { setFailed(false) }, [item.document_id, photos])
  return photos && item.image_url && !failed
    ? <div className={`photo-visual ${large ? 'art-large' : ''}`}><img src={item.image_url} alt={item.name} loading="lazy" onError={() => setFailed(true)} /></div>
    : <FoodArt item={item} large={large} />
}

function ItemCard({ item, photos, onOpen, rank, score }: { item: Item; photos: boolean; onOpen: (item: Item) => void; rank?: number; score?: number }) {
  return <button className="item-card" onClick={() => onOpen(item)} aria-label={`View ${item.name}`}>
    <div className="card-visual"><ItemVisual item={item} photos={photos} />{rank && <span className="rank-badge">{String(rank).padStart(2, '0')}</span>}</div>
    <div className="card-body">
      <span className="category-label">{item.category || item.taxonomy_l0 || 'Uncategorized'}</span>
      <h3 lang="pt-BR">{item.name}</h3>
      <p lang="pt-BR">{item.description || 'No description in the source data.'}</p>
      <div className="card-bottom"><strong>{item.price == null ? 'Price unavailable' : priceLabel(item.price)}</strong><Icon name="arrow" /></div>
      {score !== undefined && <span className="score-label">Retrieval score <span>{score.toFixed(4)}</span></span>}
    </div>
  </button>
}

function Timing({ run }: { run: SearchResponse }) {
  return <div className="timing-row">
    <span><Icon name="clock" /><strong>{ms(run.timings_ms.total)} ms</strong> total</span>
    <span>Encoding <b>{ms(run.timings_ms.query_encoding)} ms</b></span>
    <span>Retrieval <b>{ms(run.timings_ms.retrieval)} ms</b></span>
    <span className={run.cache_hit ? 'cache-tag' : 'cache-tag fresh'}><Icon name={run.cache_hit ? 'check' : 'spark'} />{run.cache_hit ? 'Query vector reused' : run.method === 'bm25' ? 'Live keyword search' : 'Fresh query encoding'}</span>
  </div>
}

function App() {
  const [view, setView] = useState<View>('search')
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [statusError, setStatusError] = useState('')
  const [queries, setQueries] = useState<Query[]>([])
  const [query, setQuery] = useState('')
  const [method, setMethod] = useState<MethodId>('bm25')
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [compare, setCompare] = useState<{ query: string; runs: CompareRun[] } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [photos, setPhotos] = useState(false)
  const [cache, setCache] = useState(true)
  const [building, setBuilding] = useState(false)
  const [selected, setSelected] = useState<Item | null>(null)
  const [detailError, setDetailError] = useState('')
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null)
  const [catalogQuery, setCatalogQuery] = useState('')
  const [category, setCategory] = useState('')
  const [offset, setOffset] = useState(0)
  const [catalogBusy, setCatalogBusy] = useState(false)
  const [catalogError, setCatalogError] = useState('')
  const request = useRef(0)
  const dialog = useRef<HTMLDialogElement>(null)

  async function refreshStatus() {
    try { setStatus(await api<SystemStatus>('/status')); setStatusError('') }
    catch (err) { setStatusError((err as Error).message) }
  }

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    Promise.all([
      api<SystemStatus>('/status', { signal: controller.signal }),
      api<{ queries: Query[] }>('/queries', { signal: controller.signal }),
    ]).then(async ([newStatus, data]) => {
      if (!active) return
      setStatus(newStatus); setQueries(data.queries); setStatusError('')
      const example = data.queries.find(q => /pizza/i.test(q.text)) ?? data.queries[0]
      if (example) {
        setQuery(example.text); setBusy(true)
        try {
          const result = await api<SearchResponse>('/search', { method: 'POST', signal: controller.signal, body: JSON.stringify({ query: example.text, method: 'bm25', top_k: 5, use_query_cache: true }) })
          if (active) setResponse(result)
        } catch (err) { if (active) setError((err as Error).message) }
        finally { if (active) setBusy(false) }
      }
    }).catch(err => { if (active) setStatusError((err as Error).message) })
    return () => { active = false; controller.abort() }
  }, [])

  useEffect(() => {
    const interval = setInterval(refreshStatus, status?.semantic.state === 'loading' ? 1500 : 8000)
    return () => clearInterval(interval)
  }, [status?.semantic.state])

  useEffect(() => {
    if (view !== 'dataset') return
    let active = true
    const controller = new AbortController()
    setCatalogBusy(true)
    const timer = setTimeout(() => {
      const params = new URLSearchParams({ query: catalogQuery, category, offset: String(offset), limit: '24' })
      api<CatalogResponse>(`/catalog?${params}`, { signal: controller.signal }).then(data => {
        if (active) { setCatalog(data); setCatalogError('') }
      }).catch(err => { if (active) setCatalogError((err as Error).message) }).finally(() => { if (active) setCatalogBusy(false) })
    }, 200)
    return () => { active = false; clearTimeout(timer); controller.abort() }
  }, [view, catalogQuery, category, offset])

  useEffect(() => {
    if (!selected) return
    dialog.current?.showModal()
    const controller = new AbortController()
    setDetailError('')
    api<Item>(`/items/${encodeURIComponent(selected.document_id)}`, { signal: controller.signal }).then(setSelected).catch(err => {
      if (!controller.signal.aborted) setDetailError((err as Error).message)
    })
    return () => controller.abort()
  }, [selected?.document_id])

  async function search(event?: FormEvent, nextQuery = query) {
    event?.preventDefault()
    if (!nextQuery.trim() || busy) return
    const current = ++request.current
    setBusy(true); setError('')
    try {
      if (view === 'compare') {
        const result = await api<{ query: string; runs: CompareRun[] }>('/compare', { method: 'POST', body: JSON.stringify({ query: nextQuery.trim(), methods: compareMethods, top_k: 5, use_query_cache: cache }) })
        if (current === request.current) setCompare(result)
      } else {
        const result = await api<SearchResponse>('/search', { method: 'POST', body: JSON.stringify({ query: nextQuery.trim(), method, top_k: 5, use_query_cache: cache }) })
        if (current === request.current) setResponse(result)
      }
    } catch (err) { if (current === request.current) setError((err as Error).message) }
    finally { if (current === request.current) setBusy(false) }
  }

  async function buildSemantic() {
    setBuilding(true)
    try { await api('/semantic/build', { method: 'POST' }); await refreshStatus(); setError('') }
    catch (err) { setError((err as Error).message) }
    finally { setBuilding(false) }
  }

  const ready = (id: MethodId) => status?.methods.find(m => m.id === id)?.ready ?? id === 'bm25'
  const examples = [...new Map(['pizza', 'vegan', 'sushi', 'hamb'].map(word => queries.find(q => q.text.toLowerCase().includes(word))).filter((q): q is Query => !!q).concat(queries.slice(0, 6)).map(q => [q.id, q])).values()].slice(0, 4)
  const closeDetail = () => { dialog.current?.close(); setSelected(null) }

  return <div className="app-shell">
    <a href="#main-content" className="skip-link">Skip to content</a>
    <aside className="sidebar">
      <button className="brand" onClick={() => setView('search')} aria-label="Search Studio home"><span className="brand-mark"><i /><i /><i /><i /></span><span>prosus<span className="brand-subtitle">SEARCH STUDIO</span></span></button>
      <div className="nav-label">THE WORKSPACE</div>
      <nav aria-label="Main navigation">
        {(['search', 'compare', 'dataset'] as View[]).map(id => <button key={id} className={`nav-item ${view === id ? 'active' : ''}`} onClick={() => { setView(id); setError('') }} aria-current={view === id ? 'page' : undefined}>
          <Icon name={id === 'dataset' ? 'grid' : id} /><span>{id === 'search' ? 'Search playground' : id === 'compare' ? 'Compare methods' : 'Explore dataset'}</span>{view === id && <span className="nav-dot" />}
        </button>)}
      </nav>
      <div className="sidebar-note"><span className="small-label">A SMALL CATALOG.<br />A BIG SEARCH QUESTION.</span><p>Find the right food,<br />in the right order.</p><div className="note-graphic"><span /><span /><span /></div></div>
      <div className="sidebar-footer"><span className="status-dot" /><span>Local demo</span><Icon name="lock" /><p>Prosus AI · Technical assignment</p></div>
    </aside>

    <div className="workspace">
      <header className="topbar"><span>Prosus AI <span className="slash">/</span> Food discovery</span><div className="topbar-right"><span className="language-pill">PT-BR catalog</span><span className={`connection ${statusError ? 'offline' : ''}`}><i />{statusError ? 'Service unavailable' : status ? 'Connected locally' : 'Connecting…'}</span></div></header>
      <main id="main-content">
        <div className="page-heading"><div><span className="eyebrow">{view === 'search' ? '01 / SEARCH PLAYGROUND' : view === 'compare' ? '02 / METHOD COMPARISON' : '03 / THE DATASET'}</span><h1>{view === 'search' ? <>Good food.<em> Better discovery.</em></> : view === 'compare' ? <>One query.<em> Three perspectives.</em></> : <>Explore the<em> ingredients.</em></>}</h1><p>{view === 'search' ? 'Try a Portuguese query and see what rises to the top.' : view === 'compare' ? 'See how keyword search, dense retrieval, and the full pipeline rank the same catalog.' : 'A closer look at the items behind every search result.'}</p></div><span className="edition">EXPERIMENT 02<span>Local open models</span></span></div>

        <div className="stats-strip" aria-label="Dataset overview">
          {[['items', 'catalog rows', 'grid'], ['unique_items', 'unique items', 'layers'], ['queries', 'example queries', 'search'], ['merchants', 'merchants', 'file']].map(([key, label, icon]) => <div className="stat" key={key}><Icon name={icon} /><div><strong>{status ? number.format(status.counts[key as keyof SystemStatus['counts']]) : '—'}</strong><span>{label}</span></div></div>)}
          <span className="dataset-origin">Portuguese food catalog<br /><b>Original source data</b></span>
        </div>

        {statusError && <div className="error-banner" role="alert"><span>{statusError}</span><button onClick={refreshStatus}>Reconnect <Icon name="refresh" /></button></div>}

        {view !== 'dataset' && <>
          <section className="search-panel" aria-label="Search controls">
            <div className="section-kicker"><span><Icon name="search" />WHAT ARE YOU CRAVING?</span><span>Search in Portuguese</span></div>
            <form onSubmit={search} className="search-form"><div className="query-input"><Icon name="search" /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Try pizza de calabresa…" aria-label="Portuguese search query" maxLength={2000} lang="pt-BR" /></div><button className="primary-button" type="submit" disabled={busy || !query.trim()}>{busy ? <><span className="spinner" /> Searching</> : <>{view === 'compare' ? 'Compare methods' : 'Find food'}<Icon name="arrow" /></>}</button></form>
            <div className="examples"><span>Try an example</span>{examples.map(example => <button key={example.id} disabled={busy} onClick={() => { setQuery(example.text); void search(undefined, example.text) }} lang="pt-BR" title={example.text}><span>{example.text}</span><Icon name="arrow" /></button>)}</div>
            {view === 'search' && <div className="method-options" aria-label="Retrieval method">{methods.map(id => <button key={id} className={`method-option ${method === id ? 'selected' : ''}`} onClick={() => setMethod(id)} disabled={!ready(id) || busy} aria-pressed={method === id}><span className="method-radio">{method === id && <span />}</span><span className="method-copy"><span className="method-title">{methodInfo[id].name}<span>{methodInfo[id].label}</span></span><small>{methodInfo[id].description}</small></span>{!ready(id) && <span className="pending-label">Preparing</span>}</button>)}</div>}
            <div className="search-options"><label className="toggle-label"><input type="checkbox" checked={cache} onChange={e => setCache(e.target.checked)} /><span className="toggle-track" /><span>Reuse cached query vectors</span></label><span>Top 5 · Same catalog for every method</span></div>
          </section>

          {status && status.semantic.state !== 'ready' && <div className="semantic-notice"><div className="notice-icon"><Icon name="spark" /></div><div className="notice-copy"><strong>{status.semantic.state === 'loading' ? 'Semantic search is getting ready' : status.semantic.state === 'error' ? 'Semantic search needs another try' : 'Add a semantic perspective'}</strong><p>{status.semantic.message || 'Prepare local E5 embeddings to unlock semantic and hybrid search.'}</p>{status.semantic.state === 'loading' && <div className="progress-track"><span style={{ width: status.semantic.progress == null ? '25%' : `${Math.max(0, Math.min(100, status.semantic.progress <= 1 ? status.semantic.progress * 100 : status.semantic.progress))}%` }} /></div>}</div>{status.semantic.state !== 'loading' && <button className="secondary-button" onClick={buildSemantic} disabled={building}>{building ? 'Starting…' : status.semantic.state === 'error' ? 'Try again' : 'Prepare semantic search'}<Icon name="arrow" /></button>}</div>}

          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => void search()}>Try again <Icon name="refresh" /></button></div>}

          {view === 'search' && <section className="results-section" aria-label="Search results" aria-busy={busy}>
            <div className="results-heading"><div><span className="eyebrow">THE SHORTLIST</span><h2>{response ? <>Ranked for <span lang="pt-BR">“{response.query}”</span></> : 'Your next discovery starts here'}</h2></div><label className="photo-toggle"><input type="checkbox" checked={photos} onChange={e => setPhotos(e.target.checked)} /><Icon name="image" /> Show photos</label></div>
            {busy ? <div className="product-grid">{Array.from({ length: 5 }, (_, i) => <div className="skeleton-card" key={i}><div /><span /><span /><span /></div>)}</div> : response ? <>
              <div className="result-meta"><span className="method-tag">{methodInfo[response.method].name}</span><span>{response.results.length} results · Ranked across the full catalog</span></div>
              {response.understanding && <p className="interpretation"><Icon name="spark" />Understood as <strong>{response.understanding.interpretation_en}</strong><span lang="pt-BR">{response.understanding.rewrite_pt}</span></p>}
              {response.warnings?.map(warning => <p className="result-warning" key={warning}>{warning}</p>)}
              {response.results.length ? <div className="product-grid">{response.results.map(item => <ItemCard key={item.document_id} item={item} photos={photos} onOpen={setSelected} rank={item.rank} score={item.score} />)}</div> : <div className="empty-state"><Icon name="search" /><h3>No matching items</h3><p>Try a broader food name or a different search method.</p></div>}
              <Timing run={response} />
            </> : <div className="empty-state"><Icon name="search" /><h3>A little curiosity goes a long way.</h3><p>Enter a query above, or choose an example from the dataset.</p></div>}
            <p className="section-footnote">Retrieval scores describe each method’s ranking. They are not relevance labels or probabilities.</p>
          </section>}

          {view === 'compare' && <section className="results-section" aria-label="Method comparison" aria-busy={busy}>
            <div className="results-heading"><div><span className="eyebrow">SIDE BY SIDE</span><h2>{compare ? <>Results for <span lang="pt-BR">“{compare.query}”</span></> : 'Different methods. Same question.'}</h2></div><label className="photo-toggle"><input type="checkbox" checked={photos} onChange={e => setPhotos(e.target.checked)} /><Icon name="image" /> Show photos</label></div>
            <div className="comparison-grid">{compareMethods.map(id => {
              const run = compare?.runs.find(r => r.method === id)
              return <article className="comparison-column" key={id}><header><span className={`method-symbol ${id}`}><Icon name={methodIcon(id)} /></span><div><h3>{methodInfo[id].name}</h3><p>{methodInfo[id].label} retrieval</p></div>{run && 'results' in run && <span className="comparison-ms">{ms(run.timings_ms.total)}<small>ms</small></span>}</header>
                {busy ? Array.from({ length: 5 }, (_, i) => <div className="mini-skeleton" key={i} />) : run && 'results' in run ? <>{run.warnings?.map(warning => <p className="result-warning compare-warning" key={warning}>{warning}</p>)}{run.understanding && <p className="interpretation compact"><Icon name="spark" />{run.understanding.interpretation_en}</p>}<div className="comparison-list">{run.results.map(item => <button className="comparison-item" key={item.document_id} onClick={() => setSelected(item)}><span className="mini-rank">{String(item.rank).padStart(2, '0')}</span><div className="mini-visual"><ItemVisual item={item} photos={photos} /></div><div className="mini-copy"><h4 lang="pt-BR">{item.name}</h4><p lang="pt-BR">{item.category}</p><span>{item.price == null ? '—' : priceLabel(item.price)}</span></div><Icon name="chevron" /></button>)}</div><div className="comparison-footer"><span>Retrieval {ms(run.timings_ms.retrieval)} ms</span><span>{run.cache_hit ? 'Cached query vector' : 'Live query'}</span></div></> : <div className="comparison-empty"><Icon name={ready(id) ? 'search' : 'spark'} /><p>{run && 'error' in run ? run.error : ready(id) ? 'Run a query to see the ranking.' : 'Available after semantic search is prepared.'}</p></div>}
              </article>
            })}</div>
            <p className="section-footnote">Compare the ranked items, not raw scores. Timings show this request only; repeated controlled runs are needed for a benchmark.</p>
          </section>}
        </>}

        {view === 'dataset' && <section className="catalog-section" aria-label="Catalog browser">
          <div className="catalog-toolbar"><div className="catalog-search"><Icon name="search" /><input value={catalogQuery} onChange={e => { setCatalogQuery(e.target.value); setOffset(0) }} placeholder="Search names and descriptions…" aria-label="Filter catalog by text" lang="pt-BR" /></div><div className="category-select"><Icon name="filter" /><select value={category} onChange={e => { setCategory(e.target.value); setOffset(0) }} aria-label="Filter catalog by category"><option value="">All categories</option>{catalog?.categories.map(c => <option key={c.name} value={c.name}>{c.name} ({number.format(c.count)})</option>)}</select></div><label className="photo-toggle"><input type="checkbox" checked={photos} onChange={e => setPhotos(e.target.checked)} /><Icon name="image" /> Show photos</label></div>
          <div className="catalog-count"><span>{catalog ? <><strong>{number.format(catalog.total)}</strong> catalog rows{category ? <> in <b lang="pt-BR">{category}</b></> : ''}</> : 'Loading catalog…'}</span><span>Click an item to see the source details</span></div>
          {catalogError && <div className="error-banner" role="alert">{catalogError}</div>}
          {catalogBusy ? <div className="catalog-grid">{Array.from({ length: 8 }, (_, i) => <div className="skeleton-card" key={i}><div /><span /><span /><span /></div>)}</div> : catalog?.items.length ? <div className="catalog-grid">{catalog.items.map(item => <ItemCard key={item.document_id} item={item} photos={photos} onOpen={setSelected} />)}</div> : !catalogError && <div className="empty-state"><Icon name="grid" /><h3>No items match these filters</h3><p>Try another category or clear the text filter.</p></div>}
          {catalog && catalog.total > 0 && <div className="pagination"><span>Showing {number.format(offset + 1)}–{number.format(Math.min(offset + 24, catalog.total))} of {number.format(catalog.total)}</span><div><button className="secondary-button" disabled={offset === 0 || catalogBusy} onClick={() => { setOffset(Math.max(0, offset - 24)); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Previous</button><button className="secondary-button" disabled={offset + 24 >= catalog.total || catalogBusy} onClick={() => { setOffset(offset + 24); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Next page<Icon name="arrow" /></button></div></div>}
        </section>}
      </main>
      <footer className="workspace-footer"><span>Built for better search conversations.</span><span>5k catalog · Open-source retrieval · Local first</span></footer>
    </div>

    <dialog ref={dialog} className="detail-dialog" onClose={() => setSelected(null)} onClick={e => { if (e.target === dialog.current) closeDetail() }} aria-labelledby="detail-name">
      {selected && <div className="detail-content"><div className="detail-top"><span className="eyebrow">FROM THE SOURCE CATALOG</span><button className="icon-button" onClick={closeDetail} aria-label="Close item details" autoFocus><Icon name="close" /></button></div><ItemVisual item={selected} photos={photos} large /><div className="detail-body"><span className="category-label">{selected.category || selected.taxonomy_l0}</span><h2 id="detail-name" lang="pt-BR">{selected.name}</h2><strong className="detail-price">{selected.price == null ? 'Price unavailable' : priceLabel(selected.price)}</strong><p className="detail-description" lang="pt-BR">{selected.description || 'No description in the source data.'}</p>
        {(selected.vegan || selected.lac_free || selected.organic) && <div className="diet-badges">{selected.vegan && <span><Icon name="leaf" />Vegan</span>}{selected.lac_free && <span>Lactose-free</span>}{selected.organic && <span>Organic</span>}</div>}
        <dl className="detail-facts"><div><dt>Document</dt><dd>{selected.document_id}</dd></div><div><dt>Item ID</dt><dd>{selected.item_id}</dd></div><div><dt>Merchant ID</dt><dd>{selected.merchant_id}</dd></div><div><dt>Source row</dt><dd>{selected.row}</dd></div><div><dt>Top-level taxonomy</dt><dd lang="pt-BR">{selected.taxonomy_l0 || 'Not provided'}</dd></div></dl>
        {detailError && <p className="detail-error" role="alert">{detailError}</p>}
        {selected.item_metadata != null && <details className="raw-details"><summary>Source metadata</summary><pre>{JSON.stringify(selected.item_metadata, null, 2)}</pre></details>}
        {selected.item_profile != null && <details className="raw-details"><summary>Historical item profile</summary><p>Historical signals from the source data; these are not query relevance labels.</p><pre>{JSON.stringify(selected.item_profile, null, 2)}</pre></details>}
      </div></div>}
    </dialog>
  </div>
}

export default App
