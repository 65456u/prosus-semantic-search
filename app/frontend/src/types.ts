export type MethodId = 'bm25' | 'e5' | 'hybrid' | 'harrier' | 'full'
export type View = 'search' | 'compare' | 'dataset'
export interface Item {
  document_id: string
  row: number
  item_id: string
  merchant_id: string
  name: string
  description: string
  category: string
  taxonomy_l0: string
  price: number | null
  images: string[]
  image_url: string | null
  vegan: boolean
  lac_free: boolean
  organic: boolean
  item_metadata?: unknown
  item_profile?: unknown
}
export interface Method {
  id: MethodId
  name: string
  ready: boolean
  description: string
}
export interface SystemStatus {
  status: 'ready' | 'loading'
  counts: { items: number; unique_items: number; queries: number; merchants: number }
  methods: Method[]
  semantic: { state: 'not_started' | 'loading' | 'ready' | 'error'; progress: number | null; message: string }
}
export interface Query { id: number; text: string }
export interface SearchResponse {
  query: string
  method: MethodId
  results: (Item & { rank: number; score: number })[]
  timings_ms: { query_encoding: number; retrieval: number; total: number }
  cache_hit: boolean
  warnings?: string[]
  understanding?: { interpretation_en: string; rewrite_pt: string }
}
export interface FailedRun { method: MethodId; error: string }
export type CompareRun = SearchResponse | FailedRun
export interface CatalogResponse {
  items: Item[]
  total: number
  categories: { name: string; count: number }[]
}
