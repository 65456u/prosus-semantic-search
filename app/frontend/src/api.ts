export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new Error('The search service is unavailable. Check that the local server is running, then try again.')
  }
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.detail
    throw new Error(typeof detail === 'string' ? detail : `The request could not be completed (${response.status}). Please try again.`)
  }
  if (payload === null) throw new Error('The service returned an unexpected response. Please try again.')
  return payload as T
}
