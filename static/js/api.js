/**
 * Cliente API centralizado.
 * Configura la URL base via window.API_BASE_URL si el frontend vive en otro host.
 */
const API_BASE = (window.API_BASE_URL || '').replace(/\/$/, '');

class ApiError extends Error {
  constructor(status, body) {
    super(`HTTP ${status}: ${typeof body === 'string' ? body : JSON.stringify(body)}`);
    this.status = status;
    this.body = body;
  }
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const ct = res.headers.get('content-type') || '';
  const body = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

const api = {
  health:                    ()                          => request('/api/health'),
  equipos:                   ()                          => request('/api/equipos'),
  partidos:                  ()                          => request('/partidos'),
  historial:                 (competencia='ucl')         => request(`/api/historial?competencia=${encodeURIComponent(competencia)}`),
  evaluaciones:              ()                          => request('/evaluaciones'),
  crearEvaluacion:           (filepath)                  => request('/evaluaciones', { method: 'POST', body: JSON.stringify({ filepath }) }),
  estadoEvaluacion:          (id)                        => request(`/api/evaluaciones/${id}/status`),
  metricas:                  (id)                        => request(`/api/evaluaciones/${id}/metricas`),
  elos:                      (id)                        => request(`/api/evaluaciones/${id}/elos`),
  prediccionesTest:          (id)                        => request(`/api/evaluaciones/${id}/predicciones-test`),
  featureImportance:         (id, top=15)                => request(`/api/evaluaciones/${id}/feature-importance?top=${top}`),
  predecir:                  (payload)                   => request('/api/predecir', { method: 'POST', body: JSON.stringify(payload) }),
  trackList:                 ()                          => request('/api/track'),
  trackStats:                ()                          => request('/api/track/stats'),
  trackCrear:                (payload)                   => request('/api/track', { method: 'POST', body: JSON.stringify(payload) }),
  trackBorrar:               (id)                        => request(`/api/track/${id}`, { method: 'DELETE' }),
};

window.api = api;
window.ApiError = ApiError;
