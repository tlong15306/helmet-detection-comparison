export const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export type ModelId = 'faster_rcnn' | 'retinanet'

export interface HealthResponse {
  status: 'ready'
  cuda_available: boolean
  device: 'cuda' | 'cpu'
  device_name: string
  loaded_model: ModelId | null
}

export interface ModelMetadata {
  id: ModelId
  name: string
  architecture: string
  description: string
  checkpoint: string
  checkpoint_available: boolean
  default_threshold: number
  threshold_source: 'val'
  classes: Record<string, string>
}

export interface DetectionRecord {
  class_id: number
  class_name: string
  confidence: number
  box: [number, number, number, number]
}

export interface InferenceResponse {
  model: { id: ModelId; name: string; architecture: string }
  device: { type: 'cuda' | 'cpu'; name: string }
  input: { filename: string; width: number; height: number }
  threshold: number
  threshold_source: 'validation_default' | 'user_override'
  latency_ms: number
  summary: Record<string, number>
  detections: DetectionRecord[]
  result_image: string
}

export interface VideoJobResponse {
  id: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  model: { id: ModelId; name: string | null }
  threshold: number
  input: {
    filename: string
    width: number | null
    height: number | null
    fps: number | null
    total_frames: number | null
  }
  progress: { processed_frames: number; total_frames: number | null; percent: number }
  summary: Record<string, number>
  average_latency_ms: number | null
  device_name: string | null
  download_url: string | null
  error: string | null
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    return payload?.detail?.message ?? payload?.detail ?? `HTTP ${response.status}`
  } catch {
    return `Không thể kết nối máy chủ (HTTP ${response.status})`
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`)
  if (!response.ok) throw new Error(await readError(response))
  return response.json() as Promise<T>
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/api/health')
}

export async function fetchModels(): Promise<ModelMetadata[]> {
  const payload = await getJson<{ models: ModelMetadata[] }>('/api/models')
  return payload.models
}

export async function inferImage(
  file: File,
  modelId: ModelId,
  threshold: number,
): Promise<InferenceResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('model_id', modelId)
  form.append('threshold', threshold.toString())
  const response = await fetch(`${API_BASE_URL}/api/infer/image`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) throw new Error(await readError(response))
  return response.json() as Promise<InferenceResponse>
}

export async function inferVideo(
  file: File,
  modelId: ModelId,
  threshold: number,
): Promise<VideoJobResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('model_id', modelId)
  form.append('threshold', threshold.toString())
  const response = await fetch(`${API_BASE_URL}/api/infer/video`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) throw new Error(await readError(response))
  return response.json() as Promise<VideoJobResponse>
}

export function fetchVideoJob(jobId: string): Promise<VideoJobResponse> {
  return getJson<VideoJobResponse>(`/api/infer/video/jobs/${jobId}`)
}

export function absoluteApiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}
