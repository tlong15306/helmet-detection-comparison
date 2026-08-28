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
  detection_id: string
  class_id: number
  class_name: string
  confidence: number
  box: [number, number, number, number]
}

export interface RiderHeadAssociation {
  head_detection_id: string
  helmet_status: 'helmet' | 'no_helmet'
  head_box: [number, number, number, number]
  association_score?: number
}

export interface RiderGroup {
  group_id: string
  bike_detection_id: string
  bike_box: [number, number, number, number]
  association_status: 'associated' | 'no_associated_head'
  heads: RiderHeadAssociation[]
  driver: {
    head_detection_id: string
    helmet_status: 'helmet' | 'no_helmet'
    role: 'driver_candidate' | 'driver'
    status: 'candidate_only' | 'rule_based'
    reason: string
    validation_evidence?: {
      split: 'validation'
      precision: number
      recall: number
      support: number
      source_tasks: string
    }
  } | null
}

export interface RiderAnalysis {
  version: 'association_baseline_v1' | 'rider_role_rule_v2'
  role_inference_status: 'candidate_only' | 'rule_based_with_abstention'
  rider_groups: RiderGroup[]
  unassigned_heads: Array<{ head_detection_id: string; helmet_status: string; reason: string }>
  ambiguous_heads: Array<{ head_detection_id: string; helmet_status: string; reason: string }>
  summary: {
    vehicles: number
    heads: number
    associated_heads: number
    unassigned_heads: number
    ambiguous_heads: number
    driver_candidates: number
    driver_candidate_no_helmet: number
    rule_based_drivers: number
    driver_no_helmet_alerts: number
    unknown_role_groups: number
    confirmed_driver_no_helmet: number
  }
  limitations: string[]
}

export type ReviewedHeadRole = 'driver' | 'passenger' | 'unknown'

export interface RoleReviewHead {
  annotation_id: number
  helmet_status: 'helmet' | 'no_helmet'
  box_xyxy: [number, number, number, number]
}

export interface RoleReviewTask {
  task_id: string
  image_id: number
  file_name: string
  bike_annotation_id: number
  difficulty_tags: string[]
  heads: RoleReviewHead[]
  review: {
    status: 'pending' | 'reviewed' | 'needs_second_review'
    reviewer: string | null
    driver_head_annotation_id: number | null
    head_roles: Record<string, ReviewedHeadRole | null>
    notes: string | null
  }
}

export interface RoleReviewQueue {
  schema_version: string
  tasks: RoleReviewTask[]
  summary: { tasks: number; pending: number; reviewed: number; needs_second_review: number }
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
  rider_analysis: RiderAnalysis
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
  preview_url: string | null
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

export function fetchRoleReviewTasks(): Promise<RoleReviewQueue> {
  return getJson<RoleReviewQueue>('/api/role-review/tasks')
}

export async function saveRoleReview(
  taskId: string,
  review: {
    reviewer: string
    driver_head_annotation_id: number | null
    head_roles: Record<string, ReviewedHeadRole>
    notes: string | null
    status: 'reviewed' | 'needs_second_review'
  },
): Promise<{ task: RoleReviewTask; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/role-review/tasks/${taskId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review),
  })
  if (!response.ok) throw new Error(await readError(response))
  return response.json() as Promise<{ task: RoleReviewTask; message: string }>
}
