import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  AppBar,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  FormControlLabel,
  IconButton,
  LinearProgress,
  Paper,
  Radio,
  Slider,
  Stack,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material'
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded'
import CameraAltRoundedIcon from '@mui/icons-material/CameraAltRounded'
import CloudUploadRoundedIcon from '@mui/icons-material/CloudUploadRounded'
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded'
import HelpOutlineRoundedIcon from '@mui/icons-material/HelpOutlineRounded'
import ImageRoundedIcon from '@mui/icons-material/ImageRounded'
import MemoryRoundedIcon from '@mui/icons-material/MemoryRounded'
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded'
import RestartAltRoundedIcon from '@mui/icons-material/RestartAltRounded'
import ShieldRoundedIcon from '@mui/icons-material/ShieldRounded'
import VideocamRoundedIcon from '@mui/icons-material/VideocamRounded'
import { useDropzone, type Accept } from 'react-dropzone'
import {
  absoluteApiUrl,
  fetchHealth,
  fetchModels,
  fetchVideoJob,
  inferImage,
  inferVideo,
  type HealthResponse,
  type InferenceResponse,
  type ModelId,
  type ModelMetadata,
  type VideoJobResponse,
} from './services/api'
import './App.css'

type Mode = 'image' | 'video' | 'camera'
type ClassThresholds = Record<'BikeWithRider' | 'NoHelmet' | 'Helmet', number>

const CLASS_THRESHOLD_LABELS: Array<{ key: keyof ClassThresholds; label: string }> = [
  { key: 'BikeWithRider', label: 'Xe và người lái' },
  { key: 'NoHelmet', label: 'Không đội mũ' },
  { key: 'Helmet', label: 'Có đội mũ' },
]

const modes: Array<{ id: Mode; label: string; icon: React.ReactElement }> = [
  { id: 'image', label: 'Hình ảnh', icon: <ImageRoundedIcon /> },
  { id: 'video', label: 'Video', icon: <VideocamRoundedIcon /> },
  { id: 'camera', label: 'Camera', icon: <CameraAltRoundedIcon /> },
]

const modelOptions: Array<{
  id: ModelId
  name: string
  description: string
  defaultThresholds: ClassThresholds
  accent: string
}> = [
  {
    id: 'faster_rcnn_vietnam_v6',
    name: 'Faster R-CNN · fine-tune VN',
    description: 'Fine-tune từ baseline với dữ liệu Việt Nam v6',
    defaultThresholds: { BikeWithRider: 0.95, NoHelmet: 0.65, Helmet: 0.70 },
    accent: '#2563EB',
  },
  {
    id: 'retinanet_vietnam_v6',
    name: 'RetinaNet · fine-tune VN',
    description: 'Fine-tune từ baseline với dữ liệu Việt Nam v6',
    defaultThresholds: { BikeWithRider: 0.65, NoHelmet: 0.40, Helmet: 0.40 },
    accent: '#7C3AED',
  },
]

const IMAGE_ACCEPT: Accept = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
}

const VIDEO_ACCEPT: Accept = {
  'video/mp4': ['.mp4'],
  'video/quicktime': ['.mov'],
  'video/x-msvideo': ['.avi'],
}

function App() {
  const [mode, setMode] = useState<Mode>('image')
  // Demo dùng các checkpoint fine-tune từ baseline với dữ liệu Việt Nam v6.
  // Baseline gốc và candidate train gộp từ đầu vẫn được giữ ở backend để rollback/đối chiếu.
  const [modelId, setModelId] = useState<ModelId>('faster_rcnn_vietnam_v6')
  const selectedModel = modelOptions.find((model) => model.id === modelId)!
  const [thresholds, setThresholds] = useState<ClassThresholds>(selectedModel.defaultThresholds)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelCatalog, setModelCatalog] = useState<ModelMetadata[]>([])
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [videoJob, setVideoJob] = useState<VideoJobResponse | null>(null)
  const [isInferring, setIsInferring] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null)
  const cameraStreamRef = useRef<MediaStream | null>(null)
  const apiModel = modelCatalog.find((model) => model.id === modelId)
  const defaultThresholds = (apiModel?.default_thresholds ?? selectedModel.defaultThresholds) as ClassThresholds

  useEffect(() => {
    setThresholds(defaultThresholds)
  }, [modelId, defaultThresholds.BikeWithRider, defaultThresholds.NoHelmet, defaultThresholds.Helmet])

  const connectBackend = useCallback(async () => {
    setIsConnecting(true)
    try {
      const [healthPayload, modelsPayload] = await Promise.all([fetchHealth(), fetchModels()])
      setHealth(healthPayload)
      setModelCatalog(modelsPayload)
      setApiError(null)
    } catch (error) {
      setApiError(error instanceof Error ? error.message : 'Không thể kết nối máy chủ')
    } finally {
      setIsConnecting(false)
    }
  }, [])

  useEffect(() => {
    void connectBackend()
    const retryTimer = window.setInterval(() => {
      if (!health) void connectBackend()
    }, 5000)
    return () => window.clearInterval(retryTimer)
  }, [connectBackend, health])

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      setResult(null)
      return
    }
    const url = URL.createObjectURL(selectedFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  const stopCamera = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
    cameraStreamRef.current = null
    setCameraActive(false)
  }, [])

  useEffect(() => stopCamera, [stopCamera])

  useEffect(() => {
    if (cameraActive && cameraVideoRef.current && cameraStreamRef.current) {
      cameraVideoRef.current.srcObject = cameraStreamRef.current
    }
  }, [cameraActive])

  useEffect(() => {
    if (!videoJob || !['queued', 'processing'].includes(videoJob.status)) return
    let cancelled = false
    const refresh = async () => {
      try {
        const next = await fetchVideoJob(videoJob.id)
        if (!cancelled) setVideoJob(next)
      } catch (error) {
        if (!cancelled) setApiError(error instanceof Error ? error.message : 'Không thể cập nhật tiến độ video')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 1200)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [videoJob?.id, videoJob?.status])

  const accept = mode === 'video' ? VIDEO_ACCEPT : IMAGE_ACCEPT

  const onDrop = useCallback((files: File[]) => {
    setSelectedFile(files[0] ?? null)
    setResult(null)
    setVideoJob(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    accept,
    maxFiles: 1,
    disabled: mode === 'camera',
    onDrop,
  })

  const handleModeChange = (_event: React.SyntheticEvent, nextMode: Mode) => {
    if (nextMode !== 'camera') stopCamera()
    setMode(nextMode)
    setSelectedFile(null)
    setResult(null)
    setVideoJob(null)
    setApiError(null)
  }

  const handleModelChange = (nextModel: ModelId) => {
    setModelId(nextModel)
    setResult(null)
    setVideoJob(null)
  }
  const resetThreshold = () => setThresholds(defaultThresholds)
  const setClassThreshold = (key: keyof ClassThresholds, value: number) => {
    setThresholds((current) => ({ ...current, [key]: value }))
  }
  const isValidationThresholds = CLASS_THRESHOLD_LABELS.every(
    ({ key }) => thresholds[key] === defaultThresholds[key],
  )
  const isExperimentalModel = apiModel?.threshold_source === 'exploratory'

  const openCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' } },
      })
      stopCamera()
      cameraStreamRef.current = stream
      if (cameraVideoRef.current) cameraVideoRef.current.srcObject = stream
      setCameraActive(true)
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : 'Không thể mở camera')
    }
  }

  const captureCameraFrame = async () => {
    const video = cameraVideoRef.current
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      setCameraError('Camera chưa sẵn sàng; hãy chờ vài giây rồi thử lại.')
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (!context) {
      setCameraError('Trình duyệt không hỗ trợ chụp frame từ camera.')
      return
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
    if (!blob) {
      setCameraError('Không thể tạo ảnh chụp từ camera.')
      return
    }
    setSelectedFile(new File([blob], `camera_${Date.now()}.png`, { type: 'image/png' }))
    setResult(null)
    setCameraError(null)
  }

  const runDetection = async () => {
    if ((mode !== 'image' && mode !== 'video' && mode !== 'camera') || !selectedFile) return
    setIsInferring(true)
    setApiError(null)
    try {
      if (mode === 'image' || mode === 'camera') {
        const payload = await inferImage(selectedFile, modelId, thresholds)
        setResult(payload)
      } else {
        const job = await inferVideo(selectedFile, modelId, thresholds)
        setVideoJob(job)
      }
      setHealth(await fetchHealth())
    } catch (error) {
      setApiError(error instanceof Error ? error.message : 'Không thể chạy suy luận')
    } finally {
      setIsInferring(false)
    }
  }

  const isVideoProcessing = videoJob?.status === 'queued' || videoJob?.status === 'processing'
  const completedVideoUrl = videoJob?.preview_url ? absoluteApiUrl(videoJob.preview_url) : null
  const displayedMediaUrl = mode === 'video'
    ? completedVideoUrl ?? previewUrl
    : result?.result_image ?? previewUrl
  const videoSummary = videoJob?.summary ?? {}
  const videoDetectionTotal = Object.entries(videoSummary)
    .filter(([className]) => className !== 'DriverNoHelmetAlert')
    .reduce((total, [, count]) => total + count, 0)
  const isVideoComplete = mode === 'video' && videoJob?.status === 'completed'
  const metricCards = [
    { label: mode === 'video' ? 'Tổng theo frame' : 'Tổng đối tượng', color: '#0F172A', value: mode === 'video' && videoJob ? String(videoDetectionTotal) : result ? String(result.detections.length) : '—', note: mode === 'video' ? 'lượt phát hiện' : result ? 'detection hậu xử lý' : 'Chờ suy luận' },
    { label: 'Cảnh báo đủ điều kiện', color: '#DC2626', value: mode === 'video' ? (videoJob ? String(videoSummary.DriverNoHelmetAlert ?? 0) : '—') : result ? String(result.alerts.length) : '—', note: 'tài xế không đội mũ' },
    { label: 'Có đội mũ', color: '#16A34A', value: mode === 'video' ? (videoJob ? String(videoSummary.Helmet ?? 0) : '—') : result ? String(result.summary.Helmet ?? 0) : '—', note: 'Helmet' },
    { label: 'Độ trễ', color: '#7C3AED', value: mode === 'video' && videoJob?.average_latency_ms != null ? `${videoJob.average_latency_ms.toFixed(1)} ms` : result ? `${result.latency_ms.toFixed(1)} ms` : '—', note: mode === 'video' ? videoJob?.device_name ?? 'Chờ xử lý' : result ? result.device.name : 'Chờ suy luận' },
  ]

  return (
    <Box className="app-shell">
      <AppBar position="static" color="transparent" elevation={0} className="topbar">
        <Container maxWidth="xl" className="topbar-inner">
          <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
            <Box className="brand-mark" aria-hidden="true">
              <ShieldRoundedIcon />
            </Box>
            <Box>
              <Typography className="brand-title">Helmet Detection AI</Typography>
              <Typography className="brand-caption">Faster R-CNN × RetinaNet</Typography>
            </Box>
          </Stack>

          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
            <Chip
              className="preview-chip"
              icon={<AutoAwesomeRoundedIcon />}
              label={health ? 'Backend sẵn sàng' : 'Đang kết nối backend'}
              size="small"
            />
            <Chip
              className="device-chip"
              icon={<MemoryRoundedIcon />}
              label={health?.device_name ?? 'Đang kiểm tra thiết bị'}
              size="small"
            />
            <Tooltip title="Số liệu chỉ xuất hiện sau khi kết nối backend">
              <IconButton aria-label="Thông tin bản xem trước" size="small">
                <HelpOutlineRoundedIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </Container>
      </AppBar>

      <Container maxWidth="xl" component="main" className="main-container">
        <Box className="page-heading">
          <Box>
            <Typography component="p" className="eyebrow">
              PHÂN TÍCH GIAO THÔNG BẰNG THỊ GIÁC MÁY TÍNH
            </Typography>
            <Typography component="h1" variant="h4">
              Phát hiện vi phạm mũ bảo hiểm
            </Typography>
            <Typography color="text.secondary" className="page-description">
              Tải dữ liệu giao thông, lựa chọn mô hình và quan sát kết quả phát hiện trên cùng một không gian làm việc.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} className="legend-row">
            <Chip className="legend-chip no-helmet" label="NoHelmet" size="small" />
            <Chip className="legend-chip helmet" label="Helmet" size="small" />
            <Chip className="legend-chip rider" label="BikeWithRider" size="small" />
          </Stack>
        </Box>

        <Paper className="mode-tabs" elevation={0}>
          <Tabs value={mode} onChange={handleModeChange} aria-label="Chọn loại dữ liệu đầu vào" variant="fullWidth">
            {modes.map((item) => (
              <Tab key={item.id} value={item.id} icon={item.icon} iconPosition="start" label={item.label} />
            ))}
          </Tabs>
        </Paper>

        <Box className="workspace-grid">
          <Paper className="workspace-card media-card" elevation={0}>
            <Box className="card-heading">
              <Box>
                <Typography variant="h6">Không gian phân tích</Typography>
                <Typography variant="body2" color="text.secondary">
                  {mode === 'image' && 'JPG, JPEG hoặc PNG'}
                  {mode === 'video' && 'MP4, MOV hoặc AVI'}
                  {mode === 'camera' && 'Ảnh chụp từ camera của thiết bị'}
                </Typography>
              </Box>
              <Chip
                label={selectedModel.name}
                size="small"
                sx={{ color: selectedModel.accent, backgroundColor: `${selectedModel.accent}12`, borderColor: `${selectedModel.accent}2E` }}
                variant="outlined"
              />
            </Box>

            {mode === 'camera' ? (
              <Box className="dropzone camera-placeholder">
                {result && selectedFile ? (
                  <>
                    <img className="media-preview" src={result.result_image} alt="Kết quả phát hiện từ camera" />
                    <Box className="file-overlay">
                      <Typography variant="body2" noWrap sx={{ fontWeight: 700 }}>Kết quả ảnh chụp · {result.model.name}</Typography>
                      <Typography variant="caption">{result.threshold_source === 'validation_default' ? 'Ngưỡng validation' : 'Ngưỡng tùy chỉnh'} · {result.detections.length} detection</Typography>
                    </Box>
                  </>
                ) : cameraActive ? (
                  <video ref={cameraVideoRef} className="media-preview" autoPlay playsInline muted />
                ) : previewUrl ? (
                  <img className="media-preview" src={previewUrl} alt="Ảnh chụp camera" />
                ) : (
                  <>
                    <Box className="upload-icon camera-icon"><CameraAltRoundedIcon /></Box>
                    <Typography variant="h6">Sẵn sàng chụp từ camera</Typography>
                    <Typography color="text.secondary" sx={{ textAlign: 'center' }}>
                      Quyền camera chỉ được yêu cầu khi bạn bấm mở camera.
                    </Typography>
                  </>
                )}
                <Stack direction="row" spacing={1} className="camera-actions">
                  {cameraActive ? (
                    <>
                      <Button variant="outlined" onClick={stopCamera}>Tắt camera</Button>
                      <Button variant="contained" startIcon={<CameraAltRoundedIcon />} onClick={() => void captureCameraFrame()}>
                        Chụp ảnh
                      </Button>
                    </>
                  ) : (
                    <Button variant="outlined" startIcon={<CameraAltRoundedIcon />} onClick={() => void openCamera()}>
                      Mở camera
                    </Button>
                  )}
                </Stack>
              </Box>
            ) : (
              <Box
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${previewUrl ? 'dropzone-with-preview' : ''}`}
              >
                <input {...getInputProps()} />
                {displayedMediaUrl && selectedFile ? (
                  <>
                    {mode === 'video' ? (
                      <video className="media-preview" src={displayedMediaUrl} controls />
                    ) : (
                      <img className="media-preview" src={displayedMediaUrl} alt={result ? 'Ảnh kết quả phát hiện' : 'Tệp ảnh vừa chọn'} />
                    )}
                    <Box className="file-overlay">
                      <Typography variant="body2" noWrap sx={{ fontWeight: 700 }}>
                        {mode === 'video' && videoJob?.status === 'completed'
                          ? `Kết quả · ${videoJob.model.name ?? selectedModel.name}`
                          : result ? `Kết quả · ${result.model.name}` : selectedFile.name}
                      </Typography>
                      <Typography variant="caption">
                        {mode === 'video' && videoJob
                          ? `${videoJob.progress.processed_frames}${videoJob.progress.total_frames ? `/${videoJob.progress.total_frames}` : ''} frame · NoHelmet ≥ ${videoJob.threshold.toFixed(2)}`
                          : result
                          ? `${result.threshold_source === 'validation_default' ? 'Ngưỡng validation' : 'Ngưỡng tùy chỉnh'} · ${result.detections.length} detection`
                          : `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB · Nhấn để đổi tệp`}
                      </Typography>
                    </Box>
                  </>
                ) : (
                  <>
                    <Box className="upload-icon"><CloudUploadRoundedIcon /></Box>
                    <Typography variant="h6">
                      {isDragActive ? 'Thả tệp vào đây' : `Kéo thả ${mode === 'video' ? 'video' : 'hình ảnh'} vào đây`}
                    </Typography>
                    <Typography color="text.secondary" sx={{ textAlign: 'center' }}>
                      hoặc chọn tệp từ máy tính để chuẩn bị suy luận
                    </Typography>
                    <Button variant="outlined" startIcon={<CloudUploadRoundedIcon />}>Chọn tệp</Button>
                  </>
                )}
              </Box>
            )}

            {fileRejections.length > 0 && (
              <Alert severity="error" sx={{ mt: 2 }}>Tệp không đúng định dạng được hỗ trợ.</Alert>
            )}
            {cameraError && <Alert severity="error" sx={{ mt: 2 }}>{cameraError}</Alert>}
            {mode === 'video' && videoJob && (
              <Box className="video-progress-panel">
                <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', mb: 0.8 }}>
                  <Typography variant="body2" sx={{ fontWeight: 760 }}>
                    {videoJob.status === 'queued' && 'Đang chờ GPU xử lý'}
                    {videoJob.status === 'processing' && 'Đang gắn nhãn video'}
                    {videoJob.status === 'completed' && 'Video kết quả đã sẵn sàng'}
                    {videoJob.status === 'failed' && 'Không thể xử lý video'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">{videoJob.progress.percent.toFixed(0)}%</Typography>
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={videoJob.status === 'queued' ? 0 : videoJob.progress.percent}
                  color={videoJob.status === 'failed' ? 'error' : 'primary'}
                />
                {videoJob.error ? (
                  <Alert severity="error" sx={{ mt: 1 }}>{videoJob.error}</Alert>
                ) : videoJob.status === 'completed' && videoJob.download_url ? (
                  <Button
                    component="a"
                    href={absoluteApiUrl(videoJob.download_url)}
                    download
                    size="small"
                    startIcon={<DownloadRoundedIcon />}
                    sx={{ mt: 1 }}
                  >
                    Tải video đã gắn nhãn
                  </Button>
                ) : (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                    Video được xử lý lần lượt từng frame. Khi GPU đang xử lý video, hãy chờ tác vụ hoàn tất trước khi đổi mô hình.
                  </Typography>
                )}
              </Box>
            )}
            {apiError && (
              <Alert
                severity="error"
                sx={{ mt: 2 }}
                action={
                  <Button color="inherit" size="small" onClick={() => void connectBackend()} disabled={isConnecting}>
                    {isConnecting ? 'Đang thử…' : 'Thử lại'}
                  </Button>
                }
              >
                {apiError}
              </Alert>
            )}

            <Box className="media-footer">
              <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                <Box className={`status-dot ${health ? 'status-dot-ready' : ''}`} />
                <Typography variant="body2" color="text.secondary">
                  {isInferring || isVideoProcessing
                    ? `Đang nạp và chạy ${selectedModel.name}…`
                    : health
                      ? `Máy chủ sẵn sàng · ${health.device_name}`
                      : 'Chưa kết nối máy chủ suy luận'}
                </Typography>
              </Stack>
              <Typography variant="caption" color="text.secondary">{mode === 'video' ? 'Khuyến nghị dùng video ngắn, tối đa 5 phút / 200 MB' : 'Kết quả sẽ giữ nguyên tỉ lệ ảnh gốc'}</Typography>
            </Box>
          </Paper>

          <Paper className="workspace-card control-card" elevation={0}>
            <Box className="card-heading compact">
              <Box>
                <Typography variant="h6">Cấu hình suy luận</Typography>
                <Typography variant="body2" color="text.secondary">Áp dụng cho dữ liệu đang chọn</Typography>
              </Box>
              <Chip
                label={isExperimentalModel ? 'Thử nghiệm' : isValidationThresholds ? 'Validation' : 'Tùy chỉnh'}
                size="small"
                color={isExperimentalModel ? 'warning' : isValidationThresholds ? 'success' : 'warning'}
                variant="outlined"
              />
            </Box>

            <Divider />

            <Box className="control-section">
              <Typography className="section-label">MÔ HÌNH</Typography>
              <Stack spacing={1.25}>
                {modelOptions.map((model) => {
                  const active = model.id === modelId
                  return (
                    <Box
                      key={model.id}
                      className={`model-option ${active ? 'model-option-active' : ''}`}
                      sx={{ '--model-accent': model.accent } as React.CSSProperties}
                      onClick={() => handleModelChange(model.id)}
                      role="radio"
                      aria-checked={active}
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') handleModelChange(model.id)
                      }}
                    >
                      <FormControlLabel
                        value={model.id}
                        control={<Radio checked={active} size="small" />}
                        label={
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 750 }}>{model.name}</Typography>
                            <Typography variant="caption" color="text.secondary">{model.description}</Typography>
                          </Box>
                        }
                        sx={{ m: 0, flex: 1, pointerEvents: 'none' }}
                      />
                      <Chip label={active ? 'Đang chọn' : 'Chọn'} size="small" variant={active ? 'filled' : 'outlined'} />
                    </Box>
                  )
                })}
              </Stack>
            </Box>

            <Divider />

            <Box className="control-section threshold-section">
              <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography className="section-label">CONFIDENCE THRESHOLD THEO LỚP</Typography>
                  <Typography variant="body2" color="text.secondary">Ngưỡng lọc chỉ ảnh hưởng demo</Typography>
                </Box>
              </Stack>
              {CLASS_THRESHOLD_LABELS.map(({ key, label }) => (
                <Box key={key} sx={{ mt: 1.25 }}>
                  <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>{label}</Typography>
                    <Typography className="threshold-value">{thresholds[key].toFixed(2)}</Typography>
                  </Stack>
                  <Slider
                    value={thresholds[key]}
                    onChange={(_event, value) => setClassThreshold(key, value as number)}
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    aria-label={`Confidence threshold ${label}`}
                    sx={{ color: selectedModel.accent, py: 0.5 }}
                  />
                </Box>
              ))}
              <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
                <Chip
                  label={isExperimentalModel ? 'Ngưỡng kế thừa baseline' : 'Mặc định validation'}
                  size="small"
                  className="validation-chip"
                />
                <Tooltip title="Đưa về ngưỡng đã chọn trên validation">
                  <IconButton aria-label="Khôi phục threshold mặc định" onClick={resetThreshold} size="small">
                    <RestartAltRoundedIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Stack>
              <Alert severity="info" icon={false} className="threshold-note">
                Thay đổi ở đây chỉ ảnh hưởng demo, không làm thay đổi metric trong báo cáo.
              </Alert>
            </Box>

            <Box className="action-area">
              <Button
                fullWidth
                size="large"
                variant="contained"
                startIcon={isInferring || isVideoProcessing ? <CircularProgress size={18} color="inherit" /> : <PlayArrowRoundedIcon />}
                onClick={runDetection}
                disabled={isInferring || isVideoProcessing || !selectedFile || !health}
                sx={{ backgroundColor: selectedModel.accent }}
              >
                {isInferring
                  ? 'Đang phát hiện…'
                  : mode === 'image'
                    ? 'Bắt đầu phát hiện'
                    : mode === 'video'
                      ? isVideoProcessing ? 'Đang xử lý video…' : 'Bắt đầu xử lý video'
                      : 'Phát hiện ảnh chụp'}
              </Button>
              <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center' }}>
                {mode === 'image'
                  ? 'Lần chạy đầu tiên cần thời gian nạp checkpoint vào GPU.'
                  : mode === 'video'
                    ? 'Video được đưa vào hàng đợi để không làm tràn bộ nhớ GPU.'
                    : 'Mở camera, chụp một frame, sau đó chạy cùng pipeline suy luận ảnh.'}
              </Typography>
            </Box>
          </Paper>
        </Box>

        <Box className="metrics-grid" aria-label="Tổng quan kết quả suy luận">
          {metricCards.map((metric) => (
            <Paper key={metric.label} className="metric-card" elevation={0}>
              <Box className="metric-accent" sx={{ backgroundColor: metric.color }} />
              <Typography variant="body2" color="text.secondary">{metric.label}</Typography>
              <Typography variant="h5">{metric.value}</Typography>
              <Typography variant="caption" color="text.secondary">{metric.note}</Typography>
            </Paper>
          ))}
        </Box>

        <Paper className="results-card" elevation={0}>
          <Box className="card-heading compact">
            <Box>
              <Typography variant="h6">{mode === 'video' ? 'Tổng hợp xử lý video' : 'Chi tiết phát hiện'}</Typography>
              <Typography variant="body2" color="text.secondary">
                {mode === 'video' ? 'Số lượt phát hiện được cộng dồn trên các frame đã xử lý' : 'Bounding box, lớp và độ tin cậy của từng đối tượng'}
              </Typography>
            </Box>
            <Chip label={mode === 'video' ? `${videoJob?.progress.processed_frames ?? 0} frame` : `${result?.detections.length ?? 0} đối tượng`} size="small" variant="outlined" />
          </Box>
          <Box className="table-head">
            <Typography>LỚP</Typography>
            <Typography>{mode === 'video' ? 'TỔNG LƯỢT' : 'CONFIDENCE'}</Typography>
            <Typography>{mode === 'video' ? 'TRẠNG THÁI' : 'BOUNDING BOX'}</Typography>
          </Box>
          {mode !== 'video' && result && result.alerts.length > 0 && (
            <Stack spacing={1} sx={{ px: 2, pt: 1.5 }}>
              {result.alerts.map((alert) => (
                <Alert key={alert.group_id} severity="error">{alert.message}</Alert>
              ))}
            </Stack>
          )}
          {mode === 'video' && videoJob && Object.keys(videoSummary).length > 0 ? (
            <Box className="detection-list">
              {Object.entries(videoSummary).map(([className, count]) => (
                <Box className="detection-row" key={className}>
                  <Chip label={className} size="small" className={`detection-class detection-${className.toLowerCase()}`} />
                  <Typography variant="body2" sx={{ fontWeight: 760 }}>{count}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {isVideoComplete ? 'Đã hoàn tất' : 'Đang cập nhật'}
                  </Typography>
                </Box>
              ))}
            </Box>
          ) : result && result.detections.length > 0 ? (
            <Box className="detection-list">
              {result.detections.map((detection, index) => (
                <Box className="detection-row" key={`${detection.class_name}-${index}`}>
                  <Chip
                    label={detection.class_name}
                    size="small"
                    className={`detection-class detection-${detection.class_name.toLowerCase()}`}
                  />
                  <Typography variant="body2" sx={{ fontWeight: 760 }}>
                    {(detection.confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary" className="box-coordinates">
                    [{detection.box.map((value) => value.toFixed(0)).join(', ')}]
                  </Typography>
                </Box>
              ))}
            </Box>
          ) : (
            <Box className="empty-results">
              <LinearProgress className="empty-line" variant="determinate" value={0} />
              <Typography variant="body2" color="text.secondary">
                {mode === 'video'
                  ? videoJob?.status === 'failed'
                    ? 'Video chưa tạo được kết quả; xem thông báo lỗi phía trên.'
                    : 'Tổng hợp theo lớp sẽ xuất hiện ngay khi video bắt đầu xử lý.'
                  : result
                  ? 'Không có detection nào đạt threshold hiện tại.'
                  : 'Kết quả chi tiết sẽ xuất hiện sau khi mô hình hoàn tất suy luận.'}
              </Typography>
            </Box>
          )}
        </Paper>
      </Container>

    </Box>
  )
}

export default App
