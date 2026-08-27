import { useCallback, useEffect, useState } from 'react'
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
import HelpOutlineRoundedIcon from '@mui/icons-material/HelpOutlineRounded'
import ImageRoundedIcon from '@mui/icons-material/ImageRounded'
import MemoryRoundedIcon from '@mui/icons-material/MemoryRounded'
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded'
import RestartAltRoundedIcon from '@mui/icons-material/RestartAltRounded'
import ShieldRoundedIcon from '@mui/icons-material/ShieldRounded'
import VideocamRoundedIcon from '@mui/icons-material/VideocamRounded'
import { useDropzone, type Accept } from 'react-dropzone'
import {
  fetchHealth,
  fetchModels,
  inferImage,
  type HealthResponse,
  type InferenceResponse,
  type ModelId,
  type ModelMetadata,
} from './services/api'
import './App.css'

type Mode = 'image' | 'video' | 'camera'

const modes: Array<{ id: Mode; label: string; icon: React.ReactElement }> = [
  { id: 'image', label: 'Hình ảnh', icon: <ImageRoundedIcon /> },
  { id: 'video', label: 'Video', icon: <VideocamRoundedIcon /> },
  { id: 'camera', label: 'Camera', icon: <CameraAltRoundedIcon /> },
]

const modelOptions: Array<{
  id: ModelId
  name: string
  description: string
  defaultThreshold: number
  accent: string
}> = [
  {
    id: 'faster_rcnn',
    name: 'Faster R-CNN',
    description: 'Mô hình phát hiện hai giai đoạn',
    defaultThreshold: 0.85,
    accent: '#2563EB',
  },
  {
    id: 'retinanet',
    name: 'RetinaNet',
    description: 'Mô hình phát hiện một giai đoạn',
    defaultThreshold: 0.6,
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
  const [modelId, setModelId] = useState<ModelId>('faster_rcnn')
  const selectedModel = modelOptions.find((model) => model.id === modelId)!
  const [threshold, setThreshold] = useState(selectedModel.defaultThreshold)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelCatalog, setModelCatalog] = useState<ModelMetadata[]>([])
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [isInferring, setIsInferring] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const apiModel = modelCatalog.find((model) => model.id === modelId)
  const defaultThreshold = apiModel?.default_threshold ?? selectedModel.defaultThreshold

  useEffect(() => {
    setThreshold(defaultThreshold)
  }, [defaultThreshold])

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

  const accept = mode === 'video' ? VIDEO_ACCEPT : IMAGE_ACCEPT

  const onDrop = useCallback((files: File[]) => {
    setSelectedFile(files[0] ?? null)
  }, [])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    accept,
    maxFiles: 1,
    disabled: mode === 'camera',
    onDrop,
  })

  const handleModeChange = (_event: React.SyntheticEvent, nextMode: Mode) => {
    setMode(nextMode)
    setSelectedFile(null)
    setResult(null)
    setApiError(null)
  }

  const handleModelChange = (nextModel: ModelId) => {
    setModelId(nextModel)
    setResult(null)
  }
  const resetThreshold = () => setThreshold(defaultThreshold)

  const runDetection = async () => {
    if (mode !== 'image' || !selectedFile) return
    setIsInferring(true)
    setApiError(null)
    try {
      const payload = await inferImage(selectedFile, modelId, threshold)
      setResult(payload)
      setHealth(await fetchHealth())
    } catch (error) {
      setApiError(error instanceof Error ? error.message : 'Không thể chạy suy luận')
    } finally {
      setIsInferring(false)
    }
  }

  const displayedMediaUrl = result?.result_image ?? previewUrl
  const metricCards = [
    { label: 'Tổng đối tượng', color: '#0F172A', value: result ? String(result.detections.length) : '—', note: result ? 'detection' : 'Chờ suy luận' },
    { label: 'Không đội mũ', color: '#DC2626', value: result ? String(result.summary.NoHelmet ?? 0) : '—', note: 'NoHelmet' },
    { label: 'Có đội mũ', color: '#16A34A', value: result ? String(result.summary.Helmet ?? 0) : '—', note: 'Helmet' },
    { label: 'Xe và người lái', color: '#2563EB', value: result ? String(result.summary.BikeWithRider ?? 0) : '—', note: 'BikeWithRider' },
    { label: 'Độ trễ', color: '#7C3AED', value: result ? `${result.latency_ms.toFixed(1)} ms` : '—', note: result ? result.device.name : 'Chờ suy luận' },
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
                <Box className="upload-icon camera-icon">
                  <CameraAltRoundedIcon />
                </Box>
                <Typography variant="h6">Camera sẽ xuất hiện tại đây</Typography>
                <Typography color="text.secondary" sx={{ textAlign: 'center' }}>
                  Quyền camera chỉ được yêu cầu sau khi backend được kết nối.
                </Typography>
                <Button variant="outlined" startIcon={<CameraAltRoundedIcon />} disabled>
                  Mở camera
                </Button>
              </Box>
            ) : (
              <Box
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${previewUrl ? 'dropzone-with-preview' : ''}`}
              >
                <input {...getInputProps()} />
                {displayedMediaUrl && selectedFile ? (
                  <>
                    {mode === 'video' && !result ? (
                      <video className="media-preview" src={displayedMediaUrl} controls />
                    ) : (
                      <img className="media-preview" src={displayedMediaUrl} alt={result ? 'Ảnh kết quả phát hiện' : 'Tệp ảnh vừa chọn'} />
                    )}
                    <Box className="file-overlay">
                      <Typography variant="body2" noWrap sx={{ fontWeight: 700 }}>
                        {result ? `Kết quả · ${result.model.name}` : selectedFile.name}
                      </Typography>
                      <Typography variant="caption">
                        {result
                          ? `Threshold ${result.threshold.toFixed(2)} · ${result.detections.length} detection`
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
                  {isInferring
                    ? `Đang nạp và chạy ${selectedModel.name}…`
                    : health
                      ? `Máy chủ sẵn sàng · ${health.device_name}`
                      : 'Chưa kết nối máy chủ suy luận'}
                </Typography>
              </Stack>
              <Typography variant="caption" color="text.secondary">Kết quả sẽ giữ nguyên tỉ lệ ảnh gốc</Typography>
            </Box>
          </Paper>

          <Paper className="workspace-card control-card" elevation={0}>
            <Box className="card-heading compact">
              <Box>
                <Typography variant="h6">Cấu hình suy luận</Typography>
                <Typography variant="body2" color="text.secondary">Áp dụng cho dữ liệu đang chọn</Typography>
              </Box>
              <Chip label="Validation" size="small" color="success" variant="outlined" />
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
                  <Typography className="section-label">CONFIDENCE THRESHOLD</Typography>
                  <Typography variant="body2" color="text.secondary">Ngưỡng lọc kết quả hiển thị</Typography>
                </Box>
                <Typography className="threshold-value">{threshold.toFixed(2)}</Typography>
              </Stack>
              <Slider
                value={threshold}
                onChange={(_event, value) => setThreshold(value as number)}
                min={0.05}
                max={0.95}
                step={0.05}
                aria-label="Confidence threshold"
                sx={{ color: selectedModel.accent }}
              />
              <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
                <Chip label={`Mặc định validation: ${defaultThreshold.toFixed(2)}`} size="small" className="validation-chip" />
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
                startIcon={isInferring ? <CircularProgress size={18} color="inherit" /> : <PlayArrowRoundedIcon />}
                onClick={runDetection}
                disabled={isInferring || mode !== 'image' || !selectedFile || !health}
                sx={{ backgroundColor: selectedModel.accent }}
              >
                {isInferring
                  ? 'Đang phát hiện…'
                  : mode === 'image'
                    ? 'Bắt đầu phát hiện'
                    : 'Sẽ nối ở giai đoạn tiếp theo'}
              </Button>
              <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center' }}>
                {mode === 'image'
                  ? 'Lần chạy đầu tiên cần thời gian nạp checkpoint vào GPU.'
                  : 'Video và camera sẽ dùng lại pipeline ảnh sau khi được nghiệm thu.'}
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
              <Typography variant="h6">Chi tiết phát hiện</Typography>
              <Typography variant="body2" color="text.secondary">Bounding box, lớp và độ tin cậy của từng đối tượng</Typography>
            </Box>
            <Chip label={`${result?.detections.length ?? 0} đối tượng`} size="small" variant="outlined" />
          </Box>
          <Box className="table-head">
            <Typography>LỚP</Typography>
            <Typography>CONFIDENCE</Typography>
            <Typography>BOUNDING BOX</Typography>
          </Box>
          {result && result.detections.length > 0 ? (
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
                {result
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
