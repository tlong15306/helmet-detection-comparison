param(
    [string]$OutputDirectory = "report_drafts\assets\png"
)

Add-Type -AssemblyName System.Drawing

$projectRoot = Split-Path -Parent $PSScriptRoot
$outPath = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Force -Path $outPath | Out-Null

$fasterTest = Get-Content (Join-Path $projectRoot "outputs\faster_rcnn\metrics\test_metrics.json") -Raw | ConvertFrom-Json
$retinaTest = Get-Content (Join-Path $projectRoot "outputs\retinanet\metrics\test_metrics.json") -Raw | ConvertFrom-Json
$fasterLatency = Get-Content (Join-Path $projectRoot "outputs\faster_rcnn\metrics\latency_validation.json") -Raw | ConvertFrom-Json
$retinaLatency = Get-Content (Join-Path $projectRoot "outputs\retinanet\metrics\latency_validation.json") -Raw | ConvertFrom-Json

$blue = [System.Drawing.ColorTranslator]::FromHtml("#2F6B9A")
$orange = [System.Drawing.ColorTranslator]::FromHtml("#E07A3F")
$ink = [System.Drawing.ColorTranslator]::FromHtml("#1F2933")
$grid = [System.Drawing.ColorTranslator]::FromHtml("#D9E1E8")

function New-ChartCanvas([int]$Width, [int]$Height) {
    $bitmap = New-Object System.Drawing.Bitmap $Width, $Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.Clear([System.Drawing.Color]::White)
    return @($bitmap, $graphics)
}

function Draw-CenteredText($g, [string]$Text, [System.Drawing.Font]$Font, [System.Drawing.Brush]$Brush, [float]$CenterX, [float]$Y) {
    $size = $g.MeasureString($Text, $Font)
    $g.DrawString($Text, $Font, $Brush, $CenterX - $size.Width / 2, $Y)
}

function Draw-VerticalAxis($g, [int]$Left, [int]$Top, [int]$Width, [int]$Height, [double]$Maximum, [System.Drawing.Font]$LabelFont, [System.Drawing.Brush]$InkBrush, [System.Drawing.Pen]$GridPen) {
    $axisPen = New-Object System.Drawing.Pen($ink, 1)
    $g.DrawLine($axisPen, $Left, $Top, $Left, $Top + $Height)
    $g.DrawLine($axisPen, $Left, $Top + $Height, $Left + $Width, $Top + $Height)
    for ($i = 0; $i -le 5; $i++) {
        $value = $Maximum * $i / 5
        $y = $Top + $Height - $Height * $i / 5
        $g.DrawLine($GridPen, $Left, $y, $Left + $Width, $y)
        $label = "{0:N2}" -f $value
        $size = $g.MeasureString($label, $LabelFont)
        $g.DrawString($label, $LabelFont, $InkBrush, $Left - 10 - $size.Width, $y - $size.Height / 2)
    }
}

function Save-TestMetricChart {
    $canvas = New-ChartCanvas 1400 760
    $bitmap, $g = $canvas[0], $canvas[1]
    $titleFont = New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Bold)
    $labelFont = New-Object System.Drawing.Font("Arial", 13)
    $valueFont = New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Bold)
    $inkBrush = New-Object System.Drawing.SolidBrush($ink)
    $blueBrush = New-Object System.Drawing.SolidBrush($blue)
    $orangeBrush = New-Object System.Drawing.SolidBrush($orange)
    $gridPen = New-Object System.Drawing.Pen($grid, 1)
    Draw-CenteredText $g "So sánh chỉ số phát hiện trên tập kiểm thử (359 ảnh)" $titleFont $inkBrush 700 34
    $left, $top, $plotWidth, $plotHeight = 120, 125, 1180, 470
    Draw-VerticalAxis $g $left $top $plotWidth $plotHeight 1.0 $labelFont $inkBrush $gridPen
    $labels = @("mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "mAR@100")
    $fasterValues = @($fasterTest.metrics.map_50_95, $fasterTest.metrics.map_50, $fasterTest.metrics.map_75, $fasterTest.metrics.mar_100)
    $retinaValues = @($retinaTest.metrics.map_50_95, $retinaTest.metrics.map_50, $retinaTest.metrics.map_75, $retinaTest.metrics.mar_100)
    $groupWidth = $plotWidth / 4
    $barWidth = 72
    for ($i = 0; $i -lt 4; $i++) {
        $center = $left + $groupWidth * ($i + 0.5)
        $x1 = $center - $barWidth - 5
        $x2 = $center + 5
        $h1 = $plotHeight * $fasterValues[$i]
        $h2 = $plotHeight * $retinaValues[$i]
        $g.FillRectangle($blueBrush, $x1, $top + $plotHeight - $h1, $barWidth, $h1)
        $g.FillRectangle($orangeBrush, $x2, $top + $plotHeight - $h2, $barWidth, $h2)
        Draw-CenteredText $g ("{0:N3}" -f $fasterValues[$i]) $valueFont $inkBrush ($x1 + $barWidth / 2) ($top + $plotHeight - $h1 - 24)
        Draw-CenteredText $g ("{0:N3}" -f $retinaValues[$i]) $valueFont $inkBrush ($x2 + $barWidth / 2) ($top + $plotHeight - $h2 - 24)
        Draw-CenteredText $g $labels[$i] $labelFont $inkBrush $center ($top + $plotHeight + 15)
    }
    $g.FillRectangle($blueBrush, 500, 668, 18, 18)
    $g.DrawString("Faster R-CNN", $labelFont, $inkBrush, 526, 669)
    $g.FillRectangle($orangeBrush, 675, 668, 18, 18)
    $g.DrawString("RetinaNet", $labelFont, $inkBrush, 701, 669)
    $bitmap.Save((Join-Path $outPath "test_metrics_comparison.png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bitmap.Dispose()
}

function Draw-BarPanel($g, [int]$Left, [int]$Top, [int]$Width, [int]$Height, [string]$Title, [string]$Unit, [double[]]$Values) {
    $titleFont = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Bold)
    $labelFont = New-Object System.Drawing.Font("Arial", 12)
    $valueFont = New-Object System.Drawing.Font("Arial", 13, [System.Drawing.FontStyle]::Bold)
    $inkBrush = New-Object System.Drawing.SolidBrush($ink)
    $blueBrush = New-Object System.Drawing.SolidBrush($blue)
    $orangeBrush = New-Object System.Drawing.SolidBrush($orange)
    $gridPen = New-Object System.Drawing.Pen($grid, 1)
    Draw-CenteredText $g $Title $titleFont $inkBrush ($Left + $Width / 2) ($Top - 38)
    $maximum = ($Values | Measure-Object -Maximum).Maximum * 1.22
    Draw-VerticalAxis $g $Left $Top $Width $Height $maximum $labelFont $inkBrush $gridPen
    $barWidth = 110
    $centers = @(($Left + $Width * 0.28), ($Left + $Width * 0.72))
    $brushes = @($blueBrush, $orangeBrush)
    $names = @("Faster R-CNN", "RetinaNet")
    for ($i = 0; $i -lt 2; $i++) {
        $barHeight = $Height * $Values[$i] / $maximum
        $x = $centers[$i] - $barWidth / 2
        $y = $Top + $Height - $barHeight
        $g.FillRectangle($brushes[$i], $x, $y, $barWidth, $barHeight)
        Draw-CenteredText $g ("{0:N2}" -f $Values[$i]) $valueFont $inkBrush $centers[$i] ($y - 25)
        Draw-CenteredText $g $names[$i] $labelFont $inkBrush $centers[$i] ($Top + $Height + 15)
    }
    Draw-CenteredText $g $Unit $labelFont $inkBrush ($Left + $Width / 2) ($Top + $Height + 52)
}

function Save-LatencyChart {
    $canvas = New-ChartCanvas 1400 720
    $bitmap, $g = $canvas[0], $canvas[1]
    $titleFont = New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Bold)
    $inkBrush = New-Object System.Drawing.SolidBrush($ink)
    Draw-CenteredText $g "Benchmark suy luận trên tập xác thực (batch size = 1)" $titleFont $inkBrush 700 34
    Draw-BarPanel $g 120 150 470 410 "Latency suy luận trung bình" "ms / ảnh" @([double]$fasterLatency.latency.mean_ms, [double]$retinaLatency.latency.mean_ms)
    Draw-BarPanel $g 810 150 470 410 "FPS từ latency trung bình" "khung hình / giây" @([double]$fasterLatency.latency.fps_from_mean_latency, [double]$retinaLatency.latency.fps_from_mean_latency)
    $bitmap.Save((Join-Path $outPath "latency_fps_comparison.png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bitmap.Dispose()
}

Save-TestMetricChart
Save-LatencyChart
Write-Output "Created PNG charts in: $outPath"
