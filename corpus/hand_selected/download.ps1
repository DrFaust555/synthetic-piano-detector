# Download the source videos of the hand-selected set as 44.1 kHz WAV:
#   <AudioRoot>/audio/_raw/<raw_src>.wav   (raw_src, url: data/protocol/manifest.csv; one download per raw_src)
# The files of the set are then cut from these (concat_segments.ps1 where the manifest note lists
# segments) and trimmed / peak-normalised (trim_normalize.ps1).
#   powershell -File corpus/hand_selected/download.ps1 [-AudioRoot <folder>]   (default: $env:SPD_AUDIO)
# Needs yt-dlp and ffmpeg on PATH (or ffmpeg in the folder FFMPEG_DIR).
param([string]$AudioRoot = $env:SPD_AUDIO)
$ErrorActionPreference = 'Continue'
if ($env:FFMPEG_DIR) { $env:Path = "$env:FFMPEG_DIR;$env:Path" }
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$base = Join-Path $AudioRoot "audio"
$raw  = Join-Path $base "_raw"
New-Item -ItemType Directory -Force $raw | Out-Null

$items = Import-Csv (Join-Path $repo 'data\protocol\manifest.csv') |
  Where-Object { $_.url -match 'youtu' } | Sort-Object raw_src -Unique |
  ForEach-Object { @{i=$_.raw_src; u=$_.url} }

$archive = Join-Path $raw "_archive.txt"
foreach ($it in $items) {
  $out = Join-Path $raw ("{0}.%(ext)s" -f $it.i)
  Write-Output "===== $($it.i)  $($it.u) ====="
  yt-dlp --no-playlist --no-warnings --retries 5 -N 4 `
    -x --audio-format wav --audio-quality 0 `
    --postprocessor-args "ffmpeg:-ar 44100" `
    -o $out `
    --download-archive $archive `
    $it.u
  if ($LASTEXITCODE -ne 0) { Write-Output "!!!! FAILED $($it.i) exit=$LASTEXITCODE" }
}
Write-Output "===== ALL DONE ====="
