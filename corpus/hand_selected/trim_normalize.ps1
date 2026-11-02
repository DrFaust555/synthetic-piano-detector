# One file of the hand-selected set from a raw download: optional ffmpeg seek (e.g. "-ss 55.3" or "-t 55.3"),
# leading/trailing silence (< -45 dB peak) removed, peak normalised to -1 dBFS, 44.1 kHz PCM-16.
#   powershell -File corpus/hand_selected/trim_normalize.ps1 <rawId> <outName> ["<seek>"] [-AudioRoot <folder>]
# reads <AudioRoot>/audio/_raw/<rawId>.wav, writes <AudioRoot>/audio/<outName>.
param([string]$rawId, [string]$outName, [string]$seek = "", [string]$AudioRoot = $env:SPD_AUDIO)
$ErrorActionPreference = 'Continue'
# ffmpeg from PATH, or from the folder in the environment variable FFMPEG_DIR
$ffexe = if ($env:FFMPEG_DIR) { Join-Path $env:FFMPEG_DIR "ffmpeg.exe" } else { "ffmpeg" }
$base  = Join-Path $AudioRoot "audio"
$raw   = Join-Path $base "_raw"
$vdlog = Join-Path $env:TEMP "_vd_seg.txt"
$trim = "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB:detection=peak,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB:detection=peak,areverse"

$src = Join-Path $raw "$rawId.wav"
$out = Join-Path $base $outName
cmd /c "`"$ffexe`" -hide_banner $seek -i `"$src`" -af volumedetect -f null NUL 2> `"$vdlog`""
$m = Select-String -Path $vdlog -Pattern 'max_volume:\s*(-?\d+(?:\.\d+)?) dB' | Select-Object -First 1
if ($m) { $mv = [double]$m.Matches[0].Groups[1].Value; $gain = -1.0 - $mv } else { $mv = 'n/a'; $gain = 0.0 }
$g = $gain.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$af = "$trim,volume=${g}dB"
cmd /c "`"$ffexe`" -hide_banner -y $seek -i `"$src`" -af `"$af`" -ar 44100 -c:a pcm_s16le `"$out`" 2> nul"
if (Test-Path $out) {
  $sz = [math]::Round((Get-Item $out).Length/1MB,2)
  Write-Output ("OK  {0,-40} maxvol={1} gain={2:N1}dB  {3}MB" -f $outName, $mv, $gain, $sz)
} else { Write-Output ("FAIL {0}" -f $outName) }
