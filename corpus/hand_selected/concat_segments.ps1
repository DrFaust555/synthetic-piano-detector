# Join the segments of one piano out of a raw download (several pianos in one video, or talk between
# the playing): 8-ms fades at every cut, 44.1 kHz PCM-16, written next to the source in audio/_raw/.
#   powershell -File corpus/hand_selected/concat_segments.ps1 <rawId> "<s0-e0,s1-e1,...>" <outName> [-AudioRoot <folder>]
# Segments (seconds of audio/_raw/<rawId>.wav) used for the hand-selected set:
#   03  -> 03a.wav 0-9,20-34,45-57,60-65,67-71         03b.wav 9-20,34-45,57-60,65-67
#   56  -> 56a.wav 0-3,7-10                            56b.wav 4-7,10-13
#   57  -> 57a.wav 0-4,10-14                           57b.wav 5-10,14-18
#   63  -> 63cat.wav 63-154,185-277
#   64  -> 64a.wav 1-61,126-227,332-433                64b.wav 64-123,230-329,436-545
#   65  -> 65cat.wav 88-114,664-690,1000-1022
#   77  -> 77a.wav 23-33,46-54                         77b.wav 34-44,56-69
#   128 -> 128cat.wav 15-21,23-33
#   160 -> 160cat.wav 154-176,216-266
# The joined file is then trimmed and normalised by trim_normalize.ps1 (rawId = e.g. 03a).
param([string]$rawId, [string]$segments, [string]$outName, [string]$AudioRoot = $env:SPD_AUDIO)
$ErrorActionPreference = 'Continue'
# ffmpeg from PATH, or from the folder in the environment variable FFMPEG_DIR
$ffexe = if ($env:FFMPEG_DIR) { Join-Path $env:FFMPEG_DIR "ffmpeg.exe" } else { "ffmpeg" }
$raw   = Join-Path $AudioRoot "audio\_raw"
$src   = Join-Path $raw "$rawId.wav"
$log   = Join-Path $env:TEMP "concat_$rawId.txt"
$inv   = [System.Globalization.CultureInfo]::InvariantCulture

function Build($segs, $outname) {
  $parts = @(); $labels = @()
  for ($i = 0; $i -lt $segs.Count; $i++) {
    $s = $segs[$i][0]; $e = $segs[$i][1]
    $fo = ([double]($e - $s - 0.008)).ToString($inv)
    $lab = "s$i"
    $parts  += "[0:a]atrim=${s}:${e},asetpts=N/SR/TB,afade=t=in:st=0:d=0.008,afade=t=out:st=${fo}:d=0.008[$lab]"
    $labels += "[$lab]"
  }
  $fc = (($parts -join ";") + ";" + ($labels -join "") + "concat=n=$($segs.Count):v=0:a=1[out]")
  $fcfile = Join-Path $env:TEMP "fc_$rawId.txt"
  Set-Content -Path $fcfile -Value $fc -Encoding ascii -NoNewline
  $out = Join-Path $raw $outname
  cmd /c "`"$ffexe`" -hide_banner -y -i `"$src`" -filter_complex_script `"$fcfile`" -map `"[out]`" -ar 44100 -c:a pcm_s16le `"$out`" 2> `"$log`""
  if (Test-Path $out) { Write-Output "built $outname" } else { Write-Output "FAIL $outname"; Get-Content $log | Select-Object -Last 15 }
}

$segs = @($segments -split ',' | ForEach-Object { , @($_ -split '-' | ForEach-Object { [double]$_ }) })
Build $segs $outName
