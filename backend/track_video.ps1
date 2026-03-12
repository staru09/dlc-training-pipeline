$VideoName = "3e382fa1-d716-4cfe-abec-304e36fbc0d0.mp4"
$ApiUrl = "https://dlc-api-service-405737646974.europe-west4.run.app"

Write-Host "🚀 Submitting video ($VideoName)..." -ForegroundColor Cyan

# 1. Submit the job using curl
$submitResponse = curl.exe -s -X POST "$ApiUrl/infer/gcs" -F "video_name=$VideoName"
$submitJson = $submitResponse | ConvertFrom-Json
$taskId = $submitJson.task_id

Write-Host "✅ Job submitted! Task ID: $taskId" -ForegroundColor Green
Write-Host "----------------------------------------"

# 2. Live tracking loop
$seenLogs = 0

while ($true) {
    # Fetch job status
    $pollResponse = curl.exe -s "$ApiUrl/jobs/$taskId"
    $job = $pollResponse | ConvertFrom-Json
    
    # Print the current status tag at the top of the terminal
    Write-Host "`r[ $(Get-Date -Format 'HH:mm:ss') ] Status: " -NoNewline
    if ($job.status -eq "completed") { Write-Host "COMPLETED" -ForegroundColor Green }
    elseif ($job.status -eq "failed") { Write-Host "FAILED" -ForegroundColor Red }
    else { Write-Host "$($job.status.ToUpper()) " -ForegroundColor Yellow -NoNewline; Write-Host "($($job.elapsed_seconds)s)"}

    # Print any new log lines that have appeared since we last checked
    for ($i = $seenLogs; $i -lt $job.logs.Length; $i++) {
        $log = $job.logs[$i]
        $color = if ($log.level -eq "ERROR") {"Red"} else {"DarkGray"}
        Write-Host "  > [$($log.level)] $($log.message)" -ForegroundColor $color
    }
    $seenLogs = $job.logs.Length

    # Exit the loop if finished
    if ($job.status -eq "completed" -or $job.status -eq "failed") {
        Write-Host "`nResult files saved to GCS:" -ForegroundColor Cyan
        $job.result_files | ForEach-Object { Write-Host "  - $_" }
        break
    }

    # Wait 3 seconds before polling again
    Start-Sleep -Seconds 3
}
