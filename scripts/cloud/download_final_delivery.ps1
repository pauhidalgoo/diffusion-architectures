param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [Parameter(Mandatory = $true)]
    [string]$KnownHostsFile,

    [string]$User = "root",
    [string]$RemoteRoot = "/workspace/hemera/delivery",
    [string]$Destination = "artifacts/final-delivery"
)

$ErrorActionPreference = "Stop"
$destinationPath = [System.IO.Path]::GetFullPath(
    (Join-Path (Get-Location) $Destination)
)
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

$files = @(
    "SHA256SUMS",
    "delivery-manifest.json",
    "hemera-final-source.tgz",
    "hemera-final-checkpoints.tar",
    "hemera-final-results.tar"
)
$common = @(
    "-P", "$Port",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=8",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=$KnownHostsFile"
)

foreach ($file in $files) {
    $remote = "${User}@${HostName}:${RemoteRoot}/${file}"
    Write-Host "Downloading $file"
    & scp @common $remote $destinationPath
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed for $file with exit code $LASTEXITCODE"
    }
}

$verifier = Join-Path $PSScriptRoot "verify_final_delivery.py"
$verification = Join-Path $destinationPath "local-verification.json"
& python $verifier $destinationPath --output $verification
if ($LASTEXITCODE -ne 0) {
    throw "Final delivery verification failed with exit code $LASTEXITCODE"
}

Write-Host "Verified final delivery at $destinationPath"
