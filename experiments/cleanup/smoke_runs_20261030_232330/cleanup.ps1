$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath 'F:\A_Edinburgh\PhysGauss').Path
$runsRoot = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot 'runs')).Path
$runsPrefix = $runsRoot.TrimEnd('\') + '\'
$relativeTargets = @(
    'pidg_c1_material\_smoke',
    'pidg_centroid_tuning\_smoke',
    'pidg_color_repair\_smoke',
    'pidg_coverage_isolated\_smoke',
    'pidg_coverage_tuning\_smoke',
    'pidg_isotropic_tuning\smoke_flags_off',
    'pidg_isotropic_tuning\smoke_I0_T3_control',
    'pidg_isotropic_tuning\smoke_I2_iso_1e3',
    'pidg_kinematic_resplit\_smoke',
    'pidg_kinematic_tuning\smoke_flags_off',
    'pidg_kinematic_tuning\smoke_V3_frame_time',
    'pidg_kinematic_tuning\smoke_V4_frame_time_kinematic',
    'pidg_metric_tuning\smoke_flags_off',
    'pidg_metric_tuning\smoke_residual_perceptual',
    'pidg_mlp_capacity\smoke_W256_D0',
    'pidg_mlp_capacity\smoke_W256_D2',
    'pidg_mlp_capacity\smoke_W512_D0',
    'pidg_mlp_capacity\smoke_W512_D2',
    'pidg_random_frame_mask\smoke_drop90_seed0',
    'pidg_smoke',
    'pidg_v3_iso_scratch\_smoke',
    'pidg_v3_isotropic\_smoke',
    'smoke'
)

function Get-SafeTreeEntries([string]$Root) {
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    $pending.Push($Root)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        foreach ($entry in Get-ChildItem -LiteralPath $current -Force) {
            $entry
            if ($entry.PSIsContainer -and ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
                $pending.Push($entry.FullName)
            }
        }
    }
}

# Resolve and validate every absolute target before any deletion.
$targets = @()
$filesByTarget = @{}
$linksByTarget = @{}
foreach ($relative in $relativeTargets) {
    $item = Get-Item -LiteralPath (Join-Path $runsRoot $relative) -Force
    $absolute = (Resolve-Path -LiteralPath $item.FullName).Path
    if (-not $item.PSIsContainer -or -not $absolute.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside intended runs directory: $absolute"
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse target: $absolute" }
    $contents = @(Get-SafeTreeEntries $absolute)
    $links = @($contents | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 })
    foreach ($link in $links) {
        if (-not $link.PSIsContainer -or $link.LinkType -ne 'Junction') { throw "Unsupported link: $($link.FullName)" }
        foreach ($linkTarget in $link.Target) {
            $normalized = [IO.Path]::GetFullPath($linkTarget)
            if (-not $normalized.StartsWith($absolute + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "Link points outside its disposable target: $($link.FullName)"
            }
        }
    }
    $linksByTarget[$absolute] = $links
    $filesByTarget[$absolute] = @($contents | Where-Object { -not $_.PSIsContainer })
    $targets += $absolute
}
if ($targets.Count -ne 23) { throw 'Unexpected target count' }
$pythonProcesses = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' })
foreach ($process in $pythonProcesses) {
    foreach ($target in $targets) {
        if ($process.CommandLine -and $process.CommandLine.IndexOf($target, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            throw "Target still used by Python process $($process.ProcessId): $target"
        }
    }
}

function Get-RetainedSnapshot {
    $records = @()
    foreach ($file in Get-SafeTreeEntries $runsRoot) {
        $isLink = ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        if ($file.PSIsContainer -and -not $isLink) { continue }
        $disposable = $false
        foreach ($target in $targets) {
            if ($file.FullName.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) { $disposable = $true; break }
        }
        if (-not $disposable) {
            if ($isLink) {
                foreach ($linkTarget in $file.Target) {
                    foreach ($target in $targets) {
                        if ($linkTarget.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) {
                            throw "Retained run link depends on a disposable target: $($file.FullName)"
                        }
                    }
                }
                $records += '{0}|LINK|{1}|{2}' -f $file.FullName.Substring($runsPrefix.Length), ($file.Target -join ','), $file.LastWriteTimeUtc.Ticks
            } else {
                $records += '{0}|{1}|{2}' -f $file.FullName.Substring($runsPrefix.Length), $file.Length, $file.LastWriteTimeUtc.Ticks
            }
        }
    }
    return @($records | Sort-Object)
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$auditRoot = Join-Path $workspaceRoot ('experiments\cleanup\smoke_runs_' + $stamp)
New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $auditRoot 'cleanup.ps1')
$protectedBefore = @(Get-RetainedSnapshot)
$protectedBefore | Set-Content -LiteralPath (Join-Path $auditRoot 'retained_files_before.txt') -Encoding UTF8
$keepExtensions = @('.json', '.jsonl', '.log', '.txt', '.md', '.yaml', '.yml', '.py', '.html', '.0')
$archiveRecords = @()
$targetRecords = @()
[long]$totalBytes = 0
[long]$totalFiles = 0
foreach ($target in $targets) {
    $fileRecords = @()
    [long]$bytes = 0
    foreach ($file in $filesByTarget[$target]) {
        $relativeFile = $file.FullName.Substring($runsPrefix.Length)
        $bytes += $file.Length
        $fileRecords += [ordered]@{ path=$relativeFile; bytes=$file.Length }
        if (($keepExtensions -contains $file.Extension.ToLowerInvariant() -or $file.Name -eq 'cfg_args') -and $file.Length -le 8MB) {
            $destination = Join-Path (Join-Path $auditRoot 'metadata') $relativeFile
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $destination
            $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
            if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $sourceHash) { throw 'Metadata archive checksum mismatch' }
            $archiveRecords += [ordered]@{ path=$relativeFile; bytes=$file.Length; sha256=$sourceHash }
        }
    }
    $totalBytes += $bytes
    $totalFiles += $fileRecords.Count
    $linkRecords = @($linksByTarget[$target] | ForEach-Object { [ordered]@{path=$_.FullName; target=$_.Target; type=$_.LinkType} })
    $targetRecords += [ordered]@{ path=$target; bytes=$bytes; file_count=$fileRecords.Count; files=$fileRecords; internal_junctions=$linkRecords }
}
$manifest = [ordered]@{
    status='prepared'; started_utc=[DateTime]::UtcNow.ToString('o'); runs_root=$runsRoot;
    target_count=$targets.Count; total_files=$totalFiles; deleted_logical_bytes=$totalBytes;
    logical_GB=[Math]::Round($totalBytes/1e9,3); logical_GiB=[Math]::Round($totalBytes/1GB,3);
    protected_retained_files=$protectedBefore.Count;
    retained_controlled_experiments='pidg_constitutive/m10_smoke_*_controlled: reported synthetic ablations';
    archive_policy='Configuration/metrics/logs/code <=8MiB archived; test weights and images removed';
    archived_metadata=$archiveRecords; targets=$targetRecords
}
$manifestPath = Join-Path $auditRoot 'manifest.json'
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

# Native PowerShell deletion, only against the resolved and validated paths above.
foreach ($target in $targets) {
    if (-not $target.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion target escaped runs root' }
    foreach ($link in $linksByTarget[$target]) {
        $currentLink = Get-Item -LiteralPath $link.FullName -Force
        if (($currentLink.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) { throw 'Expected junction changed' }
        # Nonrecursive Directory.Delete removes the junction itself, not its target.
        [IO.Directory]::Delete($currentLink.FullName)
    }
    Remove-Item -LiteralPath $target -Recurse -Force
    if (Test-Path -LiteralPath $target) { throw "Target still exists: $target" }
}
$protectedAfter = @(Get-RetainedSnapshot)
$difference = @(Compare-Object -ReferenceObject $protectedBefore -DifferenceObject $protectedAfter)
if ($difference.Count) { throw 'Retained runs file metadata changed; inspect cleanup archive' }
$manifest.status='completed'
$manifest.finished_utc=[DateTime]::UtcNow.ToString('o')
$manifest.retained_file_paths_sizes_and_timestamps_unchanged=$true
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
[ordered]@{ status='completed'; deleted_directories=$targets.Count; deleted_files=$totalFiles;
    logical_GB=$manifest.logical_GB; logical_GiB=$manifest.logical_GiB;
    archived_metadata_files=$archiveRecords.Count; retained_files_verified=$protectedAfter.Count;
    manifest=$manifestPath } | ConvertTo-Json
