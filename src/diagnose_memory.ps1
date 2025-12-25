
$mem = Get-CimInstance Win32_OperatingSystem
$total = $mem.TotalVisibleMemorySize
$free = $mem.FreePhysicalMemory
$used = $total - $free

Write-Host "--- Overall Memory ---"
Write-Host ("Total RAM: " + [math]::Round($total/1024, 2) + " MB")
Write-Host ("Used RAM: " + [math]::Round($used/1024, 2) + " MB")
Write-Host ("Used %: " + [math]::Round(($used/$total)*100, 2) + "%")

$perf = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
Write-Host "`n--- Kernel and Cache ---"
Write-Host ("Non-paged Pool: " + [math]::Round($perf.PoolNonpagedBytes/1MB, 2) + " MB")
Write-Host ("Paged Pool:     " + [math]::Round($perf.PoolPagedBytes/1MB, 2) + " MB")
Write-Host ("System Cache:   " + [math]::Round($perf.SystemCacheResidentBytes/1MB, 2) + " MB")

Write-Host "`n--- Top 20 Processes by Working Set ---"
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 20 | ForEach-Object {
    Write-Host ($_.Name.PadRight(25) + " (" + $_.Id.ToString().PadLeft(6) + "): " + [math]::Round($_.WorkingSet/1MB, 2).ToString().PadLeft(8) + " MB")
}

Write-Host "`n--- Orphaned/Relevant Processes Search (Stockfish, Python) ---"
$relevant = Get-Process | Where-Object { $_.Name -like "*stockfish*" -or $_.Name -like "*python*" }
if ($relevant) {
    $relevant | ForEach-Object {
        Write-Host ($_.Name.PadRight(25) + " (" + $_.Id.ToString().PadLeft(6) + "): " + [math]::Round($_.WorkingSet/1MB, 2).ToString().PadLeft(8) + " MB")
    }
} else {
    Write-Host "No Stockfish or Python processes found."
}
