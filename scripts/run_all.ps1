# End-to-end reproduction on Windows. Safe to re-run.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== 1/6  TigerGraph ==" -ForegroundColor Cyan
python -m scripts.setup_tigergraph --check
if ($LASTEXITCODE -eq 0) {
    python -m scripts.setup_tigergraph
} else {
    Write-Host "   No TigerGraph configured - using the offline store." -ForegroundColor Yellow
    python -m src.ingestion.build_graph
}

Write-Host "== 2/6  Generate the chained-reasoning stress set ==" -ForegroundColor Cyan
python -m src.eval.stress_set --n 60

Write-Host "== 3/6  Benchmark: 100 public questions ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark

Write-Host "== 4/6  Benchmark: 60 stress questions ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --stress

Write-Host "== 5/6  Benchmark: 50 hidden questions (submission) ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --hidden

Write-Host "== 6/6  Ablation: regex fast path + triage disabled ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --ablation
python -m src.eval.run_benchmark --stress --ablation

Write-Host "`nDone. Dashboard: python -m http.server 8000 -> http://localhost:8000/dashboard/" -ForegroundColor Green
