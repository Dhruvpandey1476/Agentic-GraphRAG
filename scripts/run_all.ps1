# Reproduce every number in RESULTS.md from scratch. Windows twin of run_all.sh.
# Safe to re-run: benchmarks are checkpointed and resume where they stopped.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== 1/7  TigerGraph ==" -ForegroundColor Cyan
python -m scripts.setup_tigergraph --check 2>$null
if ($LASTEXITCODE -eq 0) {
    python -m scripts.setup_tigergraph
} else {
    Write-Host "   No reachable TigerGraph - using the offline store." -ForegroundColor Yellow
    Write-Host "   The benchmarks below REFUSE to run if TG_HOST is set but unreachable."
    python -m src.ingestion.build_graph
}

Write-Host "== 2/7  Generate the chained-reasoning set ==" -ForegroundColor Cyan
python -m src.eval.stress_set --n 60

Write-Host "== 3/7  Controlled comparison: 100 provided questions ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --suffix _final100

Write-Host "== 4/7  Controlled comparison: 60 chained questions ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --stress --suffix _finalstress

Write-Host "== 5/7  Production configuration: 100 provided questions ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --fast-path --suffix _final_fastpath

Write-Host "== 6/7  Hidden set (submission) ==" -ForegroundColor Cyan
python -m src.eval.run_benchmark --hidden --pipelines graphrag,agentic_graphrag --suffix _tg_hidden
python -m scripts.export_submission --suffix _tg_hidden

Write-Host "== 7/7  Regenerate RESULTS.md ==" -ForegroundColor Cyan
python -m scripts.make_report

Write-Host "`nDone. RESULTS.md, outputs/submission_hidden.jsonl, and the dashboard." -ForegroundColor Green
