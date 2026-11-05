"""Quick project status checker."""
import os

files = {
    'data/processed.csv': 'T1',
    'results/preds_garch.csv': 'T3',
    'results/preds_egarch.csv': 'T3',
    'results/preds_gjr.csv': 'T3',
    'results/preds_lstm.csv': 'A3',
    'results/preds_naive.csv': 'A4',
    'results/metrics.csv': 'A4',
    'results/metrics_by_period.csv': 'A4',
    'results/dm_test.csv': 'A4',
    'results/next_day_forecast.csv': 'A4',
    'results/next_day_garch.csv': 'T3',
    'results/next_day_lstm.csv': 'A3',
    'results/lstm_history.csv': 'A2',
    'results/garch_params.csv': 'T2',
    'results/garch_fit_stats.csv': 'T2',
    'results/data_summary.csv': 'T1',
    'docs/report_facts.md': 'A5',
    'docs/challenges_log.md': 'T0',
    'docs/part1_facts.md': 'T5',
    'README.md': 'A6',
}

print('FILE STATUS:')
for f, task in sorted(files.items(), key=lambda x: x[1]):
    exists = os.path.exists(f)
    status = 'OK' if exists else 'MISSING'
    size = os.path.getsize(f) if exists else 0
    size_str = " (" + str(size) + " bytes)" if exists else ""
    print("  [" + status + "] (" + task + ") " + f + size_str)

fig_names = [
    (1, "price"), (2, "returns"), (3, "rv5_splits"), (4, "news_impact"),
    (5, "garch_pred_vs_actual"), (6, "lstm_loss"), (7, "lstm_pred_vs_actual"),
    (8, "all_models_pred_vs_actual"), (9, "metric_bars"), (10, "covid_zoom")
]
print('\nFIGURES:')
for i, name in fig_names:
    f = "results/figures/fig" + str(i).zfill(2) + "_" + name + ".png"
    exists = os.path.exists(f)
    print("  [" + ("OK" if exists else "MISSING") + "] " + f)

print("\nTESTS:")
import subprocess, sys
result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q"], capture_output=True, text=True)
print(result.stdout)
if result.returncode == 0:
    print("All tests PASSED")
else:
    print("Some tests FAILED")
    print(result.stderr)
