#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/9] Checking data splits..."
python src/data.py

echo "[2/9] Building or reusing embeddings..."
python src/embed.py

echo "[3/9] Running baselines..."
python src/baselines.py

echo "[4/9] Running the adaptation agent..."
python src/agent.py

echo "[5/9] Comparing agent, grid, random, and fixed search..."
python src/compare.py

echo "[6/9] Generating plots..."
python src/plots.py

echo "[7/9] Running validation error analysis..."
python src/error_analysis.py

echo "[8/9] Running tests..."
python -m pytest tests -q

echo "[9/9] Handling the held-out test result..."
if [[ -f results/final_test.json ]]; then
    echo "Existing results/final_test.json found; skipping final test evaluation."
else
    echo "No saved final test result found. Running final evaluation once."
    python src/final_eval.py
fi

echo
echo "Pipeline completed successfully."