@echo off
setlocal
cd /d "%~dp0"

echo [1/9] Checking data splits...
python src\data.py || goto :error

echo [2/9] Building or reusing embeddings...
python src\embed.py || goto :error

echo [3/9] Running baselines...
python src\baselines.py || goto :error

echo [4/9] Running the adaptation agent...
python src\agent.py || goto :error

echo [5/9] Comparing agent, grid, random, and fixed search...
python src\compare.py || goto :error

echo [6/9] Generating plots...
python src\plots.py || goto :error

echo [7/9] Running validation error analysis...
python src\error_analysis.py || goto :error

echo [8/9] Running tests...
python -m pytest tests -q || goto :error

echo [9/9] Handling the held-out test result...
if exist results\final_test.json (
    echo Existing results\final_test.json found; skipping final test evaluation.
) else (
    echo No saved final test result found. Running final evaluation once.
    python src\final_eval.py || goto :error
)

echo.
echo Pipeline completed successfully.
exit /b 0

:error
echo.
echo Pipeline failed with exit code %ERRORLEVEL%.
exit /b %ERRORLEVEL%
