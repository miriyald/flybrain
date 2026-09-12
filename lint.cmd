@echo off
setlocal
set TARGETS=flylab tests 01_get_the_data.py 02_explore.py 03_build_circuit.py 04_odour_code.py 05_learn.py
set FAILED=0

echo === ruff ===
python -m ruff check %TARGETS% || set FAILED=1

echo === flake8 ===
python -m flake8 --max-line-length=140 --extend-ignore=E203,W503 %TARGETS% || set FAILED=1

echo === pylint ===
python -m pylint --rcfile=pyproject.toml %TARGETS% || set FAILED=1

echo === mypy ===
python -m mypy flylab tests || set FAILED=1

if %FAILED%==1 (
    echo.
    echo LINT FAILED
    exit /b 1
)
echo.
echo LINT CLEAN
exit /b 0
