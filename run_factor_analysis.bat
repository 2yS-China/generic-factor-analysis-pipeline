@echo off
rem Simple wrapper to run the factor analysis pipeline on Windows.
rem Usage: run_factor_analysis.bat <input-data-file> [output-prefix]

if "%~1"=="" (
    echo Usage: %~nx0 ^<input-data-file^> [output-prefix]
    exit /b 1
)
set "INPUT=%~1"

if "%~2"=="" (
    set "PREFIX=results\run1"
) else (
    set "PREFIX=%~2"
)

python factor_analysis_pipeline.py ^
  --input "%INPUT%" ^
  --id-col Year ^
  --output-prefix "%PREFIX%"
pause
