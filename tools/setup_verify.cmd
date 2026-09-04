@echo off
REM ===================================================================
REM  Unattended setup verification for Windows.
REM
REM  Runs every stage in order, never aborts on failure, and writes a
REM  full transcript to eval\setup_verify.log plus a PASS/FAIL summary
REM  at the end. Safe to start and walk away from.
REM
REM      tools\setup_verify.cmd
REM
REM  Requires: the venv ACTIVATED (prompt shows "(.venv)").
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set LOG=eval\setup_verify.log
if not exist eval mkdir eval
if exist "%LOG%" del "%LOG%"

REM --- refuse to run against the wrong interpreter -------------------
python -c "import sys,pathlib; sys.exit(0 if pathlib.Path(sys.prefix).resolve()==pathlib.Path('.venv').resolve() else 1)" 2>nul
if errorlevel 1 (
  echo [ABORT] venv is not active. Run:  .venv\Scripts\activate.bat
  echo [ABORT] venv is not active.>>"%LOG%"
  exit /b 1
)

call :banner "environment"
python -c "import sys,cv2,torch,numpy;print('python',sys.version.split()[0]);print('opencv',cv2.__version__);print('torch',torch.__version__);print('numpy',numpy.__version__)" >>"%LOG%" 2>&1
python -c "import cv2;cv2.SIFT_create();print('SIFT OK')" >>"%LOG%" 2>&1
set SIFT=%errorlevel%

call :stage "1/6 regenerate datasets (slow, several minutes)" ^
  "python -m tools.eval.run_eval --regen --no-extraction"
set S1=%ERR%

call :stage "2/6 unit tests" ^
  "python -m pytest tests/unit -q"
set S2=%ERR%

call :stage "3/6 forensics gates" ^
  "python -m tools.eval.run_eval --no-extraction --check"
set S3=%ERR%

call :stage "4/6 RAG unit tests" ^
  "python -m pytest -q tests/unit/test_rag_metrics.py tests/unit/test_rag_faithfulness.py"
set S4=%ERR%

call :stage "5/6 RAG retrieval ablation (downloads ~500MB first run)" ^
  "python -m tools.eval.rag_eval"
set S5=%ERR%

call :stage "6/6 W16 open experiment: blind JPEG-ghost, difference statistic" ^
  "python -m tools.eval.double_jpeg_probe --mode blind --stat diff"
set S6=%ERR%

REM --- summary ------------------------------------------------------
call :banner "SUMMARY"
call :report "SIFT available          " %SIFT%
call :report "1 regenerate datasets   " %S1%
call :report "2 unit tests            " %S2%
call :report "3 forensics gates       " %S3%
call :report "4 RAG unit tests        " %S4%
call :report "5 RAG ablation          " %S5%
call :report "6 W16 blind probe       " %S6%

echo.>>"%LOG%"
echo Expected: 148 passed / 1 failed on stage 2 (test_extract_from_bgr_unknown_>>"%LOG%"
echo triggers_fallback is a known pre-existing failure), and ALL GATES PASS on>>"%LOG%"
echo stage 3 with overall_recall 0.9222 and genuine_fpr 0.0.>>"%LOG%"

echo.
echo ============================================================
echo  Done. Full transcript: %LOG%
echo ============================================================
type "%LOG%" | findstr /C:"[STAGE]" /C:"[ OK ]" /C:"[FAIL]" /C:"ALL GATES" /C:"passed" /C:"actual="
exit /b 0

REM ------------------------------------------------------------------
:banner
echo.>>"%LOG%"
echo ============================================================>>"%LOG%"
echo   %~1>>"%LOG%"
echo ============================================================>>"%LOG%"
echo.
echo === %~1
goto :eof

:stage
call :banner "%~1"
echo [STAGE] %~1>>"%LOG%"
echo     ^> %~2>>"%LOG%"
%~2 >>"%LOG%" 2>&1
set ERR=%errorlevel%
if "%ERR%"=="0" (echo     ...ok) else (echo     ...exit code %ERR%)
goto :eof

:report
if "%~2"=="0" (
  echo [ OK ] %~1>>"%LOG%"
  echo [ OK ] %~1
) else (
  echo [FAIL] %~1 ^(exit %~2^)>>"%LOG%"
  echo [FAIL] %~1 ^(exit %~2^)
)
goto :eof
