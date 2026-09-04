@echo off
REM ===================================================================
REM  Unattended setup verification for Windows.
REM
REM  Every stage runs inline and independently. Nothing aborts the run
REM  (except a broken font environment); each stage's exit code is
REM  captured and summarised at the end.
REM  Full transcript goes to eval\setup_verify.log.
REM
REM      tools\setup_verify.cmd
REM
REM  Requires the venv ACTIVATED (prompt shows "(.venv)").
REM ===================================================================
setlocal
cd /d "%~dp0.."

if not exist eval mkdir eval
set "LOG=%CD%\eval\setup_verify.log"
if exist "%LOG%" del "%LOG%"

REM --- refuse to run against the wrong interpreter -------------------
python -c "import sys,pathlib;raise SystemExit(0 if pathlib.Path(sys.prefix).resolve()==(pathlib.Path.cwd()/'.venv').resolve() else 1)"
if errorlevel 1 (
  echo [ABORT] venv is not active in this shell. Run:
  echo         .venv\Scripts\activate.bat
  exit /b 1
)

echo Writing transcript to %LOG%
echo.

REM ================= 0: font environment =============================
echo === 0  font environment
echo ============ 0 font environment ============>>"%LOG%"
python -c "from tools.forge.fonts import font_environment,check_font_environment as c;[print(f'{k:8s} -> {v}') for k,v in font_environment().items()];p=c();print('PROBLEMS: '+'; '.join(p)) if p else print('font environment OK');raise SystemExit(1 if p else 0)">>"%LOG%" 2>&1
set E0=%errorlevel%
if "%E0%"=="0" (echo     fonts ok) else (echo     fonts FAILED)
if not "%E0%"=="0" (
  echo.
  echo [ABORT] Font environment is broken. Regenerating now would produce
  echo         an invalid corpus. See %LOG%.
  echo [ABORT] font environment broken>>"%LOG%"
  goto :summary
)

REM ================= 1: environment ==================================
echo === 1  SIFT / torch / numpy
echo ============ 1 environment ============>>"%LOG%"
python -c "import sys,cv2,torch,numpy;print('python',sys.version.split()[0]);print('opencv',cv2.__version__);print('torch',torch.__version__);print('numpy',numpy.__version__)">>"%LOG%" 2>&1
python -c "import cv2;cv2.SIFT_create();print('SIFT OK')">>"%LOG%" 2>&1
set E1=%errorlevel%
if "%E1%"=="0" (echo     SIFT ok) else (echo     SIFT FAILED)

REM ================= 2: regenerate datasets ==========================
echo === 2/7 regenerate datasets ^(slow, several minutes^)
echo.>>"%LOG%"
echo ============ 2/7 regenerate datasets ============>>"%LOG%"
python -m tools.eval.run_eval --regen --no-extraction>>"%LOG%" 2>&1
set E2=%errorlevel%
call :tick %E2%

REM ================= 3: unit tests ===================================
echo === 3/7 unit tests
echo.>>"%LOG%"
echo ============ 3/7 unit tests ============>>"%LOG%"
python -m pytest tests/unit -q>>"%LOG%" 2>&1
set E3=%errorlevel%
call :tick %E3%

REM ================= 4: forensics gates ==============================
echo === 4/7 forensics gates
echo.>>"%LOG%"
echo ============ 4/7 forensics gates ============>>"%LOG%"
python -m tools.eval.run_eval --no-extraction --check>>"%LOG%" 2>&1
set E4=%errorlevel%
call :tick %E4%

REM ================= 5: RAG unit tests ===============================
echo === 5/7 RAG unit tests
echo.>>"%LOG%"
echo ============ 5/7 RAG unit tests ============>>"%LOG%"
python -m pytest -q tests/unit/test_rag_metrics.py tests/unit/test_rag_faithfulness.py>>"%LOG%" 2>&1
set E5=%errorlevel%
call :tick %E5%

REM ================= 6: RAG ablation =================================
echo === 6/7 RAG retrieval ablation ^(downloads ~500MB on first run^)
echo.>>"%LOG%"
echo ============ 6/7 RAG ablation ============>>"%LOG%"
python -m tools.eval.rag_eval>>"%LOG%" 2>&1
set E6=%errorlevel%
call :tick %E6%

REM ================= 7: W16 blind probe ==============================
echo === 7/7 W16 blind JPEG-ghost, difference statistic
echo.>>"%LOG%"
echo ============ 7/7 W16 blind probe ============>>"%LOG%"
python -m tools.eval.double_jpeg_probe --mode blind --stat diff>>"%LOG%" 2>&1
set E7=%errorlevel%
call :tick %E7%

REM ================= summary =========================================
:summary
echo.
echo ============================================================
echo   SUMMARY
echo ============================================================
echo.>>"%LOG%"
echo ============ SUMMARY ============>>"%LOG%"
call :report "0 font environment   " %E0%
call :report "1 SIFT available     " %E1%
call :report "2 regenerate datasets" %E2%
call :report "3 unit tests         " %E3%
call :report "4 forensics gates    " %E4%
call :report "5 RAG unit tests     " %E5%
call :report "6 RAG ablation       " %E6%
call :report "7 W16 blind probe    " %E7%

echo.
echo Stage 3 is EXPECTED to exit non-zero: one known pre-existing failure
echo ^(test_extract_from_bgr_unknown_triggers_fallback^). Look for
echo "153 passed, 1 failed" in the log.
echo.
echo --- key lines from the transcript --------------------------
findstr /C:"passed" /C:"ALL GATES" /C:"actual=" /C:"Error" /C:"error:" /C:"Traceback" /C:"font environment" "%LOG%"
echo ------------------------------------------------------------
echo Full transcript: %LOG%
exit /b 0

:tick
if "%~1"=="0" (echo     ...ok) else (echo     ...exit code %~1)
goto :eof

:report
if "%~2"=="" goto :eof
if "%~2"=="0" (
  echo [ OK ] %~1
  echo [ OK ] %~1>>"%LOG%"
) else (
  echo [FAIL] %~1 exit %~2
  echo [FAIL] %~1 exit %~2>>"%LOG%"
)
goto :eof
