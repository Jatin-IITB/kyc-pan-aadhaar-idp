@echo off
REM ===================================================================
REM  Full overnight run: rebuild the corpus with a validated font
REM  environment, re-measure everything, and leave a transcript.
REM
REM      tools\run_overnight.cmd
REM
REM  Start it and walk away. Nothing here needs input. Each stage runs
REM  independently; a failure is recorded, not fatal.
REM
REM  Why this exists: on Windows the forge previously rendered every
REM  typographic kind with PIL's default bitmap font, because
REM  tools/forge/fonts.py carried no Windows candidate paths. That gave
REM  86.7% genuine FPR and made font_swap a no-op. The fonts are fixed;
REM  the corpus generated under the broken environment must be rebuilt
REM  before any number from it means anything.
REM ===================================================================
setlocal
cd /d "%~dp0.."

if not exist eval mkdir eval
set "LOG=%CD%\eval\overnight.log"
if exist "%LOG%" del "%LOG%"

python -c "import sys,pathlib;raise SystemExit(0 if pathlib.Path(sys.prefix).resolve()==(pathlib.Path.cwd()/'.venv').resolve() else 1)"
if errorlevel 1 (
  echo [ABORT] venv not active. Run:  .venv\Scripts\activate.bat
  exit /b 1
)

echo Transcript: %LOG%
echo Started:    %DATE% %TIME%
echo Started: %DATE% %TIME%>>"%LOG%"

REM ---- 0. font environment (everything downstream depends on this) ---
echo.
echo === 0/7 font environment
echo ============ 0/7 font environment ============>>"%LOG%"
python -c "from tools.forge.fonts import font_environment,check_font_environment as c;[print(f'{k:8s} -> {v}') for k,v in font_environment().items()];p=c();print('PROBLEMS: '+'; '.join(p)) if p else print('font environment OK');raise SystemExit(1 if p else 0)">>"%LOG%" 2>&1
set E0=%errorlevel%
call :tick %E0%
if not "%E0%"=="0" (
  echo.
  echo [ABORT] Font environment is broken. Regenerating now would produce
  echo         another invalid corpus. See %LOG%.
  echo [ABORT] font environment broken>>"%LOG%"
  goto :summary_early
)

REM ---- 1. rebuild the corpus ----------------------------------------
echo === 1/7 regenerate datasets ^(SLOW — this is the long one^)
echo ============ 1/7 regenerate ============>>"%LOG%"
python -m tools.eval.run_eval --regen --no-extraction>>"%LOG%" 2>&1
set E1=%errorlevel%
call :tick %E1%

REM ---- 2. unit tests -------------------------------------------------
echo === 2/7 unit tests
echo ============ 2/7 unit tests ============>>"%LOG%"
python -m pytest tests/unit -q>>"%LOG%" 2>&1
set E2=%errorlevel%
call :tick %E2%

REM ---- 3. forensics gates -------------------------------------------
echo === 3/7 forensics gates
echo ============ 3/7 forensics gates ============>>"%LOG%"
python -m tools.eval.run_eval --no-extraction --check>>"%LOG%" 2>&1
set E3=%errorlevel%
call :tick %E3%

REM ---- 4. FPR attribution (both splits) ------------------------------
echo === 4/7 genuine-FPR attribution
echo ============ 4/7 FPR attribution ============>>"%LOG%"
python -m tools.eval.fpr_attribution --root data/holdout --verbose>>"%LOG%" 2>&1
python -m tools.eval.fpr_attribution --root data/tuning>>"%LOG%" 2>&1
set E4=%errorlevel%
call :tick %E4%

REM ---- 5. RAG ablation ----------------------------------------------
echo === 5/7 RAG retrieval ablation
echo ============ 5/7 RAG ablation ============>>"%LOG%"
python -m tools.eval.rag_eval>>"%LOG%" 2>&1
set E5=%errorlevel%
call :tick %E5%

REM ---- 6. W16 evasion curve + blind probe ----------------------------
echo === 6/7 W16 evasion curve and blind JPEG-ghost
echo ============ 6/7 W16 ============>>"%LOG%"
python -m tools.eval.evasion_probe>>"%LOG%" 2>&1
python -m tools.eval.double_jpeg_probe --mode blind --stat diff>>"%LOG%" 2>&1
set E6=%errorlevel%
call :tick %E6%

REM ---- 7. double-JPEG diagnostics ------------------------------------
echo === 7/7 double-JPEG dq + oracle
echo ============ 7/7 double-JPEG dq/oracle ============>>"%LOG%"
python -m tools.eval.double_jpeg_probe --mode dq>>"%LOG%" 2>&1
python -m tools.eval.double_jpeg_probe --mode oracle>>"%LOG%" 2>&1
set E7=%errorlevel%
call :tick %E7%

:summary_early
echo.
echo ============================================================
echo   SUMMARY
echo ============================================================
echo ============ SUMMARY ============>>"%LOG%"
call :report "0 font environment   " %E0%
call :report "1 regenerate corpus  " %E1%
call :report "2 unit tests         " %E2%
call :report "3 forensics gates    " %E3%
call :report "4 FPR attribution    " %E4%
call :report "5 RAG ablation       " %E5%
call :report "6 W16 probes         " %E6%
call :report "7 double-JPEG diag   " %E7%
echo Finished: %DATE% %TIME%
echo Finished: %DATE% %TIME%>>"%LOG%"

echo.
echo --- headline numbers ---------------------------------------
findstr /C:"genuine FPR" /C:"flagged (" /C:"ALL GATES" /C:"actual=" /C:"passed" /C:"font environment" /C:"Traceback" "%LOG%"
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
