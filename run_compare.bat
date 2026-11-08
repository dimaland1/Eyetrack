@echo off
cd /d "%~dp0"
title Eyetrack - Comparaison Scientifique IRIS vs NET
echo ========================================================================
echo  Comparaison Simultanee Iris (MVP) vs Net (ONNX) sur les memes cibles...
echo ========================================================================
python validate.py --compare
echo.
pause
