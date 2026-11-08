@echo off
cd /d "%~dp0"
title Eyetrack - Mode Iris MVP
echo ========================================================
echo  Lancement du tracker en MODE IRIS (MVP Ratios Iris)...
echo ========================================================
python eyetrack.py --gaze-mode iris
echo.
pause
