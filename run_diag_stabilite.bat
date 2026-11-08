@echo off
cd /d "%~dp0"
title Eyetrack - Diagnostic de stabilite
echo  Fixez le point au centre, tete immobile, pendant 10 secondes.
python diag_stabilite.py --profile default --gaze-mode net
echo.
pause
