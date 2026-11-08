@echo off
cd /d "%~dp0"
title Eyetrack - Validation 9 Cibles
echo ========================================================
echo  Lancement du protocole de validation (validate.py)...
echo ========================================================
python validate.py
echo.
pause
