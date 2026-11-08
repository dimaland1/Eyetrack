@echo off
cd /d "%~dp0"
title Eyetrack - Diagnostic du regard
echo ========================================================================
echo  Diagnostic : le reseau suit-il vos yeux ?
echo  Gardez la tete IMMOBILE et suivez la cible avec les YEUX seulement.
echo ========================================================================
python diag_regard.py
echo.
pause
