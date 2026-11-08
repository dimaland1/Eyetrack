@echo off
cd /d "%~dp0"
title Eyetrack + OpenTrack Launcher

echo ========================================================================
echo  Demarrage d'OpenTrack et du Tracker Eyetrack...
echo ========================================================================

rem Lancer OpenTrack s'il est present
if exist "C:\Program Files (x86)\opentrack\opentrack.exe" (
    start "" "C:\Program Files (x86)\opentrack\opentrack.exe"
)

rem Lancer le tracker Python
python eyetrack.py --gaze-mode net --profile default

pause
