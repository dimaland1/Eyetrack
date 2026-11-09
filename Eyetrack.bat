@echo off
cd /d "%~dp0"
title Eyetrack

:menu
cls
echo ==============================================================
echo                          EYETRACK
echo ==============================================================
echo.
echo   Tracker
echo     1. Lancer le tracker            (mode reseau, profil default)
echo     2. Lancer OpenTrack + le tracker
echo     3. Lancer le tracker            (profil ETS2)
echo     4. Lancer le tracker            (mode iris)
echo.
echo   Diagnostics
echo     5. Diagnostic du regard         (tete immobile, yeux seulement)
echo     6. Diagnostic de stabilite      (10 s en fixant le centre)
echo     7. Validation de precision      (9 cibles)
echo     9. Tests automatiques           (sens des axes, envoi OpenTrack)
echo.
echo   Installation
echo     8. Installer OpenTrack          (winget)
echo.
echo     Q. Quitter
echo.
choice /c 123456789Q /n /m "Votre choix : "
if errorlevel 10 goto :eof
if errorlevel 9 goto tests
if errorlevel 8 goto install
if errorlevel 7 goto validate
if errorlevel 6 goto stab
if errorlevel 5 goto regard
if errorlevel 4 goto iris
if errorlevel 3 goto ets2
if errorlevel 2 goto opentrack
if errorlevel 1 goto tracker

:tracker
python main.py --gaze-mode net --profile default
goto fin

:opentrack
if exist "C:\Program Files (x86)\opentrack\opentrack.exe" (
    start "" "C:\Program Files (x86)\opentrack\opentrack.exe"
) else if exist "C:\Program Files\opentrack\opentrack.exe" (
    start "" "C:\Program Files\opentrack\opentrack.exe"
) else (
    echo OpenTrack introuvable : utilisez l'option 8 pour l'installer.
)
python main.py --gaze-mode net --profile default
goto fin

:ets2
python main.py --profile ets2
goto fin

:iris
python main.py --gaze-mode iris
goto fin

:regard
python tools\diag_regard.py
goto fin

:stab
python tools\diag_stabilite.py --profile default --gaze-mode net
goto fin

:validate
python tools\validate.py
goto fin

:tests
python -m unittest discover -s tests -v
goto fin

:install
winget install --id opentrack.opentrack -e --source winget
goto fin

:fin
echo.
pause
goto menu
