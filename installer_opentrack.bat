@echo off
cd /d "%~dp0"
title Installation automatique d'OpenTrack
echo ========================================================================
echo  Installation d'OpenTrack via winget (Officiel, Gratuit et Open Source)
echo ========================================================================
echo.
echo Ce script va installer OpenTrack sur votre ordinateur.
echo.
pause
winget install --id opentrack.opentrack -e --source winget
echo.
echo ========================================================================
echo  Installation terminee ! Vous trouverez OpenTrack dans votre menu Demarrer.
echo ========================================================================
pause
