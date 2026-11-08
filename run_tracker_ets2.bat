@echo off
cd /d "%~dp0"
title Eyetrack - Profil ETS2 / Camion (OpenTrack)
echo ========================================================================
echo  Lancement avec Profil ETS2 (Grand angle pour retroviseurs)
echo.
echo  Raccourcis :
echo    [O] : Activer / masquer l'overlay transparent
echo    [P] : Changer de profil
echo    [R] : Recentrer (regard droit devant)
echo    [C] : Calibrer
echo ========================================================================
python eyetrack.py --profile ets2
echo.
pause
