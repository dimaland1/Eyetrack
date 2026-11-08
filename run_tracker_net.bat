@echo off
cd /d "%~dp0"
title Eyetrack - Mode Reseau ONNX (Gaming avec OpenTrack)
echo ========================================================================
echo  Lancement du tracker en MODE NET (Zhang 2018 + MobileOne-S0)
echo.
echo  Raccourcis en cours de jeu / sur le bureau :
echo    [O] : Activer / masquer l'overlay transparent in-game
echo    [P] : Basculer de profil (default, ets2, flight_sim, racing)
echo    [C] : Calibrer (5 ou 9 points avec mouvement de tete)
echo    [R] : Recentrer (zero pose + regard)
echo    [G] : Activer / couper l'influence du regard sur la camera
echo    [S] : Sauvegarder la configuration dans le profil actif
echo ========================================================================
python eyetrack.py --gaze-mode net --profile default
echo.
pause
