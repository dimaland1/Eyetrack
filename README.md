# Eyetrack

Suivi de la tête et du regard avec une simple webcam, envoyé à OpenTrack (protocole
« UDP over network », port 4242) pour piloter la caméra dans les jeux et simulateurs.

## Installation

```bash
pip install -r requirements.txt
```

Les modèles (`models/`) sont téléchargés automatiquement au premier lancement.
OpenTrack s'installe depuis le menu (option 8) ou avec
`winget install --id opentrack.opentrack -e`.

## Lancement

Double-cliquer sur **`Eyetrack.bat`** et choisir dans le menu, ou en ligne de commande :

```bash
python main.py --gaze-mode net --profile default
```

Au premier lancement, la recherche du meilleur mode caméra prend ~20 s ; il est ensuite
mémorisé dans `data/camera_mode.json` (supprimer ce fichier pour relancer la recherche).

### Raccourcis (fenêtre « Eyetrack - Moniteur »)

| Touche | Action |
|---|---|
| `C` | Calibrer le regard (9 points) |
| `G` | Activer / couper l'influence du regard sur la caméra |
| `R` | Recentrer (pose et regard actuels = zéro) |
| `P` | Profil suivant (`default`, `ets2`, `flight_sim`, `racing`) |
| `S` | Sauvegarder le profil actif |
| `O` | Overlay transparent du point de regard |
| `Q` / `Échap` | Quitter |

Conseil : régler d'abord la tête seule (`G` pour couper le regard) dans OpenTrack, puis
ajouter le regard avec un gain modéré.

## Structure

```
Eyetrack.bat        lanceur (menu)
main.py             point d'entrée : boucle capture -> visage -> regard -> filtres -> OpenTrack
paths.py            chemins du projet (models/, data/, profiles/)

capture.py          webcam : choix automatique du mode qui délivre le plus d'images
face.py             MediaPipe Face Landmarker (478 points, iris, pose 6DoF), mode vidéo
normalize.py        normalisation du visage (Zhang 2018) à partir des landmarks
gaze_model.py       regard : ratios d'iris ou réseau MobileOne-S0 (ONNX)
calibration.py      calibration Ridge + recalibration RLS sur les clics
filters.py          filtre One Euro (+ médian), maintien et fondu en cas de perte du visage
geometry.py         géométrie de l'écran (mm, pixels, angles)
output.py           envoi UDP OpenTrack, courbe de réponse du regard, HUD
overlay.py          overlay transparent click-through (Windows)
profile.py          profils JSON par jeu

profiles/           profils de jeux (versionnés)
tools/              diagnostics : diag_regard.py, diag_stabilite.py, validate.py
docs/               état de l'art de l'eye tracking webcam
models/             modèles téléchargés (hors git)
data/               calibrations, cache caméra, sorties de diagnostic (hors git)
```

## Diagnostics

| Outil | Ce qu'il mesure |
|---|---|
| `tools/diag_regard.py` | Le réseau suit-il les yeux ? Amplitude et signal/bruit par résolution |
| `tools/diag_stabilite.py` | Tremblement brut / filtré de la tête et du regard, framerate, pertes |
| `tools/validate.py` | Précision du regard calibré sur 9 cibles (export CSV) |

Chaque calibration enregistre aussi ses données brutes dans
`data/calibration_samples_<mode>.npz` pour analyse.

## Licences

Le modèle de regard (`mobileone_s0_gaze.onnx`, projet yakhyo/gaze-estimation) est entraîné
sur Gaze360 : usage **non commercial**. MediaPipe est sous licence Apache 2.0.
