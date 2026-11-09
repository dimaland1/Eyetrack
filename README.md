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

### Raccourcis en jeu (globaux)

Ils marchent même quand le jeu a le focus :

| Touches | Action |
|---|---|
| `Ctrl`+`Maj`+`F9` | Recentrer (pose et regard actuels = zéro, en douceur) |
| `Ctrl`+`Maj`+`F10` | Activer / couper l'influence du regard sur la caméra |
| `Ctrl`+`Maj`+`F11` | Pause : la caméra revient au centre et y reste (même touche pour reprendre) |

Les combinaisons se changent dans le profil (`hotkey_recenter`, `hotkey_gaze`, `hotkey_pause`,
ex. `"pause"`, `"scrolllock"`, `"ctrl+shift+f12"`, `""` pour désactiver). Beaucoup de jeux voient
quand même les touches : éviter les lettres (dans BeamNG, `R` réinitialise le véhicule) et, si
`Ctrl`/`Maj` passent les vitesses (boîte au clavier dans ETS2), prendre une touche seule.
Si une combinaison est déjà prise par un autre programme, le tracker l'indique au démarrage.

### Raccourcis de la fenêtre « Eyetrack - Moniteur »

| Touche | Action |
|---|---|
| `C` | Calibrer le regard (9 points) |
| `G` | Activer / couper l'influence du regard sur la caméra |
| `R` | Recentrer |
| `Espace` | Pause / reprise |
| `P` | Profil suivant (`default`, `ets2`, `flight_sim`, `racing`) |
| `S` | Sauvegarder le profil actif |
| `O` | Overlay transparent du point de regard |
| `Q` / `Échap` | Quitter |

Conseil : régler d'abord la tête seule (`G` pour couper le regard) dans OpenTrack, puis
ajouter le regard avec un gain modéré.

Réglages principaux d'un profil (`profiles/*.json`) :

| Clé | Effet |
|---|---|
| `head_yaw_gain`, `head_pitch_gain` | Amplification de la rotation de la tête (2,5 = 20° de tête -> 50° en jeu) |
| `gaze_yaw_gain`, `gaze_pitch_gain` | Part du regard dans la rotation de la caméra |
| `gaze_smoothing_s` | Lissage de la part regard (plus grand = caméra plus calme, plus lente) |
| `gaze_deadzone` | Zone centrale où le regard n'agit presque pas |
| `send_rate_hz` | Fréquence d'envoi à OpenTrack (120 ; 0 = une fois par image webcam) |
| `output_smoothing_s`, `prediction_s` | Lissage court entre deux images et prédiction qui en rattrape le retard |
| `recalib_on_click` | Recalibration du regard sur les clics de souris (coupée dans les profils de jeu) |
| `hotkey_recenter`, `hotkey_gaze`, `hotkey_pause` | Raccourcis globaux |

Les clignements sont détectés : le regard est figé pendant un clignement.

### Fluidité et OpenTrack

La webcam donne 30 images par seconde, mais le tracker envoie à OpenTrack **120 fois par
seconde** : entre deux images, la pose est lissée et légèrement prédite, la caméra du jeu avance
donc par petits pas réguliers. Le HUD affiche la fréquence réelle et le plus long trou entre deux
envois (`trou max`, normalement 8 à 15 ms). Le filtre Accela d'OpenTrack reste conseillé ; son
lissage peut être réduit si la caméra paraît en retard.

La recalibration sur les clics n'apprend que les clics faits curseur **visible** : en jeu, quand
la souris est capturée (curseur caché), les clics sont ignorés. Elle est de plus coupée dans les
profils de jeu (`ets2`, `flight_sim`, `racing`).

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
output.py           envoi UDP OpenTrack (120 Hz, prédiction, pause), courbe du regard, HUD
overlay.py          overlay transparent click-through (Windows)
hotkeys.py          raccourcis clavier globaux (Windows), actifs en jeu
profile.py          profils JSON par jeu

profiles/           profils de jeux (versionnés)
tools/              diagnostics : diag_regard.py, diag_stabilite.py, validate.py
tests/              tests automatiques (sens des axes, envoi OpenTrack, raccourcis)
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
| `tools/analyse_calibration.py` | Compare plusieurs modèles de calibration sur la dernière calibration enregistrée (validation croisée) |

Chaque calibration enregistre aussi ses données brutes dans
`data/calibration_samples_<mode>.npz` pour analyse.

### Tests automatiques

```bash
python -m unittest discover -s tests -v
```

Aussi dans le menu (option 9). Ils vérifient, à partir de mouvements de tête simulés, le sens de
chaque axe envoyé à OpenTrack (tel que validé en jeu, sans aucune inversion cochée dans
OpenTrack), l'absence de faux pitch tête tournée, le sens du regard, la pause, le recentrage, la
prédiction, l'envoi à 120 Hz et les raccourcis. À relancer après toute modification de
`face.py` ou `output.py` : un test qui échoue = un axe qui partirait du mauvais côté en jeu.

## Améliorations pour la prochaine fois

La phase 1 (solidifier) est faite : raccourcis globaux, envoi à 120 Hz avec prédiction,
recalibration au clic protégée en jeu, tests du sens des axes. Restent à vérifier en jeu :
la fluidité réelle avec la vraie caméra (regarder `trou max` dans le HUD) et le confort des
raccourcis par défaut.

Ensuite :

- **Phase 2 — confort** : panneau de contrôle (profil, curseurs de gains et de lissage en
  direct, calibrer / recentrer, état fps / visage / clignement), icône dans la barre des
  tâches ; calibration guidée (note de qualité, refaire un point raté, cibles plus près des
  bords, alerte reflets sur les lunettes).
- **Phase 3 — paquet cadeau** : `.exe` installable (PyInstaller + installateur), assistant
  de premier lancement (caméra, écran, distance, calibration), profil OpenTrack fourni
  (entrée UDP, Accela, axes réglés). Rappel : le modèle de regard est non commercial.
- **Phase 4 — bonus** : changement de profil automatique selon le jeu lancé, courbes de
  réponse de la tête (douces au centre, fortes sur les côtés), presets par jeu liant profil
  tracker et profil OpenTrack.

## Licences

Le modèle de regard (`mobileone_s0_gaze.onnx`, projet yakhyo/gaze-estimation) est entraîné
sur Gaze360 : usage **non commercial**. MediaPipe est sous licence Apache 2.0.
