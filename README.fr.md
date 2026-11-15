# Eyetrack

[English](README.md) · **Français** · [Site du projet](https://dimaland1.github.io/Eyetrack/)

Suivi de la tête et du regard avec une simple webcam, envoyé à OpenTrack (protocole
« UDP over network », port 4242) pour piloter la caméra dans les jeux et simulateurs.

## Installation

```bash
pip install -r requirements.txt
```

Le modèle du visage (MediaPipe, `models/`) est téléchargé automatiquement au premier lancement.
Le **modèle de regard de précision** (licence non commerciale d'un tiers) ne l'est jamais sans
accord : bouton « Modèle de précision… » du panneau (sans lui, Eyetrack tourne en mode simple).
OpenTrack s'installe depuis le menu (option 8) ou avec
`winget install --id opentrack.opentrack -e`.

### Installeur à donner : `Eyetrack-Setup.exe`

Option **I** du menu `Eyetrack.bat` (ou `python tools/build_exe.py --installer`) : construit
l'exécutable, le soumet à l'auto-test puis compile `dist\installer\Eyetrack-Setup.exe` avec
[Inno Setup 6](https://jrsoftware.org/isinfo.php) (gratuit ; `winget install --id
JRSoftware.InnoSetup -e --scope user`). Script : `installer/eyetrack.iss`. L'installeur :

- s'installe pour l'utilisateur, **sans droits administrateur**, en français ou en anglais (la
  langue choisie devient celle d'Eyetrack) ; raccourcis menu Démarrer et Bureau ;
- propose **OpenTrack** s'il n'est pas installé (winget ; Windows demande alors une autorisation) ;
- propose le **modèle de regard de précision**, case décochée : il n'est **pas inclus** (licence
  non commerciale d'un tiers) ; en cochant, l'utilisateur accepte sa licence et l'installeur le
  télécharge depuis sa source officielle, empreinte SHA-256 vérifiée. Installation silencieuse :
  `/VERYSILENT /DOWNLOADMODEL=1` ;
- refuse de s'installer pendant qu'Eyetrack tourne ; la désinstallation (« Applications
  installées ») garde la calibration et les réglages, sauf si l'utilisateur demande de les effacer.

## Lancement

### Pour jouer : `Eyetrack` sur le Bureau (sans terminal)

`Eyetrack.exe` s'ouvre directement sur le panneau de contrôle, sans fenêtre de terminal. On le
construit et l'installe une fois (puis après chaque mise à jour du code) avec l'**option E** du
menu `Eyetrack.bat`, ou `python tools/build_exe.py` ; il faut quitter Eyetrack avant. Avant
d'installer, l'exécutable construit passe un **auto-test** (`Eyetrack.exe --self-test` : modèles,
MediaPipe, interface, sans caméra) ; s'il échoue, rien n'est installé et la version précédente
reste en place.

- **Double-clic** : reprend le dernier profil utilisé, ouvre OpenTrack s'il n'est pas déjà
  ouvert (case « Ouvrir OpenTrack au lancement d'Eyetrack » dans « Réglages avancés »), puis
  démarre le suivi avec le panneau et l'icône près de l'horloge.
- **Premier lancement** (pas encore de calibration) : un écran d'accueil s'ouvre tout seul,
  explique la calibration et montre l'image de la caméra (cadre vert = visage détecté).
  `Entrée` lance la calibration, `Échap` la remet à plus tard (bouton « Calibrer le regard » du
  panneau) ; elle est reproposée au lancement suivant tant qu'il n'y a pas de calibration.
- **Langue** : français ou anglais (panneau, icône, écrans de calibration, messages d'erreur).
  Par défaut, celle de Windows ; à changer dans « Réglages avancés » → « Langue / Language »
  (appliqué tout de suite). Le journal `eyetrack.log` reste en français.
- **Profil selon le jeu** : quand ETS2 / ATS, BeamNG.drive, Assetto Corsa, iRacing, MSFS, X-Plane
  ou DCS est lancé, Eyetrack charge son profil (`ets2`, `beamng`, `racing`, `flight_sim`) en
  1 à 2 s, avec une notification, puis revient au profil d'avant à la fermeture du jeu. Un
  profil changé à la main pendant le jeu est respecté. Le profil habituel (repris au lancement)
  ne change pas. Case « Changer de profil selon le jeu lancé » dans « Réglages avancés ».
  Autres jeux : ajouter `"game_processes": ["MonJeu.exe"]` au JSON du profil (`[]` = jamais).
- **Déjà lancé ?** Un nouveau double-clic rouvre simplement le panneau (une seule instance : la
  caméra et le port d'OpenTrack ne se partagent pas).
- **Erreurs** (caméra introuvable ou occupée…) : affichées dans une fenêtre. Journal du dernier
  lancement : `%LOCALAPPDATA%\Eyetrack\eyetrack.log` (le précédent : `eyetrack.old.log`).
- **Où sont les fichiers ?** Le programme est dans `%LOCALAPPDATA%\Programs\Eyetrack`
  (remplacé à chaque mise à jour). La calibration et les profils de l'exécutable sont dans
  `%LOCALAPPDATA%\Eyetrack`, et une mise à jour ne les efface jamais. Ils sont séparés de ceux des
  sources : on peut essayer une version en développement sans toucher à la calibration du jeu.
  À la première installation, la calibration et les profils des sources y sont repris.
- L'exécutable n'est pas signé : au premier lancement, Windows SmartScreen peut afficher
  « Windows a protégé votre ordinateur » → « Informations complémentaires » → « Exécuter quand
  même ».

### Pour développer : `Eyetrack.bat` ou la ligne de commande

Double-cliquer sur **`Eyetrack.bat`** et choisir dans le menu, ou en ligne de commande :

```bash
python main.py --gaze-mode net --profile default
```

Au premier lancement, la recherche du meilleur mode caméra prend ~20 s ; il est ensuite
mémorisé dans `data/camera_mode.json` (supprimer ce fichier pour relancer la recherche).

### Panneau de contrôle

Une fenêtre « Eyetrack - Panneau de contrôle » s'ouvre avec le tracker :

- **État** : visage détecté / perdu / clignement, reflets sur les lunettes, images par seconde,
  envois à OpenTrack et plus long trou, regard actif ou coupé, note de la calibration, pause ;
- **Commandes** : recentrer, pause, regard on/off, calibrer, overlay, voir la caméra, ouvrir
  OpenTrack (la ligne « OpenTrack » de l'état passe en orange quand il est fermé), quitter ;
- **Caméra** (bouton « Voir la caméra », fermé par défaut) : aperçu de l'image avec les iris
  détectés (points jaunes), un cadre vert / orange / rouge (visage suivi / clignement / perdu),
  le temps de calcul et l'agitation. Utile pour régler le cadrage, la lumière et vérifier les
  reflets des lunettes ; en jeu, le laisser fermé (il ne coûte alors rien) ;
- **Profil** : changer de profil et sauvegarder les réglages dans son fichier JSON ;
- **Réglages en direct** : les trois essentiels (amplification de la tête, part du regard,
  lissage du regard) ; le reste dans « Réglages avancés » (amplification haut/bas, lissage de la
  tête au repos, part du regard haut/bas, zone morte, recalibration sur les clics). Les
  changements s'appliquent tout de suite ; ils ne sont gardés que si l'on clique sur
  « Sauvegarder les réglages ».

**Zone de notification** : une icône Eyetrack s'affiche près de l'horloge. La croix du panneau
le **réduit** (le suivi continue) ; un clic sur l'icône le rouvre ; clic droit : menu (recentrer,
pause, regard, quitter). Pour arrêter le tracker : bouton « Quitter » ou menu de l'icône.

L'ancienne fenêtre « Moniteur » (vidéo agrandie et HUD
de diagnostic complet) ne s'ouvre plus que sur demande : `python main.py --moniteur`, ou sans
panneau : `python main.py --no-panel`.

### Calibration guidée

Bouton « Calibrer le regard » (ou `C` dans le Moniteur) : 9 cibles, à 10 % des bords de l'écran
pour que le regard soit appris jusque dans les coins. Fixer chaque cible en bougeant doucement la
tête. Si un reflet couvre les iris, un message rouge s'affiche : déplacer la lampe ou incliner un
peu l'écran. L'écran de calibration passe au-dessus de toutes les fenêtres (jeu, navigateur,
panneau) et prend le clavier, même lancé par un raccourci pendant le jeu.

À la fin, un écran de résultats montre, pour chaque cible, où le regard a été estimé (croix) et
l'erreur en degrés. Chaque point est prédit **sans lui** (validation croisée) : la note ne flatte
pas la calibration. Note globale : **bonne** jusqu'à 5,5°, **moyenne** jusqu'à 7°, **à refaire**
au-delà (seuils calés sur de vraies calibrations avec lunettes : 3,9° et 4,8° ; un point raté donne
18 à 38°). Les coins sont naturellement un peu moins bons que le centre.

| Touche | Action |
|---|---|
| `Entrée` | Valider la calibration |
| `R` | Refaire seulement les points en rouge ou pris avec un reflet (sinon le moins bon) |
| `Échap` | Annuler : l'ancienne calibration est conservée |

La note est enregistrée avec la calibration et affichée dans le panneau.

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

### Raccourcis de la fenêtre « Eyetrack - Moniteur » (avec `--moniteur`)

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
donc par petits pas réguliers. Le panneau affiche la fréquence réelle et le plus long trou entre deux
envois (normalement 8 à 15 ms). Le filtre Accela d'OpenTrack reste conseillé ; son
lissage peut être réduit si la caméra paraît en retard.

La recalibration sur les clics n'apprend que les clics faits curseur **visible** : en jeu, quand
la souris est capturée (curseur caché), les clics sont ignorés. Elle est de plus coupée dans les
profils de jeu (`ets2`, `beamng`, `flight_sim`, `racing`).

## Structure

```
Eyetrack.bat        lanceur (menu)
main.py             point d'entrée : boucle capture -> visage -> regard -> filtres -> OpenTrack
eyetrack_app.py     point d'entrée d'Eyetrack.exe (dernier profil, OpenTrack, instance unique, erreurs)
eyetrack.spec       recette PyInstaller de l'exécutable (utilisée par tools/build_exe.py)
LICENSE             licence GPL-3.0 ; THIRD_PARTY_NOTICES.md : bibliothèques embarquées et leurs licences
paths.py            chemins (sources : models/, data/, profiles/ ; exécutable : %LOCALAPPDATA%)
app_settings.py     réglages de l'application (dernier profil, ouverture d'OpenTrack, langue)
i18n.py             langue de l'interface : français / anglais (texte français = clé, traduction anglaise)
winapp.py           application sans console : journal, fenêtres d'erreur, instance unique, OpenTrack

capture.py          webcam : choix automatique du mode qui délivre le plus d'images
face.py             MediaPipe Face Landmarker (478 points, iris, pose 6DoF), mode vidéo
normalize.py        normalisation du visage (Zhang 2018) à partir des landmarks
gaze_model.py       regard : ratios d'iris (mode simple) ou réseau MobileNetV2 (ONNX, fourni par
                    l'utilisateur ; MobileOne-S0 accepté en repli), téléchargement vérifié
calibration.py      calibration Ridge (fuite de tête compensée, centroïdes robustes) + RLS sur les clics
filters.py          filtre One Euro (+ médian), maintien et fondu en cas de perte du visage
geometry.py         géométrie de l'écran (mm, pixels, angles)
output.py           envoi UDP OpenTrack (120 Hz, prédiction, pause), courbe du regard, HUD
overlay.py          overlay transparent click-through (Windows)
tray.py             icône de la zone de notification (API Windows, sans dépendance)
panel.py            panneau de contrôle (état, commandes, réglages en direct, aperçu caméra)
hotkeys.py          raccourcis clavier globaux (Windows), actifs en jeu
profile.py          profils JSON par jeu

profiles/           profils de jeux (versionnés)
tools/              diagnostics : diag_regard.py, diag_stabilite.py, validate.py ; make_icon.py ;
                    build_exe.py (construit et installe Eyetrack.exe, avec licences et notices ;
                    --installer : Eyetrack-Setup.exe)
installer/          script Inno Setup de l'installeur (eyetrack.iss)
research/           pipeline expérimental (collecte solo, modèles personnels et différentiels,
                    évaluations) : hors du tracker, voir docs/these/
assets/             icône de l'application (eyetrack.ico)
tests/              tests automatiques (axes, envoi OpenTrack, capture, calibration, langue, modèle…)
docs/               site du projet (index.html, publié par GitHub Pages depuis docs/) ; état de l'art
                    de l'eye tracking webcam ; docs/these/ : dossier scientifique
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
| `tools/compare_gaze_models.py` | Compare les réseaux de regard (MobileNetV2, MobileOne-S0, ResNet-18) sur les mêmes images : 2 séries de 9 cibles, note de l'application, série croisée, tremblement, temps de calcul |

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

## État et suite

Fait :

- **Phase 1 — solidifier** : raccourcis globaux, envoi à 120 Hz avec prédiction (plus long trou
  mesuré : 10 ms), recalibration au clic protégée en jeu, tests du sens des axes.
- **Phase 2 — confort** : panneau de contrôle, aperçu caméra, calibration guidée, alerte reflets,
  icône de la zone de notification.
- **Phase 3 — exécutable et installeur** : `Eyetrack.exe` sans terminal, écran d'accueil et
  calibration au premier lancement (au premier plan), interface français / anglais, installeur
  `Eyetrack-Setup.exe` (OpenTrack proposé, modèle de regard jamais fourni).
- **Précision** : webcam en 720p MJPG (au lieu de 640×480 non compressé), compensation de la
  fuite de rotation de tête, centroïdes robustes, réseau MobileNetV2, overlay stabilisé. Les
  calibrations d'avant (format v5 ou autre modèle) sont refusées : en refaire une.

À faire :

- mesurer la précision réelle avec la nouvelle calibration (comparer aux 3,2° d'avant) ;
- seuils de reflet (« léger » dès 5 % de l'iris, « gênant » dès 15 %) à confirmer en vrai ;
- pour diffuser : profil OpenTrack fourni (entrée UDP, Accela, axes réglés), signature de
  l'exécutable (SmartScreen) ; avant toute vente, un modèle de regard sous
  licence libre (piste du dossier `docs/these/`, scénario B) ;
- bonus : courbes de réponse de la tête, presets par jeu liant profil tracker et profil
  OpenTrack (le profil automatique selon le jeu est fait).

## Licences

Eyetrack est un logiciel libre sous licence **GNU GPL v3** (texte complet dans [LICENSE](LICENSE)).
Les bibliothèques embarquées dans l'exécutable et leurs licences sont listées dans
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Chaque construction en écrit la liste exacte
(`THIRD_PARTY_NOTICES.txt`, avec les versions) et les textes complets (dossier `licenses`) à
côté d'`Eyetrack.exe`.

Le modèle de regard (`mobilenetv2_gaze.onnx`, ou `mobileone_s0_gaze.onnx` en repli, projet
yakhyo/gaze-estimation) est entraîné sur Gaze360 : usage **non commercial**. Il n'est donc **jamais fourni** avec Eyetrack (ni dans
l'exécutable, ni dans l'installeur) : l'utilisateur le télécharge lui-même depuis sa source
officielle, après avoir accepté sa licence (installeur ou panneau), dans
`%LOCALAPPDATA%\Eyetrack\models`. Sans lui, Eyetrack fonctionne en mode simple (MediaPipe
seul). MediaPipe est sous licence Apache 2.0.
