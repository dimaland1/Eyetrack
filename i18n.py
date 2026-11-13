"""Langue de l'interface : français ou anglais.

Le texte français sert de clé : `tr("Recentrer")` renvoie « Recenter » en anglais et le texte
lui-même en français. Les valeurs sont insérées après traduction :
`tr("{fps} images/s", fps=30)`. Une traduction manquante affiche le français (jamais d'erreur) ;
le test tests/test_i18n.py vérifie que chaque texte passé à tr() a sa traduction.

Langue choisie dans le panneau (« Langue / Language », réglage `language` de app_settings) :
"auto" suit la langue de Windows (français si Windows est en français, anglais sinon).
Les messages du journal (print) restent en français : ils servent au diagnostic.
"""

from __future__ import annotations
import ctypes
import locale
import sys

LANGUAGES = {"fr": "Français", "en": "English"}
CHOICES = ("auto", "fr", "en")  # valeurs possibles du réglage `language`

_current = "fr"

EN: dict[str, str] = {
    # ------------------------------------------------------------------ panneau : curseurs et cases
    "Tête : amplification gauche / droite": "Head: left / right amplification",
    "Regard : part dans la caméra": "Gaze: share in the camera",
    "Regard : lissage (s, haut = plus calme)": "Gaze: smoothing (s, higher = calmer)",
    "Tête : amplification haut / bas": "Head: up / down amplification",
    "Tête : lissage au repos (bas = plus calme)": "Head: smoothing at rest (lower = calmer)",
    "Regard : part haut / bas": "Gaze: up / down share",
    "Regard : zone morte au centre": "Gaze: dead zone at the center",
    "Recalibrer le regard sur les clics (bureau, curseur visible)":
        "Refine the gaze calibration on clicks (desktop, visible cursor)",
    "Ouvrir OpenTrack au lancement d'Eyetrack": "Open OpenTrack when Eyetrack starts",
    "Langue / Language": "Language / Langue",
    "Auto (langue de Windows)": "Auto (Windows language)",
    # ------------------------------------------------------------------ panneau : cadres et boutons
    "Eyetrack - Panneau de contrôle": "Eyetrack - Control panel",
    "État": "Status",
    "Visage": "Face",
    "Reflets lunettes": "Glasses glare",
    "Caméra": "Camera",
    "Regard": "Gaze",
    "Calibration": "Calibration",
    "Caméra du jeu": "Game camera",
    "{label} :": "{label}:",
    "Commandes": "Controls",
    "Recentrer": "Recenter",
    "Pause": "Pause",
    "Reprendre": "Resume",
    "Couper le regard": "Turn gaze off",
    "Réactiver le regard": "Turn gaze on",
    "Calibrer le regard": "Calibrate gaze",
    "Afficher l'overlay": "Show overlay",
    "Masquer l'overlay": "Hide overlay",
    "Voir la caméra": "Show camera",
    "Masquer la caméra": "Hide camera",
    "Ouvrir OpenTrack": "Open OpenTrack",
    "Quitter": "Quit",
    "En attente de l'image…": "Waiting for the image…",
    "Points jaunes : iris détectés. Ils doivent rester au centre des pupilles, sans sauter sur un "
    "reflet des lunettes.\nCadre vert : visage suivi ; orange : clignement ; rouge : visage perdu.":
        "Yellow dots: detected irises. They should stay at the center of the pupils, without jumping "
        "onto a reflection on the glasses.\nGreen frame: face tracked; orange: blink; red: face lost.",
    "Profil": "Profile",
    "Sauvegarder les réglages": "Save settings",
    "Réglages": "Settings",
    "Réglages avancés": "Advanced settings",
    "Réglages avancés ▸": "Advanced settings ▸",
    "Réglages avancés ▾": "Advanced settings ▾",
    # ------------------------------------------------------------------ panneau : état
    "calibration en cours…": "calibrating…",
    "clignement": "blink",
    "détecté": "detected",
    "perdu (valeur maintenue)": "lost (last position held)",
    "{fps} images/s": "{fps} fps",
    "fermé : bouton « Ouvrir OpenTrack »": "closed: \"Open OpenTrack\" button",
    "{hz} envois/s, plus long trou {gap} ms": "{hz} sends/s, longest gap {gap} ms",
    "en attente du visage": "waiting for the face",
    "actif": "on",
    "coupé": "off",
    "aucun": "none",
    "léger": "slight",
    "gênant": "strong",
    " : bougez la lampe ou l'écran": ": move the lamp or the screen",
    "{label} ({pct} % de l'iris){advice}": "{label} ({pct}% of the iris){advice}",
    "calibré": "calibrated",
    "bonne": "good",
    "moyenne": "fair",
    "à refaire": "redo",
    "inconnue": "unknown",
    "qualité {q} ({deg}°)": "quality {q} ({deg}°)",
    ", +{n} clics": ", +{n} clicks",
    "non calibré : bouton « Calibrer le regard »": "not calibrated: \"Calibrate gaze\" button",
    "PAUSE : au centre": "PAUSED: centered",
    "suit la tête": "follows the head",
    "En jeu : {keys}": "In game: {keys}",
    "recentrer": "recenter",
    "regard": "gaze",
    "pause": "pause",
    "Maj": "Shift",
    "Calcul : {calc} ms par image (visage {face}, regard {gaze})":
        "Processing: {calc} ms per frame (face {face}, gaze {gaze})",
    "Agitation (1 s, si immobile) : tête {head}°, regard {gaze} px":
        "Jitter (1 s, when still): head {head}°, gaze {gaze} px",
    "Reflets sur les iris : {pct} %": "Glare on the irises: {pct}%",
    # ------------------------------------------------------------------ icône de la zone de notification
    "Masquer le panneau": "Hide panel",
    "Afficher le panneau": "Show panel",
    "Reprendre le suivi": "Resume tracking",
    "Pause (caméra au centre)": "Pause (camera centered)",
    "Quitter Eyetrack": "Quit Eyetrack",
    "Eyetrack - démarrage (caméra, modèles)…": "Eyetrack - starting (camera, models)…",
    "en pause (caméra au centre)": "paused (camera centered)",
    "suivi actif": "tracking",
    "visage perdu": "face lost",
    "OpenTrack introuvable": "OpenTrack not found",
    "Installez OpenTrack ou indiquez son chemin (opentrack_path) dans app_settings.json.":
        "Install OpenTrack or set its path (opentrack_path) in app_settings.json.",
    "Eyetrack continue": "Eyetrack is still running",
    "Le suivi continue en arrière-plan. Cliquez sur l'icône Eyetrack pour rouvrir le panneau ; "
    "clic droit pour quitter.":
        "Tracking continues in the background. Click the Eyetrack icon to reopen the panel; "
        "right-click to quit.",
    # ------------------------------------------------------------------ modèle de regard de précision
    "Modèle de précision…": "Precision model…",
    "Modèle de regard de précision": "Precision gaze model",
    "Le modèle de regard de précision (MobileNetV2, entraîné sur le jeu de données Gaze360) n'est pas "
    "fourni avec Eyetrack : il appartient à un tiers et sa licence n'autorise qu'un usage non "
    "commercial.\n\nEyetrack peut le télécharger pour vous depuis sa source officielle :\n{source}\n\n"
    "En acceptant, vous déclarez l'utiliser à titre personnel et non commercial, conformément à sa "
    "licence. Télécharger maintenant (10 Mo) ?":
        "The precision gaze model (MobileNetV2, trained on the Gaze360 dataset) is not included with "
        "Eyetrack: it belongs to a third party and its license only allows non-commercial use.\n\n"
        "Eyetrack can download it for you from its official source:\n{source}\n\n"
        "By accepting, you state that you will use it for personal, non-commercial purposes, as its "
        "license requires. Download it now (10 MB)?",
    " - mode simple": " - simple mode",
    " - mode simple (sans modèle)": " - simple mode (no model)",
    " - téléchargement…": " - downloading…",
    " - relancez Eyetrack": " - restart Eyetrack",
    " - échec, réessayez": " - failed, try again",
    "Modèle de précision installé": "Precision model installed",
    "Quittez puis relancez Eyetrack pour l'utiliser : une nouvelle calibration vous sera proposée.":
        "Quit and restart Eyetrack to use it: a new calibration will be offered.",
    "Téléchargement impossible": "Download failed",
    "Le modèle de précision n'a pas pu être téléchargé. Vérifiez la connexion puis réessayez depuis le panneau.":
        "The precision model could not be downloaded. Check your connection and try again from the panel.",
    # ------------------------------------------------------------------ fenêtres d'erreur
    "Eyetrack - caméra introuvable": "Eyetrack - camera not found",
    "Impossible d'ouvrir la caméra (n° {n}).\n\n{err}\n\nVérifiez qu'elle est branchée et qu'aucune "
    "autre application (visio, navigateur, logiciel de la webcam…) ne l'utilise, puis relancez Eyetrack.":
        "Cannot open the camera (#{n}).\n\n{err}\n\nCheck that it is plugged in and that no other "
        "application (video call, browser, webcam software…) is using it, then start Eyetrack again.",
    "Eyetrack est déjà lancé : son icône est près de l'horloge.":
        "Eyetrack is already running: its icon is next to the clock.",
    "Arguments invalides : {args}\n\nDétails : {log}": "Invalid arguments: {args}\n\nDetails: {log}",
    "Eyetrack s'est arrêté à cause d'une erreur :\n\n{kind} : {err}\n\nDétails dans le journal :\n{log}":
        "Eyetrack stopped because of an error:\n\n{kind}: {err}\n\nDetails in the log:\n{log}",
    "Eyetrack - erreur": "Eyetrack - error",
    # ------------------------------------------------------------------ écrans de calibration (sans accents)
    "Calibration (Echap pour annuler)": "Calibration (Esc to cancel)",
    "Point {i}/{n} - Fixez le repere orange": "Point {i}/{n} - Look at the orange target",
    "Conseil : bougez doucement la tete (inclinaison / rotation) en fixant la cible":
        "Tip: gently move your head (tilt / turn) while looking at the target",
    "Reflet sur les lunettes : deplacez la lampe ou inclinez un peu l'ecran":
        "Glare on your glasses: move the lamp or tilt the screen a little",
    "Visage non detecte : restez face a la camera": "Face not detected: stay in front of the camera",
    "Tete": "Head",
    "BONNE": "GOOD",
    "MOYENNE": "FAIR",
    "A REFAIRE": "REDO",
    "Calibration terminee - qualite : {label} (erreur moyenne {deg} deg)":
        "Calibration done - quality: {label} (mean error {deg} deg)",
    "Cercle : cible ; croix : ou le regard a ete estime (chaque point predit sans lui)":
        "Circle: target; cross: estimated gaze (each point predicted without itself)",
    "reflet": "glare",
    "[R] Refaire les points {pts} (rouges ou avec reflet)": "[R] Redo points {pts} (red or with glare)",
    "[R] Refaire le moins bon point ({n})": "[R] Redo the worst point ({n})",
    "[Entree] Valider": "[Enter] Accept",
    "[Echap] Annuler (garder l'ancienne calibration)": "[Esc] Cancel (keep the previous calibration)",
    "5 points : note approximative (9 points conseilles)": "5 points: rough rating (9 points recommended)",
    "Reflets sur les lunettes aux points {pts} : deplacez la lampe ou inclinez l'ecran avant de les refaire":
        "Glare on your glasses at point(s) {pts}: move the lamp or tilt the screen before redoing them",
    "Bienvenue dans Eyetrack": "Welcome to Eyetrack",
    "Premiere utilisation : on calibre votre regard (environ 1 minute)":
        "First use: let's calibrate your gaze (about 1 minute)",
    "1. Installez-vous comme pour jouer, face a la camera (gardez vos lunettes si vous en portez).":
        "1. Sit as you do when playing, facing the camera (keep your glasses on if you wear them).",
    "2. Un repere orange apparait a {n} endroits : fixez-le jusqu'a ce qu'il change de place.":
        "2. An orange target appears in {n} places: look at it until it moves.",
    "3. Bougez doucement la tete en le fixant : le regard est appris dans toutes les positions.":
        "3. Gently move your head while looking at it: gaze is learned in every position.",
    "4. A la fin, une note s'affiche : [Entree] valide, [R] refait les points rates.":
        "4. At the end, a rating is shown: [Enter] accepts, [R] redoes the missed points.",
    "Visage detecte": "Face detected",
    "Visage non detecte : placez-vous face a la camera": "Face not detected: sit in front of the camera",
    "[Entree] Commencer la calibration": "[Enter] Start the calibration",
    "[Echap] Plus tard (bouton \"Calibrer le regard\" du panneau)":
        "[Esc] Later (\"Calibrate gaze\" button in the panel)",
}

_TABLES = {"en": EN}


def system_language() -> str:
    """Langue de Windows (ou du système) : "fr" si français, "en" sinon."""
    try:
        if sys.platform == "win32":
            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return "fr" if (langid & 0x3FF) == 0x0C else "en"  # LANG_FRENCH
        code = (locale.getlocale()[0] or "").lower()
        return "fr" if code.startswith("fr") else "en"
    except Exception:
        return "en"


def resolve(choice: str) -> str:
    """Langue effective d'un réglage ("auto" ou inconnu -> langue du système)."""
    return choice if choice in LANGUAGES else system_language()


def set_language(choice: str) -> str:
    """Change la langue de l'interface ; renvoie la langue effective ("fr" ou "en")."""
    global _current
    _current = resolve(choice)
    return _current


def language() -> str:
    return _current


def tr(text: str, **values) -> str:
    """Texte dans la langue courante (le français sert de clé), valeurs insérées ensuite."""
    out = _TABLES.get(_current, {}).get(text, text)
    return out.format(**values) if values else out
