# Estimation du Regard Monoculaire RGB Temps Réel à partir d'un Unique Sujet : De l'Hyper-Personalisation Sous-1,5° à la Généralisation Différentielle sans Poids Non Commerciaux

**Dossier de recherche expérimentale et monographie technique — Projet `Eyetrack`**

---

## Résumé (Français)

L'estimation du regard par webcam RGB grand public se heurte simultanément à deux verrous : un plafond de précision pratique autour de $3^\circ\text{--}4{,}5^\circ$ (dû notamment à la variabilité inter-individuelle de l'angle $\kappa$ et au couplage tête–œil) et une dépendance quasi universelle aux jeux de données publics sous licence strictement non commerciale (Gaze360, ETH-XGaze, MPIIFaceGaze, GazeCapture). Lorsque l'auteur d'un système ne dispose que de sa propre personne pour acquérir des données, ce régime mono-sujet pose deux problèmes symétriques : (A) comment exploiter l'identité unique pour descendre sous le seuil de $1{,}5^\circ$ en temps réel sur CPU, et (B) comment apprendre malgré tout un modèle distribuable généralisant à des utilisateurs inédits ?

Ce dossier formalise, implémente et évalue expérimentalement une réponse unifiée sur la base du tracker `Eyetrack`. L'audit de la baseline réelle ($N = 405$ captures calibrées sur écran $2560 \times 1440$ à $60\text{ cm}$ avec verres correcteurs) révèle une erreur LOO-CV de $3{,}18^\circ$ en fixation ($4{,}40^\circ$ par image) et une latence de $11{,}16\text{ ms}$ pour le réseau plein-visage `MobileOne-S0`. Nous proposons : (1) un pipeline d'auto-collecte continue combinant poursuite oculaire lisse (*Smooth Pursuit*) compensée du retard visuo-moteur $\hat{\tau}_{\text{lag}}$ et capture passive pré-saccadique sur clics souris, (2) l'architecture légère `DualEyePersonalNet` ($70{,}5\text{ k}$ paramètres, patchs binoculaires alignés $48 \times 96$ avec `CoordConv2d` et pose 6DoF, $0{,}25\text{ ms}$ en ONNX CPU), et (3) l'architecture siamoise antisymétrique `DifferentialEyeNet` ($66{,}8\text{ k}$ paramètres) entraînée par paires différentielles en luminance normalisée avec randomisation morphologique des paupières pour le transfert inter-sujets $K$-ancres.

---

## Abstract (English)

Consumer RGB webcam gaze tracking faces two simultaneous barriers: a practical accuracy ceiling around $3^\circ\text{--}4.5^\circ$ (driven by inter-subject $\kappa$-angle variability and head–eye coupling) and near-universal reliance on non-commercial public datasets (Gaze360, ETH-XGaze, MPIIFaceGaze, GazeCapture). When a single developer can only record their own face, this single-subject regime poses a dual scientific challenge: (A) how to leverage mono-identity training to break the $1.5^\circ$ barrier in real time on CPU, and (B) how to nonetheless learn a distributable model that generalizes to unseen users?

This research monograph formalizes, implements, and empirically evaluates a unified architecture built upon the `Eyetrack` system. Auditing the real human baseline ($N = 405$ calibrated samples on a $2560 \times 1440$ display at $60\text{ cm}$ with eyeglasses) yields $3.18^\circ$ fixation LOO-CV error ($4.40^\circ$ per frame) and $11.16\text{ ms}$ CPU inference latency for full-face `MobileOne-S0`. We introduce: (1) a solo collection suite combining visuomotor-lag-compensated Smooth Pursuit and pre-saccadic passive mouse-click buffering, (2) `DualEyePersonalNet` ($70.5\text{k}$ parameters on $48 \times 96$ aligned binocular eye crops with `CoordConv2d` and 6DoF pose, $0.25\text{ ms}$ ONNX CPU), and (3) `DifferentialEyeNet` ($66.8\text{k}$ parameters), a structurally antisymmetric Siamese model trained on normalized grayscale eye pairs with morphological eyelid randomization for $K$-anchor cross-subject transfer.

---

## Nomenclature mathématique unifiée

| Symbole | Signification | Unité / Domaine |
|---|---|---|
| $K \in \mathbb{R}^{3\times 3}$ | Matrice intrinsèque estimée de la webcam | $\text{px}$ |
| $R_{\text{norm}} \in SO(3)$ | Rotation de la caméra virtuelle canonique (Zhang 2018) | sans dimension |
| $\mathbf{o}_{3\text{D}} \in \mathbb{R}^3$ | Origine 3D estimée du centre facial/oculaire (repère caméra) | $\text{mm}$ |
| $D_{\text{iris}}$ | Diamètre anatomique moyen de l'iris humain ($11{,}7$) | $\text{mm}$ |
| $(\psi, \theta, \phi)$ | Angles d'Euler intrinsèques de la tête (*yaw, pitch, roll*) | $\text{deg}$ |
| $\kappa$ | Écart angulaire individuel entre axe optique et axe visuel | $\text{deg}$ |
| $(I_L, I_R)$ | Patchs oculaires alignés ($48 \times 96$) gauche et droit | $\text{uint8} / \mathbb{R}$ |
| $\tau_{\text{lag}}$ | Retard visuo-moteur en poursuite lisse | $\text{ms}$ |
| $\alpha, \lambda$ | Paramètre Ridge et facteur d'oubli RLS | sans dimension |
| $\varepsilon_{\text{vis}}$ | Erreur angulaire visuelle 3D exacte depuis $\mathbf{o}_{3\text{D}}$ | $\text{deg}$ |

---

## Architecture du manuscrit et traçabilité logicielle

| Chapitre | Objet scientifique | Modules / Artefacts associés |
|---|---|---|
| **Chapitre 1** | Problématique, limites existantes, hypothèses $H_1$–$H_3$ | `README.md`, `docs/these/01_introduction_problematique.md` |
| **Chapitre 2** | État de l'art critique (2014–2026), SOTA, licences & brevets | `docs/etat_de_l_art.md`, `references.bib` |
| **Chapitre 3** | Formalisme mathématique complet d'`Eyetrack` | `face.py`, `normalize.py`, `gaze_model.py`, `geometry.py`, `calibration.py`, `filters.py` |
| **Chapitre 4** | Protocoles d'acquisition solo (Smooth Pursuit, clics, anti-fuite) | `research/eye_crop.py`, `research/lag_estimation.py`, `research/collect_pursuit.py`, `research/collect_clicks.py` |
| **Chapitre 5** | Architectures `DualEyePersonalNet` (A) et `DifferentialEyeNet` (B) | `research/models.py`, `research/augmentations.py`, `research/train.py` |
| **Chapitre 6** | Résultats expérimentaux, ablations, statistiques et latence CPU | `research/analyze_existing_data.py`, `research/run_synthetic_ablations.py`, `research/evaluate.py` |
| **Chapitre 7** | Discussion critique, menaces à la validité et feuille de route | `docs/these/07_discussion_limites_et_perspectives.md` |
