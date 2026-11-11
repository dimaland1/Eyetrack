# Chapitre 1 — Introduction et Problématique Scientifique

## 1.1 Contexte et motivation

Le suivi du regard (*eye tracking*) et de la posture crânienne (*head tracking* à six degrés de liberté, 6DoF) constitue une modalité d'interaction homme-machine majeure dans la simulation aéronautique et automobile, l'accessibilité, l'ergonomie cognitive et la réalité étendue. Historiquement, les solutions de précision inférieure au degré angulaire reposent sur des capteurs infrarouges actifs exploitant le principe PCCR (*Pupil Center Corneal Reflection*), tels que les systèmes Tobii (environ $0{,}67^\circ$ à $< 1^\circ$ en conditions contrôlées). Cette précision matérielle s'accompagne toutefois d'un coût dédié, d'une sensibilité extérieure aux rayons infrarouges solaires et d'une absence d'ubiquité sur les postes bureautiques existants.

À l'inverse, l'estimation du regard par simple webcam RGB grand public (par exemple une Logitech C920 à $1280 \times 720$ ou $1920 \times 1080$ pixels) démocratise l'accès au suivi oculo-céphalique. En pratique, l'état de l'art commercial (Eyeware Beam, fondé sur la lignée scientifique EPFL/Idiap) revendique $1{,}5^\circ$ dans des conditions idéales après calibration à 5 points, tandis que les évaluations académiques indépendantes placent la majorité des trackers webcam non contrôlés entre $2{,}3^\circ$ et $4{,}5^\circ$.

Au-delà de la barre algorithmique, l'écosystème du *deep learning* appliqué au regard souffre d'un **verrou structurel et juridique** souvent ignoré : la quasi-totalité des grands jeux de données publics de référence — MPIIGaze / MPIIFaceGaze, GazeCapture, ETH-XGaze, Gaze360, EYEDIAP et EVE — sont distribués sous licence **strictement non commerciale** (CC BY-NC-SA ou accords académiques restrictifs). Or, dès qu'un modèle neuronal est entraîné sur ces données, ses poids héritent généralement de la restriction non commerciale, empêchant toute distribution libre ou commerciale d'un logiciel autonome, comme nous avons dû l'appliquer dans la conception même du projet `Eyetrack` (téléchargement séparé et optionnel de `mobileone_s0_gaze.onnx`).

## 1.2 Analyse du problème physique et de l'existant `Eyetrack`

Le projet logiciel `Eyetrack` implémente déjà un pipeline modulaire complet temps réel vers OpenTrack (UDP 120 Hz) combinant :
1. L'extraction de 478 points faciaux (dont les contours et centres iriens) et d'une matrice de pose rigide via MediaPipe FaceLandmarker (`face.py`) ;
2. Une normalisation projective vers une caméra virtuelle canonique à $600\text{ mm}$ selon Zhang et al. (2018) (`normalize.py`, focale virtuelle calibrée expérimentalement à $1900\text{ px}$ sur patch $448 \times 448$) ;
3. Un estimateur hybride combinant `MobileOne-S0` (ONNX, entraîné sur Gaze360) et ratios géométriques iriens (`gaze_model.py`), projetés sur le plan écran (`geometry.py`) ;
4. Une calibration affine régularisée Ridge sur 5 ou 9 points avec validation croisée *Leave-One-Target-Out* (LOO-CV), détection des reflets de verres correcteurs et adaptation en ligne par moindres carrés récursifs (RLS) sur clics souris (`calibration.py`) ;
5. Un pré-filtrage médian couplé à un filtre adaptatif One Euro (`filters.py`).

L'audit empirique de cette architecture de référence sur des données réelles capturées avec verres correcteurs ($N = 405$ échantillons, écran $2560 \times 1440$ de $27''$ à $60\text{ cm}$, diamètre irien médian de $20{,}5\text{ px}$) met en évidence **trois limitations fondamentales** :
- **Plafond de précision post-calibration** : le meilleur modèle linéaire basé sur l'existant atteint $3{,}18^\circ$ en fixation ($0{,}5\text{ s}$) et $4{,}40^\circ$ image par image en LOO-CV (avec dégradation nette dans les coins jusqu'à $7{,}40^\circ$), soit plus du double de l'objectif de $1{,}5^\circ$.
- **Coût de latence CPU du backbone plein-visage** : l'inférence de `MobileOne-S0` à résolution $448 \times 448$ consomme à elle seule $11{,}16\text{ ms}$ en moyenne sur CPU ($p_{95} = 12{,}81\text{ ms}$), occupant environ $67\,\%$ du budget temps réel à $60\text{ Hz}$ ($16{,}67\text{ ms}$).
- **Couplage statistique et dépendance NC** : l'ajout des angles de pose crânienne dans une calibration statique à 9 points dégrade l'erreur LOO-CV de $3{,}18^\circ$ à $5{,}25^\circ$ par sur-ajustement de colinéarités locales, tandis que le réseau générique reste soumis aux licences non commerciales de Gaze360.

## 1.3 Problématique et hypothèses de recherche

Ce dossier de recherche pose la problématique unifiée suivante :

> **Problématique centrale :** *Comment concevoir, collecter et entraîner un système d'estimation du regard monoculaire RGB temps réel sur CPU lorsque l'on ne dispose que d'un seul sujet humain (l'auteur du système), tout en répondant simultanément à deux régimes antagonistes : l'hyper-personnalisation sous-1,5° (Scénario A) et la généralisation inter-sujets distribuable sans poids non commerciaux (Scénario B) ?*

Nous formulons trois hypothèses scientifiques testables :

- **Hypothèse $H_1$ (Acquisition continue mono-sujet et compensation du retard visuo-moteur) :** L'échantillonnage discret par grille fixe (9 points) sous-échantillonne l'espace combiné *(regard $\times$ pose 6DoF)* et favorise le surapprentissage de la pose crânienne. Une collecte continue par **poursuite oculaire lisse (*Smooth Pursuit*)** couplée à une **récolte passive pré-saccadique sur clics souris** (tampon circulaire compensé à $t_{\text{clic}} - \tau_{\text{pre}}$) permet d'acquérir un ordre de grandeur supérieur d'échantillons variés ($10^3$–$10^4$ frames en quelques minutes), à condition d'estimer et de compenser explicitement le retard visuo-moteur $\tau_{\text{lag}}$ ($\approx 60\text{--}100\text{ ms}$).
- **Hypothèse $H_2$ (Découplage spatial bi-oculaire léger pour le Scénario A) :** Remplacer l'entrée plein-visage $448 \times 448$ par deux patchs oculaires alignés sur les commissures ($2 \times 48 \times 96$) munis d'un encodage spatial explicite (`CoordConv`) et fusionnés avec un vecteur géométrique 6DoF réduit la latence d'inférence d'un facteur $> 30\times$ sur CPU tout en empêchant la mémorisation de l'arrière-plan, ouvrant l'accès au régime $< 1{,}5^\circ$ spécifique à l'utilisateur.
- **Hypothèse $H_3$ (Représentation différentielle et randomisation morphologique pour le Scénario B) :** Un réseau direct entraîné sur un unique sujet $S_0$ mémorise irrémédiablement l'anatomie palpébrale et l'angle $\kappa$ de $S_0$, échouant sur des sujets inédits $S_k$. En revanche, une architecture **siamoise différentielle** prédisant la variation relative $\Delta\mathbf{g}(q, r) = \mathbf{g}(q) - \mathbf{g}(r)$ sur patchs normalisés en luminance avec **antisymétrie structurelle** ($f(q,r)=-f(r,q)$) et **déformations morphologiques synthétiques** (étirement palpébral, gamma, bruit capteur, reflets) permet de transférer les connaissances apprises sur un seul sujet vers des anatomies et des angles $\kappa$ inconnus avec seulement $K \in \{1, 5\}$ ancres de calibration.

## 1.4 Synthèse des contributions

Les travaux présentés dans ce dossier apportent les contributions suivantes :
1. **Formalisation mathématique unifiée** du pipeline hybride complet (projection de Zhang 2018, décomposition Euler $R_y R_x R_z$ sans repliement, géométrie métrique par diamètre irien, régression Ridge planchéisée, mise à jour RLS, filtrage médian + One Euro).
2. **Audit expérimental complet de la baseline réelle** ($N=405$ captures réelles avec lunettes, validation croisée LOO-CV, analyse d'asymétrie yaw/pitch, balayage de régularisation $\alpha$, profilage CPU exact).
3. **Suite logicielle de collecte solo** (`research/collect_pursuit.py`, `research/collect_clicks.py`, `research/lag_estimation.py`, `research/eye_crop.py`) intégrant alignement binoculaire $96 \times 48$, estimation du lag par validation croisée temporelle et tampon circulaire pré-clic.
4. **Deux architectures neuronales légères (~67–71 k paramètres, export ONNX $< 0{,}25\text{ ms}$ CPU)** : `DualEyePersonalNet` (Scénario A) et `DifferentialEyeNet` (Scénario B), évaluées sur banc paramétrique contrôlé multi-sujets et prêtes pour la campagne d'acquisition webcam multi-sessions.
