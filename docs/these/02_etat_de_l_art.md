# Chapitre 2 — État de l'Art Critique et Analytique (2014–2026)

Ce chapitre synthétise l'évolution scientifique de l'estimation du regard monoculaire sans éclairage infrarouge (*appearance-based* et hybride), en mettant en regard les avancées architecturales, le goulot d'étranglement de la généralisation de domaine, les méthodes de calibration peu coûteuses en annotations (*few-shot* et implicite) et les contraintes juridiques de distribution.

## 2.1 Taxonomie des approches d'estimation du regard

Trois paradigmes structurent la littérature du suivi oculaire distant :

### 2.1.1 Méthodes géométriques pures et modèles 3D de l'œil

Les trackers infrarouges dédiés (Tobii T60 à $0{,}67^\circ$ selon Skovsgaard et al., 2011 ; Tobii 4C $< 1^\circ$) exploitent le principe actif PCCR (*Pupil Center Corneal Reflection*) : des diodes infrarouges créent un ou plusieurs reflets cornéens (images de Purkinje) stables vis-à-vis de l'axe optique, permettant de trianguler l'orientation du globe indépendamment des translations de tête.

En lumière visible pure (webcam RGB), deux voies géométriques existent :
1. **Les ratios géométriques locaux** (ex. `GazeTracking`, ou le mode `iris` d'`Eyetrack` basé sur les 478 points MediaPipe) : le centre irien est rapporté aux commissures externe/interne et aux bords palpébraux. Si la détection requiert peu de calcul ($< 0{,}01\text{ ms}$ post-landmarks), un iris à $60\text{ cm}$ en 720p ne mesure qu'environ $20\text{ px}$ (ou $\sim 10\text{ px}$ en 480p), rendant le rapport vertical particulièrement vulnérable aux clignements, aux occlusions palpébrales lors du regard vers le bas et aux variations de pose de tête.
2. **Les modèles génératifs 3D du globe oculaire** : représentés notamment par les travaux de l'Idiap (Funes Mora & Odobez, G3E 2014 ; IJCV 2016 ; brevet WO2015192879A1), ils estiment le centre du globe, son rayon, la direction de l'axe optique et la correction anatomique vers l'axe visuel (angle $\kappa$). Leur qualité d'interprétabilité est élevée, mais leur convergence se dégrade en présence de reflets spéculaires sur verres correcteurs ou de flou de mouvement.

### 2.1.2 Méthodes basées apparence de bout en bout

Les réseaux profonds apprennent directement une fonction $f_\theta : I \mapsto \mathbf{g}$ :
- **Entrées œil seul ou multi-régions** : *iTracker* / GazeCapture (Krafka et al., CVPR 2016) associe les deux patchs oculaires, le visage et un masque binaire de grille faciale pour prédire directement un point 2D à l'écran en centimètres ($1{,}71\text{ cm}$ sur téléphone sans calibration, $1{,}34\text{ cm}$ avec calibration). *FullFace* (Zhang et al., CVPRW 2017) introduit une pondération spatiale sur le visage entier ($4{,}80^\circ$ sur MPIIFaceGaze). *CA-Net* (Cheng et al., AAAI 2020) adopte une architecture *coarse-to-fine* combinant visage et yeux ($4{,}10^\circ$ sur MPIIFaceGaze).
- **Backbones et têtes angulaires** : *Gaze360* (Kellnhofer et al., ICCV 2019) introduit un modèle temporel LSTM sur 7 frames prédisant direction et incertitude ($4{,}06^\circ$ sur MPIIGaze, $11{,}04^\circ$ sur Gaze360). *L2CS-Net* (Abdelrahman et al., 2022/2023) remplace la régression pure par une prédiction conjointe classification discrétisée (90 classes de $4^\circ$) + espérance continue par axe, atteignant $3{,}92^\circ$ sur MPIIGaze. *GazeTR-Hybrid* (Cheng & Lu, ICPR 2022) associe ResNet et Transformer ($4{,}00^\circ$ sur MPIIFaceGaze, $11{,}4\text{ M}$ paramètres, $1{,}83\text{ GFLOPs}$). *MobileGaze* (`yakhyo/gaze-estimation`) transpose ce principe sur des backbones mobiles dont `MobileOne-S0` ($4{,}8\text{ Mo}$ ONNX, $12{,}58^\circ$ sur Gaze360).

### 2.1.3 Méthodes hybrides et normalisation projective

Proposée par Sugano et al. (CVPR 2014) et formalisée par Zhang et al. (ETRA 2018), la **normalisation géométrique des données** ramène chaque visage dans le repère d'une caméra virtuelle canonique pointée vers le centre facial à distance constante ($600\text{ mm}$), l'axe horizontal aligné sur les yeux. La révision de 2018 montre qu'en ne transformant le vecteur de regard que par la rotation pure $R_{\text{norm}}$ (sans appliquer la matrice d'échelle $S$ au vecteur directionnel), la précision s'améliore nettement.

On distingue deux paramétrisations de sortie (Cheng et al., TPAMI 2024) :
- **Point de regard 2D (PoG)** : lié à une géométrie capteur-écran fixe (ex. caméra coplanaire de tablette/smartphone).
- **Direction 3D canonique** ($\text{pitch}, \text{yaw}$) couplée à une origine 3D oculaire : indépendante du moniteur, elle permet l'intersection géométrique rayon–plan sur n'importe quel écran de bureau.

## 2.2 Synthèse chiffrée : intra-domaine vs généralisation inter-domaines

| Modèle / Travail | MPIIFaceGaze / MPIIGaze | EYEDIAP | Gaze360 | Remarque méthodologique |
|---|---:|---:|---:|---|
| FullFace (Zhang et al., 2017) | $4{,}80^\circ$ | — | — | Visage complet, pondération spatiale |
| Gaze360 (Kellnhofer et al., 2019) | $4{,}06^\circ$ | $5{,}36^\circ$ | $11{,}04^\circ$ | LSTM 7 frames + incertitude |
| CA-Net (Cheng et al., 2020) | $4{,}10^\circ$ | $5{,}27^\circ$ | $12{,}26^\circ$ | Raffinement visage $\to$ yeux |
| GazeTR-Hybrid (Cheng & Lu, 2022) | $4{,}00^\circ$ | $5{,}17^\circ$ | $10{,}62^\circ$ | Hybride CNN + Transformer ($11{,}4\text{ M}$ params) |
| L2CS-Net (Abdelrahman et al., 2022) | $3{,}92^\circ$ | — | $10{,}41^\circ$ | Classification discrétisée + régression |
| OmniGaze (Qu et al., NeurIPS 2025) | $2{,}97^\circ$ | $4{,}07^\circ$ | $9{,}12^\circ$ | Semi-supervisé, pseudo-labels + reward MLLM |
| ETH-XGaze baseline (cross depuis XGaze) | $7{,}5^\circ$ | $11{,}0^\circ$ | $27{,}3^\circ$ | Effondrement hors domaine |
| UniGaze-H (Qin et al., WACV 2026, LODO) | $4{,}51^\circ$ | $5{,}88^\circ$ | $12{,}37^\circ$ | ViT-H ($632\text{ M}$ params), pré-entraînement MAE $1{,}6\text{ M}$ |

Deux résultats récents orientent directement notre conception :
1. **Le gouffre intra/inter-domaines** : entre 2017 et 2022, les gains d'architecture intra-dataset ne représentent que quelques dixièmes de degré ($4{,}80^\circ \to 3{,}92^\circ$), alors que le transfert hors domaine chute entre $5^\circ$ et $12^\circ$ (voire $27{,}3^\circ$). À $60\text{ cm}$, $4^\circ$ équivalent à $4{,}2\text{ cm}$ sur l'écran et $6\text{--}12^\circ$ à $6{,}3\text{--}12{,}8\text{ cm}$.
2. **Vulnérabilité aux lunettes et asymétrie du pitch** : l'étude de Li et al. (Huawei Canada, arXiv:2603.26945, 2026) démontre que même les modèles massifs comme UniGaze-H ($632\text{ M}$ paramètres, incompatibles CPU temps réel) chutent avec lunettes, masques ou faible luminosité, et que l'erreur verticale (*pitch*) diverge davantage entre datasets que l'erreur horizontale (*yaw*). Leur réponse combine des augmentations synthétiques ciblées (lunettes, bruit capteur, éclairage) et un backbone MobileNet compact atteignant une généralisation compétitive avec moins de $1\,\%$ des paramètres d'UniGaze-H.

## 2.3 Personnalisation few-shot, méthode différentielle et calibration implicite

La limite fondamentale d'un modèle générique non calibré provient de l'**angle kappa** ($\kappa$), l'écart anatomique individuel entre l'axe optique du globe (centré sur la pupille/iris) et l'axe visuel passant par la fovéa. Cet angle varie typiquement jusqu'à $\pm 5^\circ$ selon les individus et ne peut être observé sur une image externe sans calibration.

Trois familles de solutions adressent ce verrou :
- **Méta-apprentissage few-shot** : *FAZE* (Park et al., ICCV 2019) apprend un espace latent équivariant en rotation combiné à MAML, descendant à $3{,}18^\circ$ sur GazeCapture avec seulement $k = 3$ échantillons. *MTGLS* passe de $5{,}02^\circ$ à $2{,}22^\circ$ sur MPIIGaze avec 200 échantillons de calibration.
- **Approche différentielle** : proposée par Liu, Yu, Funes Mora & Odobez (BMVC 2018), au lieu d'estimer un angle absolu $\mathbf{g}(I_t)$, un réseau siamois estime la **différence** $\Delta\mathbf{g} = \mathbf{g}(I_t) - \mathbf{g}(I_{\text{ref}})$ entre l'image courante et une ou plusieurs images de calibration de référence. En éliminant l'offset individuel et une part de la géométrie statique par différence relative, cette formulation se prête naturellement au transfert depuis peu d'échantillons ou peu de sujets.
- **Calibration implicite continue** : *WebGazer* (Papoutsaki et al., IJCAI 2016) ajuste une régression Ridge sur les pixels oculaires à chaque clic ou mouvement de curseur ($4{,}17^\circ$). *Online-EYE* (CHI 2025) met à jour la calibration par moindres carrés récursifs (RLS) toutes les trois interactions, convergeant après 8 à 9 clics. *Tri-Cam* (2024) modélise explicitement la désynchronisation entre regard et curseur lors des clics.

## 2.4 Filtrage adaptatif et budget temps réel

Pour la restitution interactive, Casiez et al. (CHI 2012) introduisent le filtre **One Euro** ($1\text{€}$), un passe-bas du premier ordre dont la fréquence de coupure croît linéairement avec la vitesse estimée du signal. Ce mécanisme supprime les micro-oscillations pendant les fixations tout en préservant la réactivité lors des saccades oculaires. En revanche, comme l'implémentation d'`Eyetrack` l'a mis en évidence (`filters.py`), des pics isolés d'erreur neuronale ouvrent artificiellement la bande passante du filtre One Euro via la dérivée estimée — justifiant l'insertion préalable d'un court filtre médian ($w=3$).

## 2.5 Licences des jeux de données et propriété intellectuelle

| Ressource | Volume | Licence | Usage commercial |
|---|---|---|---|
| MPIIGaze / MPIIFaceGaze | 15 sujets, ~45k visages | CC BY-NC-SA 4.0 | Non |
| GazeCapture | 1 474 sujets, ~2M frames | Research purposes only | Non |
| ETH-XGaze | 110 sujets, $>1\text{M}$ images | CC BY-NC-SA 4.0 | Non |
| Gaze360 | 238 sujets, ~172k images | Non-commercial research only | Non |
| EYEDIAP | 16 sujets, RGB + RGB-D | EULA non commercial | Non |
| EVE | 54 sujets, $>12\text{M}$ frames | CC BY-NC-SA 4.0 (code MIT) | Non |
| UniGaze (poids) | ViT B/L/H | ModelGo MG-NC-RAI-2.0 | Non |
| MediaPipe FaceLandmarker | 478 landmarks | Apache 2.0 | Oui |

Parallèlement, les brevets EPFL/Idiap et Eyeware Tech SA (EP3154407B1, EP3953859B1 sur l'estimation de paramètres géométriques oculaires personnels, WO2022084803A1 sur l'auto-calibration) soulignent l'intérêt scientifique et industriel d'une architecture **entièrement entraînée sur données propriétaires ou permissives** sans réutilisation de poids non commerciaux.

C'est précisément ce double constat — **nécessité de personnalisation sous-$1{,}5^\circ$ (Scénario A)** et **contrainte d'indépendance vis-à-vis des datasets NC lorsqu'on ne dispose que d'un seul sujet (Scénario B)** — qui fonde les contributions développées dans les chapitres suivants.
