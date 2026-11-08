# État de l'art de l'eye tracking par webcam RGB (2014–2026) et pipeline recommandé pour recréer Beam Eye Tracker

Le meilleur choix pour ton projet est un pipeline **hybride** : un réseau de regard **basé apparence**, appliqué à des images **normalisées** (Zhang et al., 2018), qui prédit une direction 3D. On calcule ensuite géométriquement l'intersection de cette direction avec le plan de l'écran, puis on ajoute une **calibration personnelle légère** (5 à 9 points, complétée par une recalibration implicite via les clics) et un **filtre One Euro adaptatif**. Ton ratio d'iris MediaPipe avec régression polynomiale, lui, plafonne vite dès que la tête bouge. Les 1,5° annoncés par Eyeware correspondent à des « conditions idéales » après calibration.\[1\] Ce chiffre est atteignable en usage réel au centre de l'écran, mais il faut compter plutôt 2 à 4° au total. Le principal obstacle à la distribution n'est pas algorithmique : c'est la licence. Presque tous les datasets et poids SOTA sont non commerciaux.\[2\]

## TL;DR

- **Algorithme recommandé** :
  - détection et landmarks MediaPipe (que tu as déjà), puis normalisation de Zhang ;
  - CNN léger basé apparence (MobileOne/ResNet-18 façon L2CS en prototype ; UniGaze ou OmniGaze comme « professeur » pour la recherche uniquement) ;
  - direction 3D + origine 3D de l'œil, intersection rayon–plan écran ;
  - correction personnelle (biais d'axe visuel + régression ridge/GP sur peu de points) mise à jour par les clics ;
  - One Euro / Kalman.

  En personnalisant avec très peu d'échantillons, on passe typiquement de 4–5° (modèle générique) à 2–3,5°, alors qu'un modèle générique non calibré reste à 5–12° hors domaine.
- **SOTA chiffré** :
  - En intra-dataset, on est autour de 3,9–4,0° sur MPIIFaceGaze (L2CS-Net 3,92°, GazeTR 4,00°) et de 2,97° pour OmniGaze (NeurIPS 2025).
  - En cross-dataset, l'erreur reste de 5 à 12° (UniGaze-H, WACV 2026 : 4,51° sur MPII, 5,88° sur EYEDIAP, 12,37° sur Gaze360 en leave-one-dataset-out).
  - En point de regard (PoG) à l'écran, on obtient environ 1,6 cm sur téléphone (AFF-Net) et 2,49° / 2,75 cm en vidéo desktop (EVE).
- **Réalité commerciale** :
  - Gaze360, GazeCapture, MPIIFaceGaze, EYEDIAP, EVE et ETH-XGaze sont tous non commerciaux ou réservés à la recherche, tout comme les poids UniGaze.
  - Pour distribuer, il faut entraîner sur tes propres données (ou du synthétique sous licence permissive), ou bien reposer surtout sur la calibration personnelle à partir de features MediaPipe (Apache 2.0).
  - Le portage Rust avec ONNX Runtime (crate `ort`) est réaliste à 60 fps sur CPU pour un modèle de type MobileOne/ResNet-18.

## Key Findings

1. **La personnalisation pèse plus lourd que l'architecture.** Les gains d'architecture entre 2017 et 2022 se comptent en dixièmes de degré sur MPIIFaceGaze (FullFace 4,80° → GazeTR 4,00° → L2CS 3,92°).\[3\]\[4\] À l'inverse, une calibration personnelle réduit l'erreur de moitié : sur MPII, MTGLS passe de 5,02° à 2,22° avec 200 échantillons, et FAZE obtient 3,18° sur GazeCapture avec seulement 3 échantillons.\[5\]\[6\]
2. **Le vrai problème de 2024–2026, c'est la généralisation de domaine.** Les modèles s'effondrent hors de leur dataset d'entraînement : la baseline ETH-XGaze entraînée sur ETH-XGaze donne 7,5° sur MPIIGaze, 11,0° sur EYEDIAP et 27,3° sur Gaze360.\[7\] La réponse de la recherche passe par le pré-entraînement massif (UniGaze, MAE sur 1,6 M visages), le pseudo-étiquetage (OmniGaze) et l'augmentation ciblée lunettes/éclairage (Huawei, mars 2026).\[8\]\[9\]
3. **L'erreur verticale (pitch) domine.** L'article Huawei 2026 montre que les étiquettes divergent davantage en pitch qu'en yaw d'un dataset à l'autre. Mélanger naïvement les datasets peut même dégrader la généralisation en pitch.\[8\] Pour le gaming, prévois une calibration et un gain verticaux séparés.
4. **Eyeware Beam** vient d'une lignée Idiap (Funes Mora & Odobez) : rectification de la tête en vue frontale à l'aide d'un modèle 3D du visage, estimation du regard œil-dans-tête, puis recomposition avec la pose.\[10\]\[11\] S'y ajoutent des brevets Eyeware sur l'estimation de paramètres géométriques de l'œil et la calibration.\[10\] L'approche actuelle est très probablement hybride, avec un modèle 3D de tête/œil et un réseau d'apparence, mais aucun article Eyeware ne décrit le pipeline webcam actuel.
5. **Les webcams commerciales mesurées de façon indépendante** se situent entre 1,1 et 4,2° : WebGazer 4,17° (IJCAI 2016). Le tableau de la validation publiée en juin 2026 dans *Intelligence-Based Medicine* (vol. 14, article 100384), avec un Tobii Eye Tracker 5 comme référence, donne GazeRecorder à 2,3° RMSE et la méthode Falch & Lohan à 1,5° RMSE (toutes deux avec une Logitech C922), ainsi qu'une méthode basée MediaPipe à 1,13° d'erreur moyenne. La scoping review arXiv 2605.02898 (2026), citant Asghari et al., indique que les webcams « typically show spatial errors of about 3° to 4.5° ». Les trackers IR font environ 0,5–1° : selon Skovsgaard et al. (2011), cité dans « Comparison of Webcam and Remote Eye Tracking » (ETRA '22), le Tobii T60 atteint 0,67°.

## Details

### 1. Taxonomie des approches

**Méthodes géométriques ou basées modèle.**
- Les méthodes PCCR (pupille + reflet cornéen), utilisées par Tobii, demandent un éclairage IR : elles sont hors périmètre.
- En RGB pur, on trouve :
  - (a) le **ratio ou le décalage iris/coins de l'œil** ;
  - (b) les **modèles 3D de globe oculaire** : centre du globe estimé, rayon, centre de l'iris 3D, axe optique, puis correction de l'angle kappa vers l'axe visuel. Le brevet Idiap WO2015192879A1 (Funes Mora & Odobez, 2014) en est un exemple : il cherche l'axe optique/visuel « most likely to produce » l'observation, à partir d'une segmentation probabiliste cornée/sclère/paupière et d'une rectification frontale de la tête.\[10\]
- **Points forts** : peu de données requises, interprétables, et la géométrie explique les mouvements de tête.\[10\]
- **Point faible** : l'iris ne fait qu'une dizaine de pixels à 60 cm sur une webcam 720p, et les paupières, les reflets et les lunettes cassent la localisation.

C'est exactement la limite de ton MVP. Le ratio d'iris mélange rotation de l'œil et pose de la tête, et la régression polynomiale 9 points ne vaut que pour la pose de calibration.

**Méthodes basées apparence.** Un réseau apprend directement image → regard.
- Entrées : œil seul (GazeNet/MPIIGaze), visage + yeux + grille de position (iTracker, GazeCapture, CVPR 2016), visage complet avec spatial weights (FullFace, Zhang et al., CVPRW 2017), ou approches coarse-to-fine visage + yeux (CA-Net, AAAI 2020).\[4\]\[12\]
- Architectures : CNN (ResNet), puis transformers hybrides (GazeTR, Cheng & Lu, ICPR 2022), ViT pré-entraînés (UniGaze).\[4\]\[9\]
- Elles sont robustes au flou et à la basse résolution, mais ont besoin de beaucoup de données étiquetées et généralisent mal entre domaines.\[13\]\[14\]

**Méthodes hybrides.**
- Normalisation géométrique de la pose, puis réseau d'apparence, puis géométrie 3D vers l'écran : c'est le paradigme standard depuis MPIIFaceGaze et la normalisation de Sugano/Zhang.
- On peut aussi utiliser un réseau pour prédire des landmarks de l'œil (paupières, iris, centre du globe) puis appliquer un modèle géométrique (ex. Park et al., ETRA 2018, « Learning to find eye region landmarks »).
- Dans les deux cas, on combine l'invariance géométrique et la robustesse statistique.

**Direction 3D ou point de regard 2D.**
- Les modèles « 3D » prédisent (pitch, yaw) dans l'espace normalisé de la caméra. Ils sont indépendants de l'écran et se transfèrent d'un setup à l'autre.
- Les modèles « 2D PoG » (iTracker, AFF-Net) régressent directement des cm relatifs à la caméra. C'est plus simple, mais lié à la géométrie du device : GazeCapture suppose caméra et écran coplanaires.
- La revue de Cheng et al. (« Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark », arXiv 2104.12668, TPAMI 2024) insiste justement sur le caractère injuste des comparaisons entre ces deux familles et fournit des scripts de conversion.\[15\]
- **Pour un PC avec OpenTrack, privilégie la sortie 3D** : direction + origine par œil, comme ce que Beam expose dans son SDK.\[1\]

### 2. Modèles de référence et SOTA chiffrés

Erreurs angulaires moyennes (°), protocoles standard de la revue de Cheng et al. (leave-one-person-out sur MPIIFaceGaze, 4-fold sur EYEDIAP, split officiel sur Gaze360) :

| Modèle (année, venue) | MPIIGaze / MPIIFaceGaze | EYEDIAP | Gaze360 | Autre | Remarque |
|---|---|---|---|---|---|
| iTracker (Krafka et al., CVPR 2016) | — | — | — | GazeCapture 1,71 cm (tél.) / 2,53 cm (tablette) sans calib., 1,34–2,12 cm avec calib. | PoG 2D, mobile\[16\] |
| FullFace / spatial weights (Zhang et al., CVPRW 2017) | 4,80 | — | — | — | visage complet\[4\] |
| Gaze360 (Kellnhofer et al., ICCV 2019, LSTM 7 frames) | 4,06 | 5,36 | 11,04 (11,4 / 11,1 selon le protocole) | RT-GENE 7,06 | temporel\[17\]\[18\] |
| CA-Net (Cheng et al., AAAI 2020) | 4,1 | 5,27 | 12,26 | — | coarse-to-fine\[17\]\[19\] |
| ETH-XGaze baseline ResNet-50 (Zhang et al., ECCV 2020) | 7,5 (cross) | 11,0 (cross) | 27,3 (cross) | ETH-XGaze within 4,5 | référence têtes extrêmes\[7\]\[20\] |
| EVE / GazeRefineNet (Park et al., ECCV 2020) | — | — | — | EVE 2,49° / 2,75 cm | vidéo + contenu écran\[21\]\[22\] |
| AFF-Net (Bao et al., ICPR 2020/21) | 3,9 cm (MPIIFaceGaze PoG) | — | — | GazeCapture 1,62 cm (tél.) / 2,30 cm (tablette) | PoG 2D\[23\] |
| GazeTR-Hybrid (Cheng & Lu, ICPR 2022) | 4,00 | 5,17 | 10,62 | RT-GENE 6,55 | 11,4 M params\[24\] |
| L2CS-Net (Abdelrahman et al., 2022, arXiv 2203.03339) | 3,92 (MPIIGaze) | — | 10,41 (front 180°), 9,02 (frontal) | — | ResNet-50, classif.+régression par angle\[25\] |
| OmniGaze (Qu et al., NeurIPS 2025) | 2,97 | 4,07 | 9,12 | RT-GENE 5,40, IVGaze 6,72 | semi-supervisé, pseudo-labels + reward MLLM\[26\] |

**Cross-dataset (généralisation de domaine)**, la métrique qui compte vraiment pour un produit :
- **PureGaze** (Cheng et al., AAAI 2022), premier travail à poser la généralisation de domaine en gaze, obtient selon le tableau d'OmniGaze :\[27\]
  - ETH-XGaze → MPII 7,08° ; ETH-XGaze → EYEDIAP 7,48° ;
  - Gaze360 → MPII 9,28° ; Gaze360 → EYEDIAP 9,32°.
- Les méthodes 2023–2024 (Gaze-Consistent AAAI 2023, AGG CVPR 2024, CLIP-Gaze AAAI 2024, LG-Gaze ECCV 2024) se situent entre 5,9 et 7,9°.
- **OmniGaze** atteint 5,07 / 4,84 / 4,95 / 5,75° sur ces mêmes 4 paires.\[26\] Une source secondaire rapporte 3,03° en zero-shot sur MPIIFaceGaze pour la version Large.\[28\]
- **UniGaze** (Qin, Zhang, Sugano, WACV 2026, arXiv 2502.02307) applique un pré-entraînement MAE sur 1,6 M images de visages *normalisées* (CelebV-Text, VFHQ, VGGFace2…), avec un backbone ViT-B/L/H.\[9\] En leave-one-dataset-out (entraînement sur les 4 autres datasets) :
  - UniGaze-H : ETH-XGaze 11,29°, MPII 4,51°, GazeCapture 5,47°, EYEDIAP 5,88°, Gaze360 12,37° ;\[29\]
  - UniGaze-B : 11,78 / 4,73 / 5,86 / 6,31 / 12,41°.\[29\]

  Résultat notable : les ViT génériques (DINO, MoCo) ne battent pas un ResNet-50.\[29\] C'est le pré-entraînement *sur visages normalisés* qui fait la différence.
- **« Real-time Appearance-based Gaze Estimation for Open Domains »** (Li et al., Huawei Canada, arXiv 2603.26945, mars 2026) est, pour ton cas, le travail le plus pertinent de 2026. Il fait trois constats :
  - les modèles SOTA, UniGaze-H compris, échouent avec lunettes, masque ou faible éclairage ;\[8\]
  - l'erreur de pitch est systématiquement plus forte ;\[8\]
  - joindre les datasets naïvement peut nuire.\[8\]

  Leur réponse : augmentation synthétique (lunettes, masques, éclairage, bruit capteur) et multi-tâche (SupCon multi-vue, classification discrétisée, segmentation œil/iris).\[8\] Leur **MobileNet** atteint une généralisation « competitive with » UniGaze-H avec moins de 1 % des paramètres (UniGaze-H : 632 M).\[8\] C'est un *preprint* : les chiffres n'ont pas été évalués par des pairs.
- **Gaze-LLE** (Ryan et al., CVPR 2025) traite du *gaze following*, c'est-à-dire de la cible regardée dans une scène vue par une caméra tierce, avec un décodeur de 2,8 M sur DINOv2 gelé.\[30\] **Ce n'est pas pertinent** pour l'eye tracking écran. Il montre seulement que les foundation models encodent la direction du regard.\[31\]
- Les approches vidéo comme ST-Gaze (arXiv 2512.17673) continuent d'utiliser EVE comme benchmark temporel desktop.\[32\]

**Lecture pour ton projet.** Une erreur intra-domaine de 4° à 60 cm représente environ 4,2 cm à l'écran. Une erreur cross-domaine de 6–12° représente 6 à 13 cm. Sans calibration, aucun modèle générique n'atteint les 1,5° de Beam : la calibration personnelle est indispensable.

### 3. Personnalisation et calibration few-shot

- **Origine du biais personnel** : l'angle kappa entre axe optique et axe visuel varie d'environ ±5° selon les individus. Il explique l'essentiel de l'erreur résiduelle des modèles génériques, et une correction affine ou de biais par utilisateur en récupère une grande partie.
- **FAZE** (Park, De Mello, Molchanov, Iqbal, Hilliges, Kautz, ICCV 2019, arXiv 1905.01941, code NVlabs/few_shot_gaze) combine un encodeur-décodeur à latent rotation-équivariant et du méta-apprentissage (MAML), pour ≤ 9 échantillons. Il obtient **3,18° sur GazeCapture avec k = 3**, soit 19 % de mieux que l'état de l'art d'alors. Avec 4 échantillons sur MPIIGaze, il égale la méthode différentielle à k = 256.\[5\]\[33\]
- **ETH-XGaze** : un fine-tuning de la baseline sur environ 200 échantillons par personne descend vers ~2° (chiffre rapporté par un résumé secondaire, cohérent avec l'article).\[20\]
- **MTGLS** : 5,02° → 2,22° sur MPII avec 200 échantillons de calibration.\[6\]
- **Méthode différentielle** (Liu, Yu, Funes Mora, Odobez, BMVC 2018 / TPAMI, arXiv 1904.09459) : on prédit la *différence* de regard entre l'image courante et des images de référence calibrées.\[34\] C'est une piste de l'écosystème Idiap/Eyeware, très adaptée au few-shot.
- **Combien de points ?**
  - 1 point : corrige le biais global (offset).
  - 3–5 points : biais + gain par axe (affine). C'est le choix de Beam, avec 5 points cliqués.
  - 9 points : affine + légère non-linéarité.
  - Au-delà de 9–16 points, le gain est marginal, sauf si l'on échantillonne aussi *plusieurs poses de tête*, ce qui compte davantage que le nombre de points.
- **Calibration implicite ou continue** :
  - WebGazer (Papoutsaki et al., IJCAI 2016) s'auto-calibre sur les clics et les mouvements de curseur, avec une erreur de 4,17° (175–210 px).\[16\]\[35\]
  - Online-EYE (CHI 2025, XR) affine une matrice de calibration par RLS toutes les 3 interactions et suffit pour du gaze+pinch après 8–9 clics.\[36\]
  - vGaze (IEEE TMC) se calibre via la saillance vidéo.\[37\]
  - Tri-Cam (2024) utilise les clics souris en modélisant l'erreur « regard ≠ curseur ».\[38\]
  - Une revue de Frontiers in Psychology (2024) rapporte 3,45–4,4° pour plusieurs méthodes implicites et juge ce mode « relativement optimal ».\[39\]
- **Adaptation test-time ou de domaine non supervisée** : PnP-GA+, méthodes source-free, « Jitter does matter ». Ce sont des approches intéressantes en recherche mais lourdes en temps réel. Une simple correction de biais en ligne sur clics apporte l'essentiel.

### 4. Robustesse : tête, éclairage, lunettes, distance, géométrie écran

- **Normalisation des données** (Sugano et al., CVPR 2014 ; Zhang, Sugano, Bulling, « Revisiting Data Normalization for Appearance-Based Gaze Estimation », ETRA 2018, code xucong-zhang/data-preprocessing-gaze) :
  - on fait tourner une caméra virtuelle pour qu'elle pointe vers le centre du visage (ou de l'œil) avec l'axe x aligné sur la tête ;
  - on met à l'échelle à une distance fixe (ex. 600 mm, avec une focale virtuelle) ;
  - on applique la warp perspective, et le réseau prédit le regard dans cet espace ;
  - on dé-normalise en multipliant par R⁻¹.

  La version 2018 **ne tourne que le vecteur de regard** (g_n = R·g_r) sans appliquer l'échelle, ce qui améliore la précision.\[40\] UniGaze pré-entraîne justement sur ces images normalisées.\[9\] Dans ton cas, la pose fournie par la matrice de transformation MediaPipe (ou un solvePnP sur un modèle 3D moyen) sert directement à construire R.
- **Estimation conjointe tête + regard** : la normalisation découple en partie les deux, mais la pose reste une entrée utile (MPIIGaze injecte la pose normalisée). Pour OpenTrack, la pose 6DoF de MediaPipe suffit. Il faut la filtrer séparément, et Beam expose d'ailleurs la pose de tête et le regard comme des flux distincts.
- **Éclairage et lunettes** : c'est la première cause d'échec en conditions réelles (Huawei 2026).\[8\] Les remèdes sont l'augmentation photométrique, la synthèse de lunettes cohérente avec la pose, et un éclairage frontal recommandé à l'utilisateur. Le contre-jour et les reflets sur les verres restent mal résolus, alors que l'IR de Tobii y résiste mieux.
- **Distance et résolution** : Beam spécifie 30–80 cm et conseille, au-delà de 1080p, de « prefer increasing frame rate over resolution ». Il déconseille aussi le cadrage automatique des webcams, qui casse la calibration.\[1\]\[41\]\[42\] Pour le réseau, ce qui compte est la résolution du visage normalisé (224×224 ou 448×448), donc avoir 100 px ou plus entre les deux yeux.
- **Passer de la direction 3D au point écran** :
  1. origine o = centre 3D de l'œil (ou point milieu entre les deux yeux) dans le repère caméra, à partir des landmarks MediaPipe mis à l'échelle métrique (distance interpupillaire ≈ 63 mm ou iris ≈ 11,7 mm de diamètre) ;
  2. direction d = R⁻¹·g_n ;
  3. intersection avec le plan écran, défini par la pose écran ↔ caméra (pour un laptop ou une webcam posée sur l'écran : plan z = 0 décalé, avec la taille physique de l'écran fournie par l'OS/EDID) ;
  4. conversion des mm en pixels.

  Les erreurs d'extrinsèques écran-caméra sont *absorbées* par la calibration personnelle, qui peut aussi estimer conjointement un petit offset de la pose de l'écran (approche « calibration système » des brevets Eyeware). EVE et EFE (arXiv 2305.05526) apprennent l'origine et la direction de bout en bout à partir de la frame brute.\[43\]

### 5. Lissage temporel et latence

- **Filtres** :
  - le **One Euro** (Casiez et al., CHI 2012), que tu as déjà, est le bon défaut pour le pointage : sa coupure adaptative à la vitesse laisse passer les saccades avec peu de lag ;
  - un **Kalman** à vitesse constante (ou IMM fixation/saccade) est utile pour fusionner tête et regard et pour prédire une frame à l'avance, afin de compenser la latence.

  Beam documente lui-même ce compromis : « Higher values [of smoothing] are smoother but add latency ».\[44\]
- **Contrôle caméra en jeu** : il faut une **zone morte et une courbe de réponse** sur le regard (ex. pas de rotation tant que le regard reste dans les 60 % centraux), ce qui est plus important que la précision brute. Les tremblements de 1–2° deviennent invisibles.
- **Modèles temporels** : le LSTM de Gaze360 (7 frames), le GRU de GazeRefineNet (EVE) et ST-Gaze améliorent la précision de quelques dixièmes à 28 % (EVE, avec le contenu écran).\[22\]\[45\] En contrepartie, ils ajoutent latence et complexité. Pour du 60 fps, mieux vaut un modèle image par image avec un filtre explicite. Le temporel est à envisager plus tard.
- **Budget de latence à 60 fps** (16,7 ms par frame) :
  - MediaPipe Face Landmarker : quelques ms sur CPU ;
  - warp de normalisation : < 1 ms ;
  - CNN de regard : MobileOne-S0 (4,8 MB ONNX) ou ResNet-18 (~1,8 GFLOPs à 224²) tiennent quelques ms sur CPU AVX2 et < 2 ms sur GPU ; GazeTR fait 1,83 GFLOPs.\[46\]

  UniGaze-H (632 M params) et les ViT-L/H sont **inutilisables en temps réel sur CPU** : garde-les comme professeurs pour la distillation.\[8\] L'exposition de la webcam limite aussi la fréquence en faible lumière, car beaucoup de webcams tombent à 30 fps.\[41\]

### 6. Eyeware Beam : ce qu'on sait et ce qu'on peut déduire

- **Specs publiques** (beam.eyeware.tech/developers) :
  - précision **1,5°** en « Ideal conditions », qui « decreases towards the edges of the screen and especially for larger screens » ;\[1\]
  - performance réduite au-delà de 30" ;\[1\]
  - distance de fonctionnement 30–80 cm ;\[1\]
  - tête 6DoF, vecteur de regard 3D (origine + direction) par œil, PoG 2D multi-écrans ;\[1\]
  - fréquence « hardware-limited » : 60 fps ou plus avec une webcam compatible (Steam) ;\[1\]\[47\]
  - calibration à **5 points cliqués** (« Move your head if needed, but let your eyes do most of the work ») ;\[42\]
  - application iOS utilisant la caméra TrueDepth pour une calibration plus robuste et une charge CPU quasi nulle ;\[41\]
  - traitement local sans stockage de données biométriques ;\[48\]
  - Windows, puis Linux et macOS en v2.6 / SDK 2.2.0.\[49\]

  **Aucune latence en ms ni aucune précision mesurée indépendamment n'est publiée.** Un plugin tiers évoque « < 16 ms », sans vérification possible.\[50\]
- **Lignée scientifique** :
  - Eyeware (fondée en 2016) est une spin-off de l'EPFL et de l'Idiap. Le CEO Kenneth Funes Mora a fait sa thèse sur le regard 3D avec capteurs RGB-D et a créé EYEDIAP (ETRA 2014).\[51\]\[52\]
  - Publications clés : « Gaze Estimation from Multimodal Kinect Data » (CVPRW 2012) ; G3E (CVPR 2014) ; « Gaze Estimation in the 3D Space Using RGB-D Sensors » (IJCV 2016) ; la méthode différentielle (BMVC 2018) ; HeadFusion (PAMI 2018) pour le suivi de tête 360° par 3DMM.\[11\]\[34\]\[53\]\[54\]\[55\]
  - Leur principe constant : suivre la tête avec un modèle 3D personnalisé, rectifier l'œil en vue frontale, estimer le regard œil-dans-tête, puis recomposer avec la pose.\[56\]
- **Brevets** :
  - WO2015192879A1 / EP3154407B1 (Idiap, 2014) sur le modèle géométrique génératif de l'œil.\[10\]
  - Eyeware Tech SA : WO2020208494A1 / US20220148333A1 / EP3953859B1 (estimation des paramètres géométriques de l'œil d'un utilisateur, EP délivré en 2024) ; WO2022084803A1 (calibration automatique tracker externe ↔ ordinateur) ; WO2022149094A1 (capture temps réel du regard sur un écran) ; EP4027220A1 (calibration) ; EP4113982A1 / EP4113983A1 (visioconférence).\[10\]
  - Je n'ai pu consulter que les titres de ces familles, pas leurs revendications. **Avant de distribuer, fais vérifier par un conseil en propriété industrielle les revendications sur l'estimation des paramètres géométriques personnels et sur la calibration automatique**, surtout pour une commercialisation en Europe.
- **Approche probable** (déduction) : suivi de tête 3DMM, normalisation, réseau d'apparence (passage au deep learning après 2018), paramètres personnels (kappa, rayon du globe, position) estimés pendant la calibration, puis géométrie vers l'écran. C'est cohérent avec la sortie « origine + direction par œil » et avec la dégradation sur les bords et les grands écrans, typique d'une erreur angulaire constante projetée.
- **Concurrents et références** :
  - **Tobii** (IR, PCCR) : 0,67° pour le Tobii T60 selon Skovsgaard et al. (2011), cité dans « Comparison of Webcam and Remote Eye Tracking » (ETRA '22) ; Tobii 4c < 1°. C'est la référence matérielle, robuste en faible lumière.
  - **WebGazer.js** : 4,17° (IJCAI 2016), régression ridge sur les pixels de l'œil + clics.\[16\]\[35\]
  - **GazeRecorder** : 2,3° RMSE et méthode Falch & Lohan : 1,5° RMSE (toutes deux avec une Logitech C922) ; une méthode basée MediaPipe : 1,13° d'erreur moyenne (référence Tobii ET5, tableau de « Validation of webcam-based eye-tracking for clinically relevant paradigms », *Intelligence-Based Medicine*, vol. 14, juin 2026). Les auteurs de cette validation estiment l'erreur à environ 2,4–3,7° en déploiement non contrôlé.
  - **RealEye** : environ 100–125 px plein écran.\[57\]
  - **OpenFace 2.0** (Baltrušaitis, Zadeh, Lim, Morency, FG 2018) : 9,10° d'erreur en cross-dataset sur MPIIGaze selon le tableau V de l'article, et 26,7° sur Gaze360 selon OpenFace 3.0 (arXiv 2506.02891). Licence académique non commerciale, à vérifier.
  - **MediaPipe Iris / Face Landmarker** (Apache 2.0) : landmarks seulement, sans regard calibré.
  - **GazeTracking** (antoinelame) : ratio de pupille OpenCV, de niveau proche de ton MVP.
  - **OpenSeeFace** : suivi de tête/visage pour VTubing, sans eye tracking écran de précision.
  - **EyeTrax** (MIT) : routines de calibration et lissage Kalman/EMA.\[2\]

### 7. Datasets, poids pré-entraînés et licences

| Dataset | Taille | Licence | Usage commercial |
|---|---|---|---|
| MPIIGaze / MPIIFaceGaze | 15 sujets, ~45 K images visage (213 K au total) | CC BY-NC-SA 4.0 | Non\[58\]\[59\] |
| GazeCapture | 1 474 sujets, ~2 M frames | « research purposes only » | Non\[60\]\[61\] |
| ETH-XGaze | 110 sujets, > 1 M images, têtes ±80° | CC BY-NC-SA 4.0 + conditions ; labels test publiés en juin 2026 | Non\[20\]\[62\]\[63\] |
| Gaze360 | 238 sujets, ~172 K images, 360° | « non-commercial research use only » | Non\[64\]\[65\] |
| EYEDIAP | 16 sujets, RGB + RGB-D | EULA « strictly restricted to non-commercial research » (signataire en poste permanent) | Non\[66\] |
| EVE | 54 sujets, > 12 M frames, 4 vues, Tobii Pro Spectrum 150 Hz | CC BY-NC-SA 4.0 + conditions (le code du repo est MIT) | Non\[21\]\[22\]\[32\]\[43\]\[67\] |
| GazeGene (synthétique) | 56 identités, ~1 M | à vérifier | ? |

| Modèle | Poids | Licence | ONNX |
|---|---|---|---|
| UniGaze B/L/H (`pip install unigaze`) | HF, plusieurs variantes « joint » et « cross_X » | **ModelGo MG-NC-RAI-2.0 (non commercial)** | export possible (ViT) mais lourd\[68\]\[69\] |
| L2CS-Net | ResNet-50, Gaze360/MPIIGaze | code MIT, **poids issus de datasets NC** | export facile\[2\]\[70\] |
| yakhyo/gaze-estimation (MobileGaze) | ResNet-18/34/50, MobileNetV2, MobileOne S0–S4 ; Gaze360 MAE 11,34° (R50) à 12,58° (MobileOne-S0, 4,8 MB) | code MIT, poids entraînés sur Gaze360 (NC) | **ONNX fourni**\[2\]\[71\]\[72\] |
| ETH-XGaze baseline | ResNet-50 | CC BY-NC-SA 4.0 | export facile\[62\] |
| FAZE (NVlabs) | DT-ED + MAML | licence NVIDIA (non commerciale) | partiel |
| ptgaze (hysts) | MPIIGaze ResNet-preact 5,73°, ETH-XGaze | code MIT, poids NC | oui\[73\] |

**Conséquence.** Pour un projet perso, tu peux tout utiliser. Pour une distribution, même gratuite, les poids entraînés sur ces datasets posent un risque juridique, puisque la licence NC s'étend généralement aux modèles dérivés.\[2\] Tu as trois options propres :
- (a) collecter ton propre dataset avec consentement (dizaines à centaines de sujets, cibles à l'écran, plusieurs poses et éclairages, idéalement avec un Tobii ET5 comme vérité terrain) ;
- (b) entraîner sur du synthétique sous licence permissive ;
- (c) un mode « sans réseau » où seuls MediaPipe (Apache 2.0) et des features géométriques et d'apparence alimentent un régresseur **entraîné uniquement sur la calibration de l'utilisateur**, avec ses propres données donc sans problème de licence.

## Recommendations

### Pipeline cible

1. **Capture** : 720p/1080p à 60 fps, exposition fixe, autofocus et cadrage automatique désactivés. Avertir l'utilisateur en cas de contre-jour.
2. **Visage, landmarks et pose** : MediaPipe Face Landmarker, avec 478 points et la matrice de transformation, filtrée par One Euro sur la pose. Mise à l'échelle métrique via le diamètre de l'iris. La pose 6DoF part vers OpenTrack (UDP), comme aujourd'hui.
3. **Normalisation Zhang 2018** : visage 224×224 (ou les deux yeux 36×60 / 64×96) à distance virtuelle fixe ; matrice R conservée.
4. **Réseau de regard** :
   - *Prototype (perso)* : MobileOne-S0/S2 ou ResNet-18 ONNX de yakhyo, ou L2CS. Les fine-tuner sur ETH-XGaze + MPIIFaceGaze + GazeCapture, avec labels harmonisés.
   - *Recherche* : distiller UniGaze-B/H ou OmniGaze vers un MobileNet/MobileOne, en reprenant les recettes Huawei 2026 (augmentation lunettes/éclairage, classification discrétisée façon L2CS, segmentation auxiliaire). Sorties : (pitch, yaw) et **incertitude**, à utiliser pour pondérer le filtre.
   - *Distribution* : réentraîner la même architecture sur ton dataset maison.
5. **Géométrie** : dé-normalisation R⁻¹, rayons par œil depuis les centres 3D des yeux, intersection avec le plan écran, en mm puis en px.
6. **Calibration personnelle** :
   - 5–9 points cliqués, *en demandant de bouger la tête entre les points* ;
   - modèle = biais angulaire par œil (kappa) + affine en (pitch, yaw), éventuellement ridge sur features [regard prédit, pose, position de l'œil] ;
   - **mise à jour continue** par RLS sur les clics et les fixations stables (rejet des paires regard-clic trop éloignées, comme Tri-Cam / brevet « Gaze calibration »).
7. **Lissage et contrôle** : One Euro (ou Kalman sur l'état [position, vitesse]) avec détection de saccade. Pour la caméra de jeu : zone morte, courbe et gain vertical séparé.
8. **Sorties** : UDP OpenTrack (tête + éventuellement « regard → yaw/pitch additionnel »), overlay transparent du PoG, puis SDK (C ABI + bindings Python) exposant les mêmes flux que Beam : pose 6DoF, origine/direction par œil, PoG.

### Gains attendus par rapport au MVP

| Étape | Erreur typique attendue | Commentaire |
|---|---|---|
| MVP ratio iris + poly 9 pts | ~2–4° tête immobile, se dégrade fortement au moindre mouvement de tête | sensible aux paupières et au pitch |
| + modèle d'apparence générique (sans calib.) | 4–6° en domaine proche, 6–12° hors domaine | robuste à la pose, mais biaisé |
| + calibration 5–9 pts | ~2–3,5° | proche de l'écart FAZE / MTGLS |
| + recalibration implicite + multi-poses + filtrage | ~1,5–2,5° au centre, plus sur les bords | niveau Beam « idéal » au centre |

Ces fourchettes sont des **estimations** déduites des benchmarks cités, pas des mesures sur ton setup. Mesure-les toi-même avec un protocole de 9–16 cibles de validation à plusieurs poses de tête. Un Tobii ET5 d'occasion donnerait une vérité terrain si tu veux comparer à Beam.

### Portage Rust

- **Faisable et recommandé** :
  - `ort` (bindings ONNX Runtime) avec les execution providers CPU, CUDA, DirectML ou CoreML ;
  - `nokhwa` ou des bindings MediaFoundation/V4L2 pour la capture ;
  - `nalgebra` pour la géométrie, les rotations et le RLS.
- **Point dur** : MediaPipe n'a pas de binding Rust officiel. Deux options :
  - (a) exporter les modèles TFLite (BlazeFace + Face Mesh/Attention) en ONNX et réimplémenter le post-traitement (anchors, ROI tracking, solve de la matrice de pose) ;
  - (b) détecteur de visage ONNX (SCRFD/YuNet) + landmarks ONNX + `solvePnP` sur un modèle 3D moyen.
- **Performances** :
  - le pipeline détection + landmarks + CNN léger tient sous 10 ms par frame sur un CPU AVX2 récent, en monothread pour le réseau ;
  - pipeliner sur 2 threads (capture/détection ‖ regard/filtre) ;
  - pré-allouer les tenseurs ;
  - quantifier en INT8 avec prudence, car la précision angulaire est sensible à la quantification : mesure avant/après.

### Feuille de route

1. **S1–S2** : banc de mesure (cibles de validation, enregistrement des erreurs en ° et en cm, conditions d'éclairage, lunettes). Sans métrique, tu ne sauras pas si tu bats ton MVP.
2. **S3–S4** : normalisation Zhang + modèle ONNX yakhyo/L2CS + géométrie rayon-écran + calibration affine 5–9 points. Comparer au MVP.
3. **S5–S6** : recalibration implicite RLS sur les clics, gain vertical séparé, zone morte et courbes pour le gaming, overlay.
4. **M2–M3** : fine-tuning multi-datasets + augmentation lunettes/éclairage, distillation depuis UniGaze. Sortie d'incertitude.
5. **M3–M4** : portage Rust (`ort`), SDK C ABI, bindings Python.
6. **Avant distribution** : dataset maison sous consentement, réentraînement, audit des licences et des brevets Eyeware.

## Caveats

- **Chiffres non comparables entre eux** : protocoles (leave-one-out, splits, prétraitement de Gaze360 et EYEDIAP), unités (° ou cm) et conditions diffèrent. L'article Huawei 2026 note que les labels d'EYEDIAP varient selon les études. Les résultats « joint » d'UniGaze sont évalués sur des portions de datasets vus à l'entraînement.\[8\]
- Les **1,5° de Beam** sont une spec constructeur « ideal conditions », sans validation indépendante publiée trouvée.\[1\] Les comparaisons webcam/Tobii publiées viennent de setups contrôlés.
- **OmniGaze et l'article Huawei** sont récents : Huawei n'est qu'un preprint, passé en v2 le 1er octobre 2026, qui renvoie vers un dépôt GitHub (github.com/liszth87/GazeTorch) ; je n'ai pas vérifié le contenu ni la licence de ce dépôt. Le chiffre de 3,03° zero-shot d'OmniGaze-L vient d'un résumé secondaire.
- **Brevets Eyeware** : seuls les titres ont été consultés. Ce rapport n'est pas un avis juridique.
- Les licences d'OpenFace, OpenSeeFace, GazeTracking et GazeGene sont à revérifier sur leurs dépôts avant usage.

## Sources

1. [Beam Eye Tracker SDK](https://beam.eyeware.tech/developers/)
2. [github.com](https://github.com/RealEye-io/community/discussions/3)
3. [L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments](https://deepai.org/publication/l2cs-net-fine-grained-gaze-estimation-in-unconstrained-environments)
4. [An End-to-End Review of Gaze Estimation and its Interactive Applications on Handheld Mobile Devices](https://arxiv.org/html/2307.00122)
5. [\[1905.01941\] Few-Shot Adaptive Gaze Estimation](https://arxiv.org/abs/1905.01941)
6. [MTGLS: Multi-Task Gaze Estimation with Limited Supervision](https://arxiv.org/pdf/2110.12100)
7. [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation](https://arxiv.org/pdf/2007.15837)
8. [Real-time Appearance-based Gaze Estimation for Open Domains](https://arxiv.org/html/2603.26945)
9. [UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training](https://arxiv.org/html/2502.02307v1)
10. [WO2015192879A1 - A gaze estimation method and apparatus - Google Patents](https://patents.google.com/patent/WO2015192879A1/en)
11. [Gaze Estimation from Multimodal Kinect Data](https://www.idiap.ch/~odobez/publications/FunesOdobez-CVPR-WGR-2012.pdf)
12. [Eye Tracking for Everyone](https://arxiv.org/html/1606.05814v1.pdf)
13. [Jitter Does Matter: Adapting Gaze Estimation to New Domains](https://arxiv.org/pdf/2210.02082)
14. [Eye gaze estimation: A survey on deep learning-based approaches - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417422003347)
15. [\[2104.12668\] Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://arxiv.org/abs/2104.12668)
16. [Attention, not scale, drives human-AI alignment in multimodal language prediction](https://arxiv.org/pdf/2308.06035)
17. [gaze eye screen pointer](https://github.com/Stepan5024/gaze__eye_screen_pointer)
18. [OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](https://arxiv.org/html/2510.13660)
19. [GazeSymCAT: A symmetric cross-attention transformer for robust gaze estimation under extreme head poses and gaze variations](https://academic.oup.com/jcde/article/12/3/115/8003774)
20. [ETH-XGaze: A Large Scale Dataset for Gaze Estimation Under Extreme Head Pose and Gaze Variation — Lacuna](https://lacuna.tiptreesystems.com/work/eth-xgaze-a-large-scale-dataset-for-gaze-estimation-under-extreme-head-pose-and/wrk_d09349ce163217c720346358c32d9294)
21. [Towards End-to-end Video-based Eye-Tracking](https://arxiv.org/pdf/2007.13120)
22. [Towards End-to-end Video-based Eye-Tracking](https://ait.ethz.ch/eve)
23. [Introduction](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.796825/xml/nlm)
24. [Gaze Estimation using Transformer Yihua Cheng Beihang University](https://arxiv.org/pdf/2105.14424)
25. [L2CS-Net](https://models.luxonis.com/luxonis/l2cs-net/7051c9d2-78a4-420b-91a8-2d40ecf958dd?backTo=%2F)
26. [OMNIGAZE: Reward-inspired Generalizable Gaze Estimation in the Wild](https://web3.arxiv.org/pdf/2510.13660)
27. [(PDF) Deep Learning-Based Appearance-Based Gaze Estimation](https://www.researchgate.net/publication/399198815_Deep_Learning-Based_Appearance-Based_Gaze_Estimation)
28. [OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild — Lacuna](https://lacuna.tiptreesystems.com/work/omnigaze-reward-inspired-generalizable-gaze-estimation-in-the-wild/wrk_4b626075287df30d92162cc1da1ffbc1)
29. [UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training](https://arxiv.org/pdf/2502.02307)
30. [Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders](https://www.alphaxiv.org/abs/2412.09586)
31. [Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders Fiona Ryan1](https://openaccess.thecvf.com/content/CVPR2025/papers/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.pdf)
32. [Learning Spatio-Temporal Feature Representations for Video-Based Gaze Estimation](https://arxiv.org/pdf/2512.17673)
33. [Few-Shot Adaptive Gaze Estimation \[Quick Review\]](https://liner.com/review/fewshot-adaptive-gaze-estimation)
34. [A Differential Approach for Gaze Estimation](https://arxiv.org/pdf/1904.09459)
35. [WebGazer: Scalable Webcam Eye Tracking Using User Interactions](https://cs.brown.edu/people/apapouts/papers/ijcai2016webgazer.pdf)
36. [Online-EYE: Multimodal Implicit Eye Tracking Calibration for XR](https://dl.acm.org/doi/10.1145/3706598.3713461)
37. [Continuous Gaze Tracking With Implicit Saliency-Aware Calibration on Mobile Devices](https://dl.acm.org/doi/abs/10.1109/TMC.2022.3185134)
38. [Tri-Cam: Practical Eye Gaze Tracking via Camera Network](https://arxiv.org/html/2409.19554v1)
39. [Frontiers](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1309047/full)
40. [Revisiting Data Normalization for Appearance-Based Gaze Estimation - Max Planck Institute for Informatics](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation)
41. [Webcams](https://beam.eyeware.tech/webcams/)
42. [Get Started](https://beam.eyeware.tech/get-started/)
43. [EFE: End-to-end Frame-to-Gaze Estimation](https://arxiv.org/pdf/2305.05526)
44. [Microsoft Flight Simulator 2024 (MSFS2024) Head & Eye Tracking](https://beam.eyeware.tech/games/microsoft-flight-simulator-2024-msfs2024/)
45. [Towards End-to-end Video-based Eye-Tracking](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570732.pdf)
46. [Lightweight Gaze Estimation Model Via Fusion Global Information Zhang Cheng](https://arxiv.org/pdf/2411.18064)
47. [Beam Eye Tracker on Steam](https://store.steampowered.com/app/2375780/Beam_Eye_Tracker/)
48. [Beam Eye Tracker](https://www.aerosoft.com/us/shop/flight/p3d-fsx/flight-simulator-x/tools-missions/4657/beam-eye-tracker)
49. [Steam Community :: Beam Eye Tracker](https://steamcommunity.com/app/2375780)
50. [github.com](https://github.com/Millsy102/BEAMPluginTesting)
51. [About Eyeware Tech SA](https://eyeware.tech/about/)
52. [EYEDIAP — EN](https://www.idiap.ch/dataset/eyediap)
53. [Geometric Generative Gaze Estimation (G3E) for Remote RGB-D Cameras](https://openaccess.thecvf.com/content_cvpr_2014/papers/Mora_Geometric_Generative_Gaze_2014_CVPR_paper.pdf)
54. [360 Head Pose tracking combining 3D Morphable Model and ...](https://www.idiap.ch/~odobez/publications/YuFunesOdobez-PAMI2018.pdf)
55. [(PDF) Gaze Estimation in the 3D Space Using RGB-D Sensors](https://www.researchgate.net/publication/284136543_Gaze_Estimation_in_the_3D_Space_Using_RGB-D_Sensors)
56. [\[PDF\] Gaze Estimation in the 3 D Space Using RGB-D sensors Towards Head-Pose And User Invariance](https://www.semanticscholar.org/paper/Gaze-Estimation-in-the-3-D-Space-Using-RGB-D-And-Funes-Mora-Odobez/1b0d295343b0931e7e6f4eaed14e398af65f5717)
57. [What Shapes Participant Data Quality? A Scoping Review and Case Study of Crowdsourced Webcam Eye Tracking in AI Interviews](https://arxiv.org/pdf/2605.02898)
58. [Appearance-Based Gaze Estimation in the Wild - Max Planck Institute for Informatics](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)
59. [Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://arxiv.org/pdf/2104.12668)
60. [GitHub - CSAILVision/GazeCapture: Eye Tracking for Everyone · GitHub](https://github.com/CSAILVision/GazeCapture)
61. [Appearance-based Gaze Estimation with Deep Learning: A Review and Benchmark](https://arxiv.org/html/2104.12668v2)
62. [GitHub - xucong-zhang/ETH-XGaze: Official implementation of ETH-XGaze dataset baseline · GitHub](https://github.com/xucong-zhang/ETH-XGaze)
63. [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation](https://ait.ethz.ch/xgaze)
64. [GitHub - erkil1452/gaze360: Code for the Gaze360: Physically Unconstrained Gaze Estimation in the Wild Dataset · GitHub](https://github.com/erkil1452/gaze360)
65. [Gaze360: Physically Unconstrained Gaze Estimation in the Wild](https://arxiv.org/pdf/1910.10088)
66. [EYEDIAP](https://zenodo.org/records/4467455)
67. [GitHub - swook/EVE: Towards End-to-end Video-based Eye-tracking. ECCV 2020. https://ait.ethz.ch/eve](https://github.com/swook/EVE)
68. [github.com](https://github.com/ut-vision/unigaze)
69. [GitHub - ut-vision/UniGaze: Official PyTorch implementation of the paper: UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training. · GitHub](https://github.com/ut-vision/UniGaze)
70. [GitHub - Ahmednull/L2CS-Net: The official PyTorch implementation of L2CS-Net for gaze estimation and tracking · GitHub](https://github.com/ahmednull/l2cs-net)
71. [Releases · yakhyo/gaze-estimation · GitHub](https://github.com/yakhyo/gaze-estimation/releases)
72. [MobileGaze MobileOne S0 MLX fp32](https://huggingface.co/majentik/MobileGaze-MobileOne-S0-MLX-fp32)
73. [ptgaze MPIIGaze model (resnet\_preact)](https://huggingface.co/hysts/ptgaze-mpiigaze-resnet-preact)
