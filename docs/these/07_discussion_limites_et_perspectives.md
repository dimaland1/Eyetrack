# Chapitre 7 — Discussion Critique, Menaces à la Validité et Perspectives

Un dossier de recherche rigoureux exige de distinguer explicitement ce qui a été **démontré sur données humaines réelles**, ce qui a été **vérifié sur banc paramétrique contrôlé**, et les **limites physiques ou méthodologiques** qui subsistent avant une déploiement industriel à grande échelle.

## 7.1 Validité interne, externe et limites du régime mono-sujet

### 7.1.1 Limites de la validation sur banc synthétique pour le Scénario B

Nos ablations inter-sujets du Chapitre 6 ($S_0 \to S_1 \dots S_5$) prouvent mathématiquement qu'une architecture siamoise différentielle munie de randomisation morphologique surmonte la mémorisation d'identité et l'écart d'angle $\kappa$ là où un régresseur direct échoue. Toutefois, le simulateur `research/synthetic_bench.py` modélise la géométrie oculaire de façon idéalisée (iris circulaire, paupières elliptiques symétriques, absence d'occlusion ciliaire complexe ou de double réflexion de monture épaisse).

En conditions humaines réelles sur des sujets distincts, des facteurs non modélisés apparaîtront : ptôse palpébrale marquée, plis épicanthiques, verres progressifs ou fort astigmatisme induisant une distorsion optique locale du limbe. Par conséquent, **l'entraînement sur un seul visage humain réel doit être considéré comme un levier d'amorçage fort**, et non comme un substitut absolu à une validation humaine multi-sujets (même réduite à 5–10 volontaires consentants pour le test externe final).

### 7.1.2 Validité du Smooth Pursuit comme vérité terrain

En poursuite lisse lente ($< 15^\circ/\text{s}$), l'hypothèse d'une erreur humaine intrinsèque $< 0{,}5^\circ$ repose sur une attention soutenue du sujet. En cas de micro-distraction ou de fatigue oculaire au-delà de 60 secondes, des saccades intrusives peuvent introduire des étiquettes bruitées. Deux garde-fous logiciels limitent ce risque :
1. Des sessions courtes de $60\text{ s}$ réparties dans des conditions variées (matin, soir, lampe frontale, lumière tamisée, avec/sans lunettes) ;
2. L'emploi d'une perte robuste (`SmoothL1Loss`) et le rejet automatique des clignements via `BlinkDetector`.

### 7.1.3 Impact des verres correcteurs et du contre-jour

Conformément aux observations de Li et al. (Huawei, 2026) et aux mesures d'`Eyetrack` (`face.py`), la présence de lunettes demeure la première source de variance dans les coins de l'écran : les bords de monture et les reflets d'écran ou de lampe de bureau peuvent recouvrir plus de $15\,\%$ du disque irien. Si l'alerte de reflet pendant la calibration prévient la corruption du jeu de référence, la robustesse en jeu requiert à la fois des augmentations spéculaires réalistes à l'entraînement et une gestion explicite de l'incertitude (pondération du filtre One Euro ou repli temporaire sur la pose 6DoF lorsque le reflet dépasse le seuil critique).

## 7.2 Enjeux juridiques et propriété intellectuelle

L'architecture proposée résout intégralement la dépendance aux poids non commerciaux :
- **Détection faciale** : MediaPipe FaceLandmarker est distribué sous licence **Apache 2.0** (usage commercial autorisé).
- **Modèles de regard (`DualEyePersonalNet` / `DifferentialEyeNet`)** : architectures originales ($< 71\text{ k}$ paramètres) entraînées exclusivement sur les données personnelles consenties de l'auteur et/ou des données synthétiques permissives, sans distillation depuis UniGaze ou Gaze360.
- **Vigilance brevets** : avant toute commercialisation en Europe ou aux États-Unis, l'analyse de liberté d'exploitation devra examiner les revendications exactes des familles EP3953859B1 (estimation de paramètres oculaires géométriques personnels) et WO2022084803A1 (auto-calibration). L'approche purement différentielle par apparence sur patchs normalisés se distingue toutefois des modèles géométriques génératifs du globe oculaire décrits dans le brevet fondateur EP3154407B1.

## 7.3 Feuille de route vers l'intégration produit et la thèse finalisée

Pour transformer ce dossier en système livrable de niveau industriel et académique accompli :

1. **Campagne d'acquisition humaine multi-sessions (J+1 à J+3)** : enregistrer 6 à 10 sessions de 60 secondes (`python -m research.collect_pursuit`) en variant éclairage, posture, distance et port de lunettes, complétées par 500–1 000 clics passifs (`python -m research.collect_clicks`), puis exécuter la ré-évaluation croisée stricte sur session de test indépendante + `tools/validate.py` (cibles fixes 9 points avec mouvement de tête).
2. **Intégration dans `gaze_model.py` et calibration hybride** : ajouter un mode `'personal_onnx'` dans `create_gaze_estimator` remplaçant `MobileOne-S0` ($11{,}16\text{ ms}$) par `DualEyePersonalNet` ($0{,}25\text{ ms}$), libérant plus de $10\text{ ms}$ par frame sur CPU.
3. **Estimation d'incertitude hétéroscédastique** : ajouter une sortie de variance $\hat{\sigma}^2 = (\sigma_\psi^2, \sigma_\theta^2)$ entraînée par log-vraisemblance gaussienne négative pour moduler dynamiquement la fréquence de coupure $f_{c,\min}$ du filtre One Euro et rejeter automatiquement les frames dégradées par un reflet.
4. **Portage natif Rust (`ort`)** : tirer parti de l'absence d'opérateurs dynamiques dans `DualEyePersonalNet` (grâce à `AvgPool2d(1, 2)` statique et grille `CoordConv` figée) pour déployer un moteur d'inférence C ABI / Rust à $< 5\text{ ms}$ totaux de bout en bout.

## 7.4 Conclusion générale

Ce travail démontre qu'il est possible de dépasser le compromis classique entre précision, latence CPU et indépendance juridique dans l'estimation du regard par webcam RGB. En remplaçant les lourds réseaux génériques plein-visage par un découplage géométrique bi-oculaire léger, en exploitant l'acquisition continue par poursuite lisse compensée du retard visuo-moteur et le tampon pré-saccadique sur clics, et en résolvant le transfert inter-sujets depuis un unique utilisateur par modélisation différentielle antisymétrique et randomisation morphologique, `Eyetrack` pose les bases rigoureuses d'un système à la fois scientifique, temps réel et librement distribuable.
