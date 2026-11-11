# Guide Pratique de Campagne d'Acquisition Solo (Webcam Réelle)

Ce protocole permet de remplacer la validation préliminaire sur banc synthétique (Section 6.4) par vos **vraies mesures webcam multi-sessions** (Section 6.5) en moins de **10 minutes** au total.

## Étape 1 — Enregistrer 4 à 6 sessions Smooth Pursuit (1 min chacune)

Exécutez une session de 60 secondes à la fois en variant volontairement **l'éclairage, la distance et la posture de tête** (bougez doucement la tête en yaw, pitch, roulis et distance pendant le suivi visuel de la bille) :

```powershell
# Session 1 : bureau éclairage normal, avec lunettes
python -m research.collect_pursuit --duration 60 --tag s1_jour_lunettes --lighting jour

# Session 2 : éclairage différent ou lampe latérale
python -m research.collect_pursuit --duration 60 --tag s2_lampe_lunettes --lighting lampe

# Session 3 : sans lunettes (si applicable, pour l'ablation lunettes vs œil nu)
python -m research.collect_pursuit --duration 60 --tag s3_jour_sans_lunettes --no-glasses

# Session 4 : distance plus proche (~45-50 cm)
python -m research.collect_pursuit --duration 60 --tag s4_proche --distance 50

# Session 5 : SESSION DE TEST INDÉPENDANTE (gardée strictement hors entraînement)
python -m research.collect_pursuit --duration 60 --tag s5_test_independant --lighting normal
```

## Étape 2 — Récolte passive pré-saccadique sur clics (optionnel mais recommandé)

Laissez tourner le collecteur silencieux pendant 5–10 minutes de navigation web / bureautique normale :

```powershell
python -m research.collect_clicks --max-clicks 250 --pre-lag-ms 75 --tag bureau_navigation
```

## Étape 3 — Entraînement, estimation du lag et évaluation croisée automatique

Lancez l'évaluation automatique complète (estime $\hat{\tau}_{\text{lag}}$, entraîne `DualEyePersonalNet` et `DifferentialEyeNet`, exporte en ONNX, réalise le test de Wilcoxon et génère `fig6_human_sessions_results.png`) :

```powershell
python -m research.evaluate_human_sessions
```

## Étape 4 — Régénérer le dossier HTML / PDF complet

```powershell
python -m research.build_dossier --pdf
```
