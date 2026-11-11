# Chapitre 5 — Architectures Proposées : Du Modèle Personnel Sous-1,5° au Réseau Différentiel Transférable

Ce chapitre détaille les deux architectures légères développées dans `research/models.py` et `research/augmentations.py` pour répondre respectivement au **Scénario A** (précision maximale sur un utilisateur unique) et au **Scénario B** (modèle entraîné sur un seul sujet mais transférable à de nouveaux utilisateurs sans dépendance à des poids non commerciaux).

## 5.1 Principes de conception communs

Les architectures CNN généralistes comme `MobileOne-S0` plein-visage ($448 \times 448$) ou `ResNet-50` répondent mal aux contraintes d'un tracker temps réel sur CPU bureautique :
1. Leur coût d'inférence ($11{,}16\text{ ms}$ pour `MobileOne-S0` sur le poste de référence) laisse peu de marge sous le seuil des $60\text{ Hz}$ ($16{,}67\text{ ms}$) une fois additionnés la capture, MediaPipe et le rendu du jeu ;
2. Le *Global Average Pooling* (GAP) terminal, classique en classification d'images, détruit l'invariance/équivariance spatiale fine : savoir *où* se situe précisément le limbe irien par rapport aux commissures oculaires devient plus difficile ;
3. L'entrée plein-visage facilite le sur-apprentissage du fond ou de la coiffure lorsque le jeu d'entraînement provient d'un seul individu.

Nous introduisons donc un encodeur léger `SpatialEyeEncoder` partageant trois choix structurels :

### 5.1.1 Encodage explicite des coordonnées (`CoordConv2d`)

Dès la première couche (`research/models.py:CoordConv2d`), deux canaux de grille statiques normalisés $(x, y) \in [-1, 1]^2$ sont concaténés à l'image oculaire $48 \times 96$. Conformément à Liu et al. (NeurIPS 2018), cet apport rompt la symétrie de translation des convolutions standard et permet aux filtres précoces d'associer directement la frontière sclère–iris à sa coordonnée horizontale et verticale relative aux coins de l'œil.

### 5.1.2 Blocs séparables en profondeur et préservation topologique $3 \times 3$

Après un *stem* `CoordConv2d` de pas 2 ($24 \times 48$), trois étages de convolutions séparables en profondeur (`DSConvBlock`, convolution *depthwise* $3 \times 3$ + *BatchNorm* + *SiLU* + *pointwise* $1 \times 1$ avec connexion résiduelle à résolution constante) réduisent la carte d'activation à $3 \times 6$. Plutôt qu'un GAP global ($1 \times 1$), un sous-échantillonnage `AvgPool2d(1, 2)` produit une **grille spatiale $3 \times 3$** ($9$ régions locales correspondant aux secteurs haut/milieu/bas $\times$ gauche/centre/droite de l'œil) avant la projection linéaire vers un vecteur latent $z \in \mathbb{R}^{64}$. Le choix exact de `AvgPool2d(1, 2)` sur $3 \times 6$ garantit également une exportation ONNX statique sans opérateur adaptatif dynamique.

## 5.2 Scénario A : `DualEyePersonalNet` (Hyper-personnalisation)

### 5.2.1 Formulation et couplage géométrique

`DualEyePersonalNet` (`70\,482` paramètres entraînables) exploite un encodeur `SpatialEyeEncoder` siamois à 3 canaux RGB sur les deux patchs oculaires alignés $(I_L, I_R) \in \mathbb{R}^{2 \times 3 \times 48 \times 96}$ :
$$\mathbf{z}_L = E_\theta(I_L), \qquad \mathbf{z}_R = E_\theta(I_R) \in \mathbb{R}^{64}$$
Parallèlement, le vecteur géométrique 6DoF $\mathbf{g}_{\text{pose}} \in \mathbb{R}^9$ est projeté via un perceptron multicouche `LayerNorm + SiLU` vers $\mathbf{z}_p \in \mathbb{R}^{32}$. La tête de fusion `MLP(160 -> 96 -> 48 -> 2)` prédit directement les angles $(\hat{\psi}, \hat{\theta})$ de la direction de regard 3D dans le repère caméra depuis l'origine oculaire $\mathbf{o}_{3\text{D}}$ :
$$(\hat{\psi}, \hat{\theta}) = H_\phi\big([\mathbf{z}_L \,;\, \mathbf{z}_R \,;\, \mathbf{z}_p]\big)$$
L'intersection rayon–plan écran (`gaze_angles_deg_to_screen_px`) restitue ensuite la position pixel exacte quelle que soit la distance ultérieure à l'écran.

### 5.2.2 Fonction de perte et export temps réel

L'entraînement minimise la perte de Huber lisse (`SmoothL1Loss`, $\beta = 1{,}0^\circ$) avec optimiseur AdamW ($\text{lr} = 2{,}5 \times 10^{-3}$, décroissance cosinus). Cette perte concilie sensibilité quadratique près de zéro ($< 1^\circ$) et robustesse linéaire face aux rares distractions oculaires résiduelles des sessions écologiques. Exporté en ONNX (`opset 17`), le modèle atteint **$0{,}25\text{ ms}$ par frame** sur CPU ($45\times$ plus rapide que `MobileOne-S0`).

## 5.3 Scénario B : `DifferentialEyeNet` et randomisation morphologique

Lorsqu'un modèle direct est entraîné sur un unique sujet $S_0$, deux facteurs provoquent son effondrement sur un utilisateur inconnu $S_k$ :
1. **L'angle kappa inconnu** ($\kappa_k \neq \kappa_0$, écart pouvant atteindre $\pm 5^\circ$–$8^\circ$) ;
2. **La signature anatomique et photométrique** (teinte de peau, largeur d'ouverture palpébrale, couleur d'iris, présence ou non de lunettes).

### 5.3.1 Architecture siamoise différentielle antisymétrique

Inspirée de Liu et al. (BMVC 2018), `DifferentialEyeNet` (`66\,834` paramètres) opère sur des **paires requête–ancre** $(q, r)$ dont les patchs ont préalablement été convertis en luminance et normalisés à moyenne nulle et variance unitaire *par patch* (`preprocess_eye_batch_gray_norm`) :
$$\tilde{I}_e = \frac{Y(I_e) - \mu(Y(I_e))}{\max(\sigma(Y(I_e)), 10^{-4})}$$
Cette normalisation locale élimine dès l'entrée les écarts de carnation cutanée et de niveau d'éclairage global. L'encodeur partagé extrait $(z_{L,q}, z_{R,q})$ et $(z_{L,r}, z_{R,r})$, puis la tête différentielle `diff_head` reçoit uniquement les différences latentes et géométriques (assorties du contexte moyen de pose pour le couplage 3D) :
$$g_\omega(q, r) = \text{MLP}\big([z_{L,q}-z_{L,r} \,;\, z_{R,q}-z_{R,r} \,;\, \mathbf{g}_q - \mathbf{g}_r \,;\, \tfrac{1}{2}(\boldsymbol{\theta}_{\text{tête},q} + \boldsymbol{\theta}_{\text{tête},r})]\big)$$
Pour imposer mathématiquement la cohérence physique d'un opérateur différentiel, la sortie finale est **antisymétrisée par construction** :
$$\widehat{\Delta}(q, r) = \frac{1}{2}\Big(g_\omega(q, r) - g_\omega(r, q)\Big)$$
Cette propriété garantit sans perte auxiliaire que $\widehat{\Delta}(r, r) \equiv (0, 0)$ (erreur nulle sur les ancres exactes) et $\widehat{\Delta}(q, r) = -\widehat{\Delta}(r, q)$.

### 5.3.2 Agrégation multi-ancres $K \in \{1, 5\}$ à coût amorti

À l'utilisation sur un nouvel utilisateur $S_k$, il suffit de capturer $K$ images ancres de calibration $\{r_1, \dots, r_K\}$ (par ex. $K=1$ point central ou la croix 5 points d'Eyeware Beam). Leurs embeddings $(z_{L,r_j}, z_{R,r_j})$ sont **pré-calculés une seule fois** à la fin de la calibration ; chaque frame temps réel ne nécessite donc qu'une unique passe encodeur pour $q$ suivie de $K$ évaluations du petit MLP (`1,94 ms` mesurés pour $K=5$ en PyTorch) :
$$\hat{\mathbf{g}}(q) = \frac{1}{K} \sum_{j=1}^{K} \Big(\mathbf{g}(r_j) + \widehat{\Delta}(q, r_j)\Big)$$
Par construction, la différence $\mathbf{g}(q) - \mathbf{g}(r_j)$ annule exactement l'angle $\kappa_k$ constant du sujet cible !

### 5.3.3 Randomisation morphologique et photométrique mono-sujet

Sans augmentation adaptée, un encodeur entraîné sur le seul visage $S_0$ risque néanmoins de spécialiser ses filtres sur l'ouverture palpébrale ou le contraste exact de $S_0$. Dans `research/augmentations.py`, chaque élément des paires $(q, r)$ subit indépendamment durant l'entraînement :
- Un étirement morphologique anisotrope autour du centre oculaire ($s_x \in [0{,}86, 1{,}14]$, $s_y \in [0{,}75, 1{,}28]$, rotation $\pm 4^\circ$, léger décalage $\pm 1{,}5\text{ px}$), simulant des formes d'yeux et d'ouvertures palpébrales variées ;
- Une perturbation photométrique complète (gamma $\in [0{,}65, 1{,}45]$, gain/biais d'exposition, flou gaussien simulant la défocalisation webcam, bruit capteur) ;
- L'injection probabiliste ($p = 0{,}20$) de reflets spéculaires elliptiques doux simulant des verres de lunettes.

L'ablation comparant `DifferentialEyeNet` *avec* et *sans* cette randomisation morphologique sur des sujets inédits $S_1 \dots S_5$ est quantifiée au Chapitre 6.
