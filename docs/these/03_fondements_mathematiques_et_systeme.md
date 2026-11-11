# Chapitre 3 — Fondements Mathématiques et Architecture du Système `Eyetrack`

Ce chapitre formalise rigoureusement chaque étape géométrique, statistique et algorithmique implémentée dans la plateforme de base `Eyetrack` (`face.py`, `normalize.py`, `gaze_model.py`, `geometry.py`, `calibration.py`, `filters.py`), en précisant les repères de coordonnées et les conventions d'orientation vérifiés par la suite de tests unitaires.

## 3.1 Repères et estimation de pose 6DoF sans repliement d'axes

Soit une image webcam acquise au format $W \times H$ (typiquement $1280 \times 720$ en flux MJPG à $30\text{--}60\text{ Hz}$). MediaPipe FaceLandmarker extrait un ensemble de 478 points caractéristiques $\mathbf{p}_i = (u_i, v_i)^\top \in \mathbb{R}^2$ (dont les contours iriens et centres $\mathbf{p}_{468}, \mathbf{p}_{473}$) ainsi qu'une matrice de transformation rigide $M_{\text{MP}} \in \mathbb{R}^{4\times 4}$.

### 3.1.1 Extraction orthogonale et ordre d'Euler intrinsèque

Pour éliminer toute composante d'échelle résiduelle dans la sous-matrice $M_{3\times 3}$, la rotation pure est obtenue par décomposition en valeurs singulières (`face.py:pose_from_matrix`) :
$$M_{3\times 3} = U \Sigma V^\top \implies R = U V^\top$$
Contrairement à une décomposition $X$-$Y$-$Z$ classique (type `cv2.RQDecomp3x3`), qui induit un couplage parasite (*cross-talk*) entre lacet et tangage dès que la tête pivote (ex. $+40^\circ$ en *yaw* puis $+15^\circ$ en *roll* créant un faux *pitch*), la factorisation dans l'ordre intrinsèque OpenTrack $R = R_y(\psi) R_x(\theta) R_z(\phi)$ garantit l'indépendance des axes :
$$\begin{aligned}
\psi &= \operatorname{atan2}(R_{0,2}, R_{2,2}), \\
\theta &= \arcsin\!\big(\operatorname{clip}(-R_{1,2}, -1, 1)\big), \qquad
\phi = \operatorname{atan2}(R_{1,0}, R_{1,1})
\end{aligned}$$
La translation $\mathbf{t}_{\text{cm}} = (M_{0,3}, M_{1,3}, M_{2,3})^\top$ exprime la position crânienne en centimètres.

### 3.1.2 Détection de clignements et quantification des reflets

L'ouverture oculaire instantanée $o_t$ est définie par la moyenne bilatérale du ratio hauteur palpébrale sur largeur inter-commissurale :
$$o_t = \frac{1}{2} \sum_{e \in \{L, R\}} \frac{\|\mathbf{p}_{\text{top},e} - \mathbf{p}_{\text{bot},e}\|_2}{\max(\|\mathbf{p}_{\text{outer},e} - \mathbf{p}_{\text{inner},e}\|_2, 1)}$$
Un détecteur adaptatif (`BlinkDetector`) maintient une ligne de base asymétrique $b_t$ (remontée rapide avec gain $0{,}10$, adaptation lente à $0{,}02$ lorsque $o_t > 0{,}8\,b_t$) et invalide la mesure du regard dès que $o_t < 0{,}55\,b_t$, avec une fenêtre de maintien post-réouverture de $\Delta t_{\text{hold}} = 0{,}15\text{ s}$.

Parallèlement, pour détecter les reflets spéculaires sur verres correcteurs (`glare_on_iris`), le système mesure la proportion maximale sur les deux disques péri-iriens de rayon $r = \max(3{,}0,\; 0{,}75\,d_{\text{iris}})$ de pixels quasi saturés en luminance ($L \ge 235 / 255$). Les seuils opérationnels distinguent le reflet cornéen physiologique ($< 5\,\%$), le reflet léger ($5\text{--}15\,\%$) et le reflet gênant ($\ge 15\,\%$).

## 3.2 Normalisation projective de Zhang et al. (2018) et échelle métrique

Afin d'immuniser le réseau d'apparence aux variations de distance et de roulis de la tête, `FaceNormalizer` (`normalize.py`) construit une caméra virtuelle canonique :

1. **Modèle sténopé approché de la webcam** : à partir du champ de vision horizontal $\theta_{\text{fov}} = 60^\circ$, la matrice intrinsèque $K$ s'écrit avec $f_x = f_y = \frac{W}{2\tan(\theta_{\text{fov}}/2)}$, $(c_x, c_y) = (W/2, H/2)$.
2. **Profondeur monoculaire par invariant anatomique** : le diamètre irien humain présentant une très faible variance inter-individuelle ($D_{\text{iris}} \approx 11{,}7\text{ mm}$), la distance instantanée est estimée à partir du diamètre mesuré $d_{\text{iris,px}}$ puis filtrée exponentiellement ($\alpha_z = 0{,}15$) :
   $$z_{\text{raw}} = \operatorname{clip}\!\left(\frac{f_x \, D_{\text{iris}}}{d_{\text{iris,px}}}, \; 250\text{ mm}, \; 1500\text{ mm}\right), \qquad \hat{z}_t = \hat{z}_{t-1} + 0{,}15(z_{\text{raw}} - \hat{z}_{t-1})$$
3. **Origine 3D et base orthonormée virtuelle** : le centre facial $\bar{\mathbf{p}}$ (moyenne des 4 coins oculaires $\{33, 133, 362, 263\}$ et 2 commissures labiales $\{61, 291\}$) est rétro-projeté dans le repère caméra OpenCV ($x$ droite, $y$ bas, $z > 0$ devant la caméra) :
   $$\mathbf{o} = \hat{z}_t \, \frac{K^{-1}\tilde{\mathbf{p}}}{\|K^{-1}\tilde{\mathbf{p}}\|_2}, \qquad \mathbf{z}_v = \frac{\mathbf{o}}{\|\mathbf{o}\|_2}$$
   L'axe horizontal $\mathbf{x}_{\text{head}}$ est aligné sur les coins externes $\mathbf{p}_{263}, \mathbf{p}_{33}$, puis orthogonalisé :
   $$\mathbf{y}_v = \frac{\mathbf{z}_v \times \mathbf{x}_{\text{head}}}{\|\mathbf{z}_v \times \mathbf{x}_{\text{head}}\|_2}, \qquad \mathbf{x}_v = \mathbf{y}_v \times \mathbf{z}_v, \qquad R_{\text{norm}} = \begin{bmatrix} \mathbf{x}_v^\top \\ \mathbf{y}_v^\top \\ \mathbf{z}_v^\top \end{bmatrix} \in SO(3)$$
4. **Homographie de redressement** : avec distance virtuelle $d_v = 600\text{ mm}$, matrice d'échelle $S = \operatorname{diag}(1, 1, d_v / \|\mathbf{o}\|_2)$ et matrice intrinsèque virtuelle $C_v$ de focale $f_v = 1900 \cdot (S_{\text{patch}}/448)$, l'image redressée est obtenue par transformation projective :
   $$M_{\text{warp}} = C_v \, S \, R_{\text{norm}} \, K^{-1}$$

> **Constat empirique d'échelle focale :** À la focale canonique de Zhang ($f_v = 960\text{ px}$ pour un patch $448$), le visage n'occupe qu'environ la moitié du cadre et `MobileOne-S0` ne suit le regard qu'à $\sim 30\,\%$ d'amplitude (`normalize.py`). L'élévation à $f_v = 1900\text{ px}$ multiplie le rapport signal/bruit par $\sim 4$ en ajustant le cadrage à la distribution d'entraînement de Gaze360.

## 3.3 Estimation du regard et intersection rayon–plan écran

### 3.3.1 Décodage L2CS et dé-normalisation 3D

`NetGazeEstimator` (`gaze_model.py`) infère sur le patch $448 \times 448$ normalisé ImageNet deux vecteurs de logits $\boldsymbol{\ell}_\psi, \boldsymbol{\ell}_\theta \in \mathbb{R}^{90}$ (classes de largeur $\Delta = 4^\circ$, décalage $-180^\circ$) décodés par espérance softmax :
$$\hat{\alpha} = 4{,}0 \sum_{b=0}^{89} b \, \frac{e^{\ell_b}}{\sum_{j=0}^{89} e^{\ell_j}} - 180^\circ, \qquad \alpha \in \{\psi, \theta\}$$
Le vecteur unitaire normalisé $\mathbf{g}_{\text{norm}} = (-\cos\theta\sin\psi,\; -\sin\theta,\; -\cos\theta\cos\psi)^\top$ est ensuite replacé dans le repère caméra réel par rotation inverse (Zhang et al., 2018, sans appliquer $S$) :
$$\mathbf{g}_{\text{cam}} = \frac{R_{\text{norm}}^\top \mathbf{g}_{\text{norm}}}{\|R_{\text{norm}}^\top \mathbf{g}_{\text{norm}}\|_2}$$

### 3.3.2 Projection sur le plan écran et métrique angulaire exacte

Le plan écran étant modélisé à $z = 0$ dans le repère caméra (bord supérieur centré horizontalement à $x=0$, décalage vertical webcam $y_{\text{cam}}$), le point d'impact $(h_x, h_y)$ en millimètres s'obtient analytiquement par intersection rayon–plan :
$$t = -\frac{o_z}{\min(g_{\text{cam},z}, -0{,}05)}, \qquad \begin{pmatrix} h_x \\ h_y \end{pmatrix} = \begin{pmatrix} o_x + t\,g_{\text{cam},x} \\ o_y + t\,g_{\text{cam},y} \end{pmatrix}$$
Cette formulation absorbe géométriquement les translations 3D de la tête ($1\text{ cm}$ de décalage facial $\implies 1\text{ cm}$ à l'écran) sans que la calibration statistique n'ait à les réapprendre.

Pour toute cible $\mathbf{p}_{\text{tgt}}$ et prédiction $\mathbf{p}_{\text{pred}}$ converties en coordonnées métriques 3D sur le plan écran, l'erreur angulaire visuelle exacte $\varepsilon_{\text{vis}}$ est calculée depuis l'origine oculaire $\mathbf{o}$ (`geometry.py:visual_angle_deg`) :
$$\varepsilon_{\text{vis}} = \arccos\!\left(\operatorname{clip}\!\left(\frac{(\mathbf{p}_{\text{tgt}} - \mathbf{o}) \cdot (\mathbf{p}_{\text{pred}} - \mathbf{o})}{\|\mathbf{p}_{\text{tgt}} - \mathbf{o}\|_2 \, \|\mathbf{p}_{\text{pred}} - \mathbf{o}\|_2}, \; -1, \; 1\right)\right) \cdot \frac{180}{\pi}$$

## 3.4 Calibration affine régularisée Ridge et mise à jour séquentielle RLS

Sur les vecteurs de caractéristiques brutes $\mathbf{x}_i \in \mathbb{R}^d$ collectés durant les 5 ou 9 points de calibration ($d = 4$ en mode `net` : $[h_x, h_y, r_x, r_y]$ ; $d = 7$ en mode `iris` : $[r_x, r_y, r_x^2, r_x r_y, r_y^2, \psi_{\text{tête}}, \theta_{\text{tête}}]$), `AffineGazeModel` (`calibration.py`) applique d'abord une standardisation planchéisée pour éviter l'explosion de variance des variables peu excitées :
$$z_{i,j} = \frac{x_{i,j} - \mu_j}{\max(\sigma_j, \sigma_{j,\min})}, \qquad \boldsymbol{\phi}_i = \begin{bmatrix} \mathbf{z}_i \\ 1 \end{bmatrix} \in \mathbb{R}^{d+1}$$
Les poids $W \in \mathbb{R}^{(d+1)\times 2}$ minimisent la perte quadratique avec pénalité Ridge non appliquée au terme de biais :
$$\hat{W} = \left(\Phi^\top \Phi + \alpha N \, \operatorname{diag}(1, \dots, 1, 0)\right)^{-1} \Phi^\top Y$$

Lors de l'adaptation implicite sur clics souris, la matrice de covariance inverse initiale est volontairement dilatée d'un facteur $N / N_{\text{equiv}}$ ($N_{\text{equiv}} = 15$ échantillons équivalents) :
$$P_0 = \frac{N}{N_{\text{equiv}}} \left(\Phi^\top \Phi + \alpha N \, \tilde{I}\right)^{-1}$$
Chaque clic valide (curseur visible, erreur résiduelle $\|y_{\text{clic}} - \boldsymbol{\phi}_t^\top W\|_2 \le 220\text{ px}$) met à jour $W$ et $P$ via les équations RLS à facteur d'oubli $\lambda = 0{,}998$ :
$$\begin{aligned}
\mathbf{k}_t &= \frac{P_{t-1} \boldsymbol{\phi}_t}{\lambda + \boldsymbol{\phi}_t^\top P_{t-1} \boldsymbol{\phi}_t}, \\
W_t &= W_{t-1} + \mathbf{k}_t \big(y_{\text{clic}}^\top - \boldsymbol{\phi}_t^\top W_{t-1}\big), \qquad
P_t = \frac{1}{\lambda}\left(P_{t-1} - \mathbf{k}_t \boldsymbol{\phi}_t^\top P_{t-1}\right)
\end{aligned}$$

## 3.5 Filtrage temporel médian + One Euro

Avant l'envoi à $120\text{ Hz}$ vers OpenTrack, les séries temporelles de pose et de regard passent par un pré-filtre médian sur fenêtre $w=3$ suivi d'un filtre One Euro (Casiez et al., 2012) :
$$\begin{aligned}
\alpha(f_c, \Delta t) &= \left(1 + \frac{1}{2\pi f_c \Delta t}\right)^{-1}, \qquad
f_c(t) = f_{c,\min} + \beta \, |\hat{\dot{x}}_t|, \\
\hat{x}_t &= \alpha(f_c, \Delta t)\, x_t + \big(1 - \alpha(f_c, \Delta t)\big)\,\hat{x}_{t-1}
\end{aligned}$$
où $\hat{\dot{x}}_t$ est la dérivée discrète lissée à coupure fixe $f_{d} = 1{,}0\text{ Hz}$. L'impact quantitatif de ce couplage médian + One Euro sur des captures réelles est mesuré au Chapitre 6.
