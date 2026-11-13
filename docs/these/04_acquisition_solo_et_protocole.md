# Chapitre 4 — Méthodologie d'Acquisition Solo et Protocoles Expérimentaux

Un des verrous centraux d'un projet individuel d'eye tracking réside dans l'acquisition des données : comment constituer un corpus massif ($10^3$ à $10^4$ images annotées au dixième de degré près), diversifié en poses de tête et en conditions photométriques, sans matériel Tobii et avec **un seul participant** ? Ce chapitre formalise les protocoles implémentés dans `research/eye_crop.py`, `research/collect_pursuit.py`, `research/collect_clicks.py` et `research/lag_estimation.py`.

## 4.1 Extraction canonique des patchs binoculaires ($96 \times 48$)

L'analyse de la note méthodologique et de l'état de l'art montre qu'un réseau plein-visage ($448 \times 448$) souffre de deux défauts majeurs en contexte mono-sujet :
1. **Surcharge de calcul CPU** ($11{,}16\text{ ms}$ par frame pour `MobileOne-S0`) ;
2. **Risque de sur-apprentissage du contexte** (arrière-plan de la pièce, vêtements, coiffure), créant une illusion de précision intra-session qui s'effondre dès le moindre changement d'éclairage ou de posture.

Dans `research/eye_crop.py`, chaque œil est extrait indépendamment par transformation affine alignée sur ses commissures $(\mathbf{p}_{\text{gauche}}, \mathbf{p}_{\text{droite}})$ (indices MediaPipe $\{33, 133\}$ pour l'œil droit du sujet et $\{362, 263\}$ pour l'œil gauche) :
$$\mathbf{v}_e = \mathbf{p}_{\text{droite},e} - \mathbf{p}_{\text{gauche},e}, \qquad w_e = \|\mathbf{v}_e\|_2, \qquad \phi_e = \operatorname{atan2}(v_{e,y}, v_{e,x})$$
Le centre du patch $\mathbf{c}_e$ combine équitablement le milieu inter-commissural et le milieu vertical palpébral $(\mathbf{p}_{\text{top},e} + \mathbf{p}_{\text{bot},e})/2$, et le facteur d'échelle $s_e = W_{\text{crop}} / (1{,}6 \, w_e)$ garantit une largeur fixe $W_{\text{crop}} = 96\text{ px}$ ($H_{\text{crop}} = 48\text{ px}$), incluant systématiquement les paupières supérieure et inférieure tout en éliminant le roulis crânien et l'arrière-plan.

L'entrée du réseau est complétée par un vecteur géométrique normalisé $\mathbf{g}_{\text{pose}} \in \mathbb{R}^9$ :
$$\mathbf{g}_{\text{pose}} = \left[\frac{x}{100}, \frac{y}{100}, \frac{z}{100}, \psi_{\text{rad}}, \theta_{\text{rad}}, \phi_{\text{rad}}, \frac{\bar{u}_{\text{yeux}}}{W} - 0{,}5, \frac{\bar{v}_{\text{yeux}}}{H} - 0{,}5, \frac{d_{\text{iris}}}{W}\right]^\top$$
Ce découplage réduit la charge mémoire à $2 \times 48 \times 96 \times 3 = 27{,}6\text{ Ko}$ bruts ($\sim 10\text{ Ko}$ compressés par frame) et le temps d'extraction sur CPU à $0{,}12\text{ ms}$.

## 4.2 Protocole 1 : Poursuite oculaire lisse (*Smooth Pursuit*) et estimation du lag $\tau$

### 4.2.1 Justification physiologique et trajectoire de Lissajous saturée

Les calibrations classiques par points fixes (5 à 9 cibles) imposent une contrainte cognitive forte tout en ne couvrant qu'une poignée de directions du regard. En revanche, lorsque le sujet suit visuellement une cible continue se déplaçant à vitesse modérée ($< 15^\circ/\text{s}$), le système oculomoteur humain engage la **poursuite lisse** (*smooth pursuit*) avec un gain proche de l'unité ($\approx 0{,}95$) et une erreur de suivi intrinsèque inférieure à $0{,}5^\circ$, tout en autorisant des mouvements lents de tête (réflexe vestibulo-oculaire compensé naturellement par le sujet).

Dans `research/lag_estimation.py`, la trajectoire écran $(x(t), y(t))$ est engendrée par deux paires d'harmoniques incommensurables ($f_{x1}=0{,}11\text{ Hz}$, $f_{x2}=0{,}047\text{ Hz}$, $f_{y1}=0{,}13\text{ Hz}$, $f_{y2}=0{,}061\text{ Hz}$) passées dans une compression douce $\tanh(1{,}25\,s)/\tanh(1{,}25)$. Ce profil maintient une continuité $C^\infty$, passe fréquemment dans les bords et coins (à $6\,\%$ des marges écran), et borne la vitesse angulaire maximale sous les limites de saccades de rattrapage (vérifiée analytiquement à $\le 25^\circ/\text{s}$ en pic et modérée en moyenne). À $30\text{ fps}$, une session de $60\text{ s}$ produit $1\,800$ échantillons étiquetés.

### 4.2.2 Modélisation et identification du retard visuo-moteur $\tau_{\text{lag}}$

En poursuite lisse, l'œil accuse un retard de traitement visuo-moteur et d'acquisition caméra quasi constant $\tau_{\text{lag}} \approx 50\text{--}120\text{ ms}$. Étiqueter l'image $I(t)$ par la cible synchrone $\mathbf{y}(t)$ introduit une erreur systématique proportionnelle à la vitesse de la cible :
$$\varepsilon_{\text{lag}} \approx \|\dot{\mathbf{y}}(t)\| \cdot \tau_{\text{lag}}$$
À $12^\circ/\text{s}$ et $\tau = 80\text{ ms}$, ce biais atteint déjà $\sim 1{,}0^\circ$ — suffisant à lui seul pour empêcher de descendre sous $1{,}5^\circ$ !

Pour identifier $\hat{\tau}_{\text{lag}}$ empiriquement sans fuite sur la cible, `estimate_visuomotor_lag` balaie $\tau \in [-40, +220]\text{ ms}$ par pas de $5\text{ ms}$, interpole les cibles décalées $\mathbf{y}(t - \tau)$ sur un cœur temporel commun et évalue la RMSE d'une régression Ridge en validation croisée temporelle en deux blocs disjoints (première moitié / seconde moitié de la session) :
$$\hat{\tau}_{\text{lag}} = \arg\min_{\tau} \; \text{RMSE}_{\text{CV2}}\!\big(X(t), \; \mathbf{y}(t - \tau)\big)$$

## 4.3 Protocole 2 : Récolte passive pré-saccadique sur clics souris

Pour accumuler des données en conditions écologiques réelles (bureautique, navigation, variations naturelles de posture et de luminosité), `research/collect_clicks.py` implémente un collecteur passif silencieux doté d'un **tampon circulaire** (*ring buffer*) de $90$ trames ($\sim 1{,}5\text{--}3\text{ s}$).

L'innovation méthodologique par rapport à l'échantillonnage synchrone au clic (`main.py`) tient à la chronométrie oculo-motrice : au moment exact $t_{\text{clic}}$ où le doigt enfonce le bouton gauche de la souris, l'œil a fréquemment déjà amorcé une saccade vers la zone d'action suivante ou cligné. Le collecteur inspecte donc le tampon circulaire autour de l'instant pré-saccadique optimal $t_{\text{clic}} - \tau_{\text{pre}}$ (défaut $\tau_{\text{pre}} = 75\text{ ms}$, fenêtre d'acceptation $\pm 60\text{ ms}$) et ne valide l'échantillon que sous quatre conditions cumulatives :
1. Le curseur Windows est explicitement visible (`CURSOR_SHOWING`, excluant les jeux FPS/simulation où la souris est capturée) ;
2. Le visage est détecté et les deux patchs oculaires ont une largeur $\ge 6\text{ px}$ ;
3. Aucun clignement ni phase post-réouverture ($0{,}15\text{ s}$) n'est actif ;
4. Le taux de reflet irien reste inférieur au seuil `GLARE_STRONG` ($0{,}15$).

## 4.4 Règle stricte de partitionnement inter-sessions (anti-fuite temporelle)

Une erreur méthodologique fréquente dans la littérature amateure consiste à diviser aléatoirement les trames d'une même vidéo (*random frame split*) entre entraînement et test. À $30\text{ fps}$, deux images distantes de $33\text{ ms}$ sont quasi identiques : l'erreur de test mesure alors la mémorisation locale, pas la généralisation.

Dans l'ensemble de ce travail (`research/train.py` et `research/run_synthetic_ablations.py`), nous imposons une **séparation stricte par sessions indépendantes** :
- Les sessions d'entraînement, de validation et de test sont capturées lors d'exécutions distinctes (ou, à défaut sur une session unique, découpées en blocs temporels contigus 80/20 sans chevauchement).
- L'évaluation du **Scénario A** teste la robustesse inter-sessions sur un même sujet (changement d'éclairage, de distance, de posture).
- L'évaluation du **Scénario B** entraîne exclusivement sur un sujet $S_0$ et évalue la performance sur des sujets inédits $S_1, \dots, S_5$ jamais vus à l'entraînement.
