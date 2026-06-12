# Calibration caméra → base robot (eye-to-hand) — document de référence

Date : 2026-06-12 (fusion de la doc initiale, des notes critiques et de l'arbitrage
des conventions OpenCV ; remplace `ur3e_camera_base_calibration_notes.md`).

Ce document décrit la procédure pour estimer `T_base_camera`, la pose de la caméra
événementielle DVXplorer dans le repère **base** du robot UR3e. C'est le maillon
manquant pour convertir une position de balle mesurée dans le repère caméra vers
le repère du robot et lui permettre d'intercepter.

## 1. But et configuration

La caméra est **fixe** dans la pièce (elle observe la balle voler). Pour la
calibrer, on monte un smartphone (Poco X7 Pro) dans un support imprimé en 3D fixé
sur le flasque `tool0` du robot ; l'**écran** du téléphone sert de mire. Le robot
promène la mire à des poses connues, la caméra l'observe à chaque pose.

C'est la configuration **eye-to-hand** (caméra déportée, mire sur le robot).

### Repères

```text
base            repère base du robot (FK / TF, convention UR)
tool0           flasque / outil du robot
mire            centre de l'écran actif ; X vers la droite de l'écran,
                Y vers le bas de l'écran, Z RENTRANT dans l'écran
                (Z sort du DOS du téléphone)
camera_optical  repère optique de la caméra (modèle pinhole OpenCV)
```

**Attention au Z de la mire.** Le script caméra construit les points objet avec
X à droite et Y en bas (Z = 0). Vue de face par la caméra (X image à droite,
Y image en bas), la rotation `solvePnP` est ≈ identité, donc l'axe Z de la mire
est aligné avec le Z caméra : il pointe **de la caméra vers l'écran et au-delà**,
c'est-à-dire qu'il rentre dans l'écran. Définir exactement ce repère dans le
CAD/URDF du support (origine au centre de l'écran actif) et vérifier après
calibration que la normale estimée pointe du bon côté.

### Notation

`T_a_b` = pose du repère `b` exprimée dans `a` = matrice qui transforme les
**coordonnées d'un point** de `b` vers `a` : `p_a = T_a_b · p_b`.

### Transformations en jeu (à chaque pose i)

| Transform          | Source                                                        | Statut          |
|--------------------|--------------------------------------------------------------|-----------------|
| `T_base_tool0(i)`  | FK du robot (TF `base → tool0`, publié par le stack contrôle) | mesurée         |
| `T_camera_mire(i)` | `solvePnP` de la mire dans le flux caméra                     | mesurée         |
| `T_base_camera`    | **inconnue principale** recherchée                            | à résoudre      |
| `T_tool0_mire`     | pose de l'écran par rapport au flasque                        | co-résolue      |

### Équation physique

À chaque pose, le centre de l'écran a une position physique unique, donc :

```text
T_base_camera · T_camera_mire(i) = T_base_tool0(i) · T_tool0_mire
```

Forme `A X = Z B`, résolue par `cv2.calibrateRobotWorldHandEye` — **mais pas en
passant les mesures telles quelles** : voir §4, c'est le point le plus piégeux
de tout le projet.

## 2. Matériel et géométrie de la mire

- Caméra événementielle **DVXplorer** (640×480), fixe.
- Smartphone **Poco X7 Pro**, écran clignotant comme mire (voir §3).
- Support imprimé en 3D maintenant le téléphone sur `tool0`. Masse téléphone +
  support ≪ 3 kg de charge utile UR3e : aucun souci.

### Géométrie écran / téléphone

| Grandeur            | Valeur (mm) |
|---------------------|-------------|
| Écran largeur       | 69,55       |
| Écran hauteur       | 154,50      |
| Corps largeur       | 75,24       |
| Corps hauteur       | 160,75      |
| Bezel gauche        | 2,90        |
| Bezel droite        | 2,75        |
| Bezel haut          | 2,93        |
| Bezel bas           | 3,30        |

`solvePnP` place l'origine de la mire **au centre de l'écran actif**. Le support
CAD, lui, référence le **corps** du téléphone. Décalage corps → centre écran
déduit des bezels :

- Horizontal : centre écran ≈ centre corps **+0,06 mm** (bezels 2,90 vs 2,75).
- Vertical : centre écran ≈ centre corps **−0,20 mm** (bezels 2,93 haut vs 3,30 bas).

En pratique centre écran = centre corps à <0,2 mm près. On l'applique dans le CAD
comme valeur initiale, mais l'erreur dominante vient du jeu d'emboîtement du
téléphone et des tolérances d'impression — **c'est pourquoi `T_tool0_mire` est
co-résolu par hand-eye** plutôt que figé depuis le CAD.

### Contraintes dalle AMOLED (à tester en premier)

Le Poco X7 Pro a une dalle AMOLED dont le **PWM dimming** (~kHz) module toute la
surface en permanence : une caméra événementielle peut être noyée d'événements
qui masquent le clignotement utile des points. Parades :

- **Luminosité à 100 %** (le PWM est souvent désactivé ou très réduit à pleine
  luminosité), ou activer l'option **DC dimming** de HyperOS si disponible.
- Désactiver la luminosité automatique, l'always-on display et les notifications.
- Fixer le taux de rafraîchissement (60 Hz) au lieu de l'adaptatif.
- Mise en veille de l'écran sur « jamais » pendant la session.

C'est le seul point qui peut invalider toute l'approche : **valider la détection
19/19 points avec le téléphone avant d'écrire la moindre ligne de code ROS**
(étape 0 du plan, §10).

## 3. Affichage de la mire sur le téléphone (point critique non couvert avant)

L'outil existant (`event_mire_calibration.py`) affiche la mire en plein écran Qt
sur un **moniteur du PC** (détection xrandr/Qt) et synchronise le clignotement
avec l'accumulation. Un Poco X7 Pro n'est **pas** un moniteur du PC (pas de
DisplayPort alt-mode sur cette gamme). Solution recommandée :

### Page web mire servie par l'outil

1. L'outil de calibration sert une **page HTML** (le PC et le téléphone sont sur
   le même réseau Wi-Fi). Le téléphone l'ouvre en **plein écran réel**
   (Fullscreen API / PWA, aucune barre de navigateur).
2. La page **récupère la disposition des points auprès du serveur** (source de
   vérité unique : même algorithme `build_mire_layout`, mêmes constantes 4×5,
   point manquant (1,2), ancre (0,0), facteur 0,82) et ne fait que dessiner.
3. Le mapping px → mm utilise les dimensions d'écran actives connues
   (69,55 × 154,50 mm) — équivalent des options `--screen-width-mm` /
   `--screen-height-mm` déjà prévues par l'outil. Attention au
   `devicePixelRatio` : raisonner en pixels physiques du viewport plein écran.
4. **Clignotement libre** (la page clignote en continu à ~6 Hz) : à 6 Hz la
   période est ~167 ms, une fenêtre d'accumulation de 240 ms capture donc
   toujours des transitions ON et OFF de tous les points. Aucune
   synchronisation réseau n'est nécessaire pour commencer. Un canal WebSocket
   (noir → clignote sur commande, comme `start_calibration_blink`) reste une
   amélioration possible si le fond événementiel gêne.
5. **Vérification physique obligatoire** : mesurer au pied à coulisse
   l'espacement réel des points affichés et le comparer à la valeur théorique ;
   recouper ensuite avec la métrique `spacing` déjà calculée par l'outil
   (`plane_spacing_metrics`). Si la page n'est pas en vrai plein écran ou si un
   zoom s'applique, c'est ici que ça se voit.

Côté outil PC, il faut un mode `--external-mire` : ne pas ouvrir de `MireWindow`
locale, charger la géométrie de la mire (dimensions écran du téléphone) et
simplement accumuler pendant la fenêtre demandée.

L'alternative « téléphone en second écran réseau » (Deskreen, weylus…) ferait
voir le téléphone comme un moniteur Qt, mais la mise à l'échelle/letterboxing et
la latence la rendent fragile ; à n'envisager qu'en secours.

## 4. Conventions de transformations et solveur (LE point qui fâche)

### Le piège des noms OpenCV

Dans la doc OpenCV, un paramètre nommé `X2Y` désigne la matrice qui transforme
les **coordonnées d'un point** du repère X vers le repère Y, soit `T_y_x` dans
notre notation. Exemple : `R_base2gripper` = `T_gripper_base` =
**l'inverse** du TF ROS `base → tool0`. Passer les mesures telles quelles donne
un résultat numériquement plausible mais physiquement faux : avec le mapping
naïf (world=mire, gripper=tool0), les « constantes » que le solveur cherche
(`T_mire_base`, `T_camera_tool0`) **varient à chaque pose** puisque la mire
bouge avec le robot — le problème résolu n'est pas le bon.

### Recette correcte (vérifiée par dérivation)

Mapping des rôles OpenCV pour notre montage eye-to-hand :

| Rôle OpenCV | Notre repère  |
|-------------|---------------|
| `world`     | base robot    |
| `cam`       | tool0         |
| `base`      | caméra fixe   |
| `gripper`   | mire          |

```python
import cv2
import numpy as np

def invert(R, t):
    Rt = R.T
    return Rt, -Rt @ t

# Entrées mesurées, listes alignées sur les N poses :
#   R_base_tool0[i],  t_base_tool0[i]   : TF base -> tool0 (FK, en mètres)
#   R_camera_mire[i], t_camera_mire[i]  : solvePnP (tvec converti mm -> m !)

R_w2c, t_w2c, R_b2g, t_b2g = [], [], [], []
for i in range(N):
    R, t = invert(R_base_tool0[i], t_base_tool0[i])    # = T_tool0_base(i)
    R_w2c.append(R); t_w2c.append(t)
    R, t = invert(R_camera_mire[i], t_camera_mire[i])  # = T_mire_camera(i)
    R_b2g.append(R); t_b2g.append(t)

R_base_camera, t_base_camera, R_tool0_mire, t_tool0_mire = \
    cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_w2c,    t_world2cam=t_w2c,
        R_base2gripper=R_b2g, t_base2gripper=t_b2g,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

# Sorties DIRECTES (aucune inversion) :
#   (R_base_camera, t_base_camera) = T_base_camera   <- la cible
#   (R_tool0_mire,  t_tool0_mire)  = T_tool0_mire    <- vérification CAD
```

Équation effectivement résolue, constante par construction :

```text
T_tool0_base(i) · T_base_camera = T_tool0_mire · T_mire_camera(i)
        (les deux côtés = T_tool0_camera(i) ; X et Z bien fixes)
```

C'est la même équation que §1 prémultipliée par `T_tool0_base(i)`.

**Résumé : les deux entrées mesurées s'inversent avant l'appel ; les deux
sorties se lisent directement.**

### Contre-vérification avec `calibrateHandEye`

L'astuce eye-to-hand classique : échanger les rôles base/gripper en passant les
poses robot inversées. Les deux solveurs doivent coïncider (~10⁻⁶ en
synthétique, quelques mm/dixièmes de degré en réel) :

```python
R_bc, t_bc = cv2.calibrateHandEye(
    R_gripper2base=R_w2c, t_gripper2base=t_w2c,   # T_tool0_base (FK inversée)
    R_target2cam=R_camera_mire,                    # solvePnP DIRECT (pas inversé)
    t_target2cam=t_camera_mire_m,                  # en mètres
    method=cv2.CALIB_HAND_EYE_PARK)
# rôles échangés => la sortie "cam2gripper" est en réalité T_base_camera
```

### Test synthétique obligatoire avant tout passage sur données réelles

```text
1. Tirer aléatoirement T_base_camera* et T_tool0_mire* (vérité terrain).
2. Générer N ≥ 10 poses T_base_tool0(i) à axes de rotation non parallèles.
3. Calculer T_camera_mire(i) = inv(T_base_camera*) · T_base_tool0(i) · T_tool0_mire*.
4. Lancer le solveur : récupération exacte attendue (puis re-tester avec un
   bruit de 0,1 mm / 0,05° : l'erreur doit rester de l'ordre du bruit).
5. Croiser calibrateRobotWorldHandEye et calibrateHandEye.
```

Ce test (`solve_handeye.py --self-test`) fige les conventions une fois pour
toutes ; ne jamais faire confiance à la doc (y compris celle-ci) sans lui.

### Unités

L'outil caméra travaille en **mm**, ROS en **mètres**. Convertir explicitement
`tvec` (mm → m) avant le solveur. Première source d'erreur grossière.

## 5. Formats de fichiers

Bien distinguer la **géométrie visuelle** (pour afficher) de la **transformation
de repère** (une pose, pas un mesh).

### Géométrie visuelle du support

- **glTF 2.0 (`.glb`)** recommandé : moderne, hiérarchie + matériaux + unités,
  chargé nativement par three.js. Alternative : Collada `.dae` (homogène avec
  les meshes `ur_description`).
- **À l'échelle réelle, en mètres** (ROS/URDF en mètres ; le viewer convertit
  Z-up → Y-up).
- Conserver le **STEP `.step`** issu du CAD comme source dimensionnelle, et le
  **`.stl`** pour l'impression. Conversion STEP → glTF pour le web.
- Ne **jamais** utiliser un STL comme source de vérité d'une transformation :
  pas d'unité, pas de repère, pas de hiérarchie.

### Fichiers support disponibles

Les fichiers ajoutés dans `docs/3D_model/` sont utilisables comme base :

```text
docs/3D_model/Support3D.step   source CAD Onshape, STEP AP242, unités en mètres
docs/3D_model/Support3D.obj    mesh exporté en mètres, utile comme intermédiaire
docs/3D_model/Support3D.mtl    matériau associé à l'OBJ
```

L'OBJ annonce `Units = meters` et son encombrement approximatif est :

```text
X : 244.0 mm
Y : 69.9 mm
Z : 85.0 mm
```

Ces dimensions sont cohérentes avec un support de smartphone monté au poignet du
UR3e. Pour le viewer web, garder le STEP comme source de vérité CAO, mais exporter
une version `Support3D.glb` en mètres à partir du STEP/OBJ. L'OBJ peut rester comme
format intermédiaire lisible ; il référence maintenant correctement
`Support3D.mtl`.

La photo de montage réel est :

```text
docs/Smartphone_Sur_Support.jpeg
```

Elle confirme que le téléphone est bien maintenu face à la caméra, avec la mire
visible sur l'écran. Elle sert de référence visuelle pour le montage et
l'orientation générale, mais **ne remplace pas** la mesure/calibration de
`T_tool0_mire` : la pose exacte du centre d'écran par rapport à `tool0` reste
estimée par hand-eye, puis comparée au CAD.

### Transformations `T_tool0_mire` et `T_base_camera`

- **YAML/JSON** : `xyz` (mètres) + quaternion `xyzw` (et `rpy` pour lecture
  humaine), avec le **repère parent explicite**.
- ou **joint fixe URDF/xacro** : `tool0 → phone_support → screen_center`, et
  `base → camera_optical_frame` pour le résultat — s'intègre directement dans
  TF et le viewer.

### Schéma du JSON multi-échantillons (collecte)

```json
{
  "created_at": "2026-06-12T10:30:00.000",
  "units": "meters",
  "frames": {
    "robot_parent": "base",
    "robot_child": "tool0",
    "camera": "camera_optical",
    "mire": "screen_center"
  },
  "intrinsics_xml": "chemin/vers/calibration.xml",
  "samples": [
    {
      "index": 0,
      "stamp": "2026-06-12T10:30:01.234",
      "T_base_tool0":  {"xyz": [x, y, z], "quat_xyzw": [qx, qy, qz, qw]},
      "T_camera_mire": {"xyz": [x, y, z], "quat_xyzw": [qx, qy, qz, qw],
                         "rvec": [rx, ry, rz], "tvec_mm": [tx, ty, tz]},
      "joint_positions_rad": [j0, j1, j2, j3, j4, j5],
      "stationarity": {"trans_delta_mm": 0.02, "rot_delta_deg": 0.003},
      "reproj_rms_px": 0.42,
      "matched_dots": 19,
      "tilt_deg": 27.5,
      "ippe_ambiguity_ratio": 3.2
    }
  ]
}
```

- `rvec/tvec_mm` bruts : permettent de re-résoudre hors-ligne sans refaire la
  capture.
- `stationarity` : écart TF entre le début et la fin de l'accumulation (§7).
- `ippe_ambiguity_ratio` : rapport d'erreur entre les deux solutions de
  `solvePnPGeneric` (ambiguïté planaire) ; rejeter l'échantillon si proche de 1
  avec un tilt faible.
- `joint_positions_rad` : traçabilité — attention, l'ordre de `/joint_states`
  n'est pas l'ordre physique des joints UR.

## 6. Pré-requis

- **Intrinsèques caméra valides.** Le hand-eye n'est jamais meilleur que les
  intrinsèques. Les intrinsèques actuelles sont approximatives (≈0,486 px sur
  échiquier bruité) — valider avec les outils existants (« Test calib »,
  « Test carré ») : viser erreur de distance < 1 % et erreur d'espacement < 1 %
  avant de lancer le hand-eye.
- **Mire affichée et détectée sur le téléphone** (§3) : 19/19 points, RMS < 1 px.
- **Stack robot lancée** (driver UR + MoveIt) pour que TF `base → tool0` soit
  publié (`ur3e_stack`, voir `docs/ur3e_web_ui.md`).
- **Même `ROS_DOMAIN_ID`** pour l'outil caméra et le stack robot.

## 7. Procédure pas-à-pas

La capture est **statique** (stop-and-go) : l'accumulation événementielle dure
~240 ms et la mire **clignote**, donc le robot doit être **immobile** pendant la
capture.

### Poses de calibration : en espace articulaire, pas en TCP

L'IK (KDL) du TCP Target a des branches multiples et la pose home est proche
d'une singularité de poignet : viser des cibles cartésiennes peut changer de
configuration d'une session à l'autre. Définir les poses de calibration comme
**configurations articulaires enregistrées** : on les crée une fois (jog ou TCP
Target), on les sauvegarde (« enregistrer pose courante » → JSON), puis on les
rejoue à l'identique. C'est le rôle de l'onglet Calibration de l'UI (§10,
étape 4).

### Diversité des poses (critique)

Le hand-eye ne contraint la **rotation** que si les axes de rotation entre poses
ne sont **pas parallèles** :

- 15 à 20 poses retenues ;
- incliner l'écran vers la caméra sous des angles variés (±25–40°), en variant
  les trois axes ;
- couvrir le champ de la caméra et 2–3 distances ;
- à chaque pose : mire **entièrement visible**, **nette**, remplissant bien le
  champ.

### Séquence par pose

1. Aller à la pose articulaire i (UI, gardes de sécurité actives).
2. Laisser le robot se stabiliser.
3. Déclencher l'accumulation (~240 ms) ; la page mire clignote en continu.
4. L'outil lit TF `base → tool0` **au début et à la fin** de la fenêtre
   d'accumulation (listener tf2 intégré, §9) : si l'écart dépasse ~0,1 mm /
   0,02°, l'échantillon est rejeté (robot pas immobile).
5. `solvePnP` → `T_camera_mire` ; contrôle RMS, tilt, ambiguïté IPPE.
6. Enregistrer la paire dans le JSON multi-échantillons (en mètres).

Puis lancer `solve_handeye.py` hors-ligne sur le JSON.

## 8. Validation

Une calibration n'est acceptable que si **tous** ces contrôles concordent :

- **Résidu pixel bout-en-bout** (le critère principal) : pour chaque pose,
  prédire `T̂_camera_mire(i) = inv(T_base_camera) · T_base_tool0(i) · T_tool0_mire`,
  projeter les 19 points et comparer aux blobs mesurés. Viser un RMS global du
  même ordre que le RMS solvePnP par pose (< ~1–2 px).
- **Leave-one-out** : résoudre sur N−1 poses, prédire la pose mire de la Nème ;
  la translation de `T_base_camera` doit rester stable (< ~2–3 mm d'écart).
- **Cohérence CAD** : `T_tool0_mire` estimé vs valeur CAD (< ~5 mm, < ~2–3°).
  Un gros écart signale un problème de montage, d'unités ou d'intrinsèques.
- **Mètre ruban** : position grossière de la caméra dans le repère base vs
  `t_base_camera` (à 2–3 cm près).
- **Normale de l'écran** : l'axe Z estimé de la mire pointe bien dans le sens
  attendu (§1).
- **Viewer 3D** : convertir un point connu caméra → base et vérifier la
  plausibilité dans le viewer.

## 9. Intégration dans le workspace DV-ROWS

### Lecture de la pose robot : tf2, pas de service custom

TF `base → tool0` est déjà diffusé par le driver. L'outil caméra (ou un petit
collecteur dédié) embarque un **nœud rclpy + listener tf2** (spin dans un thread
à côté de Qt) et fait un lookup au moment de la capture — début et fin de la
fenêtre pour le contrôle de stationnarité. Conséquences :

- **pas de package `calibration_interfaces`**, pas de `.srv` ;
- **aucune modification du backend `ur3e_web_ui`** pour la collecte ;
- il suffit que les deux processus partagent le même `ROS_DOMAIN_ID`.

(Un service `GetCalibrationSample` ne redeviendrait utile que si l'UI devait
orchestrer elle-même les captures caméra — hors périmètre actuel.)

### Workspaces : chaîner les overlays plutôt que copier

Plutôt que copier `ur3e_web_ui` dans Dv-Rosws (divergence garantie), chaîner :

```bash
source /opt/ros/humble/setup.bash
source ~/Documents/6-Dof-Ur3e-Catch-a-ball/ros2_ws/install/setup.bash
# puis builder/sourcer l'overlay Dv-Rosws par-dessus si besoin
```

Copier le package reste une option si un workspace 100 % autonome est exigé,
mais c'est à éviter par défaut.

### `base` vs `base_link`

Le stack distingue `base` (convention UR, utilisé par le viewer TCP) et
`base_link` (REP-103, utilisé par MoveIt), tournés de π autour de Z l'un par
rapport à l'autre (voir `docs/ur3e_robot_control_architecture.md`). La
calibration publie explicitement **`base → camera_optical_frame`** et le
mentionne dans tous les fichiers de résultat.

## 10. Plan d'implémentation (ordonné par risque)

| Étape | Contenu | Critère d'acceptation |
|-------|---------|----------------------|
| 0. Validation matérielle | Page web mire sur le téléphone (§3), téléphone tenu à la main devant la caméra | 19/19 associations, RMS < 1 px, distance vs mètre ruban < 2 % |
| 1. Solveur + test synthétique | `solve_handeye.py --self-test` (§4) | récupération exacte sans bruit ; erreur ~bruit avec bruit ; les 2 solveurs OpenCV coïncident |
| 2. Collecte | Mode `--external-mire`, listener tf2, contrôle de stationnarité, export JSON multi-échantillons (§5) | échantillons complets et auto-validés |
| 3. Onglet Calibration UI | Liste de poses articulaires (enregistrer / aller à / suivante), ghost du support `.glb` sur `tool0` | rejouer 15–20 poses sans IK surprise |
| 4. Session réelle | 15–20 poses (§7), solveur, validations (§8) | tous les contrôles §8 au vert |
| 5. Publication | YAML résultat + `static_transform_publisher` `base → camera_optical_frame` + joint fixe URDF pour le viewer | TF visible et cohérent dans le viewer |

## 11. État actuel du code

`/home/rigon/Dv-Rosws/Dv-Rosws/scripts/event_mire_calibration.py` fournit déjà :

- construction de la mire 19 points en mm depuis la taille d'écran active
  (overrides `--screen-width-mm` / `--screen-height-mm`) ;
- détection des blobs événementiels et association 2D/3D robuste ;
- `solvePnP` (`SOLVEPNP_IPPE`, fallback `ITERATIVE`) ;
- exports `mire_observation_*.json` (les matches objet_mm ↔ pixel suffisent à
  recalculer la pose hors-ligne) et `calibration_test_*.json` ;
- validation d'intrinsèques : reprojection, held-out, espacement physique,
  distance vs mesure, séquence « test carré » ;
- un self-test synthétique (`--self-test`).

État d'implémentation du plan §10 (2026-06-12) :

- **Étape 0** : `serve_phone_mire.py` + `phone_mire.html` (Dv-Rosws) — la page
  téléphone dessine le layout servi par `build_mire_layout` (source de vérité
  unique), détection plein-écran, mode mesure pied à coulisse. Reste la
  validation matérielle (19/19, RMS < 1 px, distance < 2 %).
- **Étape 1** : `solve_handeye.py` (Dv-Rosws) — recette §4 vérifiée par
  `--self-test` (récupération exacte, solveurs croisés, garde anti-mapping
  naïf, bruit 0,1 mm/0,05°).
- **Étape 2** : `event_mire_calibration.py --external-mire` — listener tf2,
  stationnarité début/fin, ambiguïté IPPE, export JSON §5 (+ matches pour le
  résidu pixel §8).
- **Étape 3** : onglet Calibration de `ur3e_web_ui` — poses articulaires
  enregistrées/rejouées, ghost du support sur `tool0`
  (`static/models/Support3D.glb`, en mètres).
- **Étape 4** : `run_handeye_session.sh` (Dv-Rosws) lance serveur mire +
  collecteur ; la session physique reste à faire.
- **Étape 5** : `scripts/publish_camera_tf.py` (YAML →
  `static_transform_publisher`, fragment xacro) + `GET /api/calibration/camera`
  + affichage du repère caméra dans le viewer.

## 12. Pièges

- **Conventions OpenCV** : les deux entrées mesurées s'inversent avant
  `calibrateRobotWorldHandEye` (§4). Une recette fausse donne un résultat
  plausible mais faux ; seul le test synthétique protège.
- **PWM AMOLED** : luminosité 100 % / DC dimming, sinon la caméra événementielle
  est noyée (§2).
- **Mire pas en vrai plein écran** sur le téléphone : px→mm faux ; vérifier au
  pied à coulisse (§3).
- **Capture non statique** : mire floue/traînée → pose fausse. Le contrôle de
  stationnarité tf2 (§7) rejette ces échantillons automatiquement.
- **Manque de diversité d'orientation** : rotation sous-contrainte → résultat
  instable.
- **Ambiguïté planaire de `solvePnP`** : à faible inclinaison, deux solutions
  proches ; surveiller `ippe_ambiguity_ratio`, incliner l'écran.
- **Unités** : mm (caméra) vs m (ROS).
- **Intrinsèques médiocres** : le hand-eye ne les rattrape jamais.
- **Poses définies en TCP** : branches IK + singularité de poignet → préférer
  les configurations articulaires enregistrées (§7).
- **`base` vs `base_link`** : repère parent explicite partout (§9).
- **Z de la mire** : il rentre dans l'écran, pas l'inverse (§1).

## Prompt d'implémentation (pour une future session)

```text
Lis :
- docs/ur3e_camera_base_calibration.md   (document de référence, à suivre)
- docs/ur3e_web_ui.md
- docs/ur3e_robot_control_architecture.md
- /home/rigon/Dv-Rosws/Dv-Rosws/scripts/event_mire_calibration.py

Implémente le plan §10 dans l'ordre, une étape par PR :
étape 0 (page web mire téléphone), étape 1 (solve_handeye.py --self-test),
étape 2 (collecte tf2 + JSON multi-échantillons), étape 3 (onglet Calibration),
puis 4 et 5. Respecte strictement les conventions du §4 (les deux entrées
mesurées s'inversent ; sorties directes) et le schéma JSON du §5. Le test
synthétique de l'étape 1 doit passer avant tout code de collecte.
```

## Références

- Outil mire / détection / `solvePnP` :
  `/home/rigon/Dv-Rosws/Dv-Rosws/scripts/event_mire_calibration.py`
- Stack de contrôle robot : `docs/ur3e_robot_control_architecture.md`,
  `docs/ur3e_web_ui.md`
- OpenCV : `cv2.calibrateRobotWorldHandEye`, `cv2.calibrateHandEye`,
  `cv2.solvePnP`, `cv2.solvePnPGeneric`
