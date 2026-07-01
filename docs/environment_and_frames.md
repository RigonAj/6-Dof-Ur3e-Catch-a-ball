# Environnement `Firsttraining` — description complète et repères (frames)

Tâche : **UR3e qui attrape / fait passer une balle à travers un cerceau (hoop)** monté sur son
poignet. Entraînement RL (PPO / skrl) dans Isaac Lab.

Ce document décrit la scène, les repères géométriques, l'espace d'observation/action, la
logique de récompense et de reset, ainsi que les valeurs relevées lors de l'audit sim-to-real.

Source principale :
- [`firsttraining_env.py`](../source/FirstTraining/FirstTraining/tasks/direct/firsttraining/firsttraining_env.py)
- [`firsttraining_env_cfg.py`](../source/FirstTraining/FirstTraining/tasks/direct/firsttraining/firsttraining_env_cfg.py)
- [`ur_gripper.py`](../source/FirstTraining/FirstTraining/tasks/direct/firsttraining/ur_gripper.py)

---

## 1. Vue d'ensemble de la scène

| Élément | Valeur | Source |
|---|---|---|
| Robot | UR3e (6 DoF) + cerceau (hoop) sur `wrist_3_link` | `UR-with-gripper.usd` |
| Objet | Balle rigide, rayon 3 cm, masse 50 g, orange | `ball_cfg` |
| Nombre d'envs (train) | 512 (config) — l'alias `train` en lance 12000 | `scene.num_envs` |
| Espacement des envs | 4.0 m | `env_spacing` |
| `dt` physique | 1/120 s | `sim.dt` |
| `decimation` | 2 → pas de contrôle = 1/60 s (`_step_dt`) | `decimation` |
| Durée d'épisode | 4.0 s (≈ 240 pas de contrôle) | `episode_length_s` |
| Espace d'observation | 33-D | `observation_space` |
| Espace d'action | 6-D continu, borné `[-1, 1]` | `action_space` |

La balle est lancée vers le robot ; le but est de faire **traverser** la balle à travers le
disque du cerceau (détection de passage, `detect_pass_through`).

---

## 2. Repères (frames) — le point clé de l'audit

### 2.1 Résultat du FRAME CHECK

```
body_names: ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
             'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
  base_link world quat (w,x,y,z): [1.0, 0.0, 0.0, 0.0]   ← IDENTITÉ
  base: NOT PRESENT
```

**Verdict :** dans l'observation, le repère « local » d'Isaac est
`position_monde − origine_de_l'env`. Le corps qui est **orienté identité** en monde définit ce
repère. Ici c'est **`base_link`** (`quat = (1,0,0,0)`), et **il n'existe aucun corps `base`**
dans l'articulation (l'USD ne contient que `base_link`).

➡️ **Conséquence sim-to-real : le déploiement ROS doit construire l'observation dans le repère
`base_link`, PAS `base`.**

Sur un vrai UR, `base` est `base_link` **tourné de 180° autour de Z**. Si le code de déploiement
utilise `base`, toutes les positions locales `ball`/`disk` seront **inversées en X et en Y** →
comportement incohérent / « à l'envers » sur le robot réel.

### 2.2 Repère de la balle et du disque (observation)

Toutes les positions dans l'observation sont exprimées en **frame local de l'env** :

```
pos_local = pos_monde − scene.env_origins
```

Comme `base_link` est à l'identité et posé à l'origine de chaque env, ce frame local
correspond au frame `base_link` du robot (translation près). C'est ce frame que le
déploiement réel doit reproduire.

> ⚠️ **Note sur les valeurs `ball_pos_local` / `disk_pos_local = [-12.0, 8.0, 0.0]` du log.**
> Elles sont identiques et « rondes » parce que le FRAME CHECK s'exécute au **tout premier
> `_get_observations`, à l'instant du reset, avant le premier pas de physique**. Les buffers de
> pose physique (`body_pos_w`, `root_pos_w`) sont alors encore à zéro, donc
> `pos_local = 0 − env_origins`, ce qui donne pour l'env 0 la valeur `−origine_env_0 ≈ (-12, 8, 0)`.
> **Ces deux valeurs ne sont donc PAS des lectures réelles de la balle/disque** — seul le
> quaternion `base_link = (1,0,0,0)` est l'information exploitable de cet audit.

### 2.3 Repère du disque (cerceau) dans le corps `wrist_3_link`

Relevé au démarrage (`_read_disk_pose_in_body_frame`) :

```
=== Disk trigger === path=/World/envs/env_0/Robot/ur3e/wrist_3_link/Hoop/node_/Disk
offset = (-0.5, ~0, ~0)          # centre du disque dans le frame wrist_3_link (m)
normal = (~0, ~0, -1.0)          # normale du disque dans le frame wrist_3_link
radius = 0.1000                  # rayon du cerceau (m)
```

Interprétation (les `e-09`/`e-15` sont du bruit numérique → 0) :
- Le centre du cerceau est à **−0,5 m le long de l'axe X** de `wrist_3_link`.
- La **normale du disque** est `(0, 0, -1)` dans le frame `wrist_3_link` (axe de passage de la balle).
- Le **rayon** de détection de passage est **0,10 m**.

Ces valeurs sont figées à l'init et réutilisées à chaque pas via `quat_rotate_wxyz` pour
obtenir la pose monde du disque à partir de la pose de `wrist_3_link`.

---

## 3. Chaîne cinématique et joints

Ordre des joints du bras (identique sim et réel) :

```
['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
```

Corps (bodies) : `base_link → shoulder_link → upper_arm_link → forearm_link →
wrist_1_link → wrist_2_link → wrist_3_link` (le hoop est enfant de `wrist_3_link`).

Pose articulaire initiale (`UR3E_HOOP_CFG.init_state`) :

| Joint | Angle init (rad) |
|---|---|
| shoulder_pan | 0.0 |
| shoulder_lift | −1.57 |
| elbow | 0.0 |
| wrist_1 | −1.57 |
| wrist_2 | 0.0 |
| wrist_3 | 0.0 |

Actionneurs (`ImplicitActuatorCfg`) : `stiffness = 800`, `damping = 40`.
Limites d'effort (N·m) : pan/lift 56, elbow 28, wrist_1/2/3 12.

---

## 4. Espace d'observation (33-D)

Construit dans `_get_observations`, concaténation dans cet ordre :

| # | Champ | Dim | Frame / unité |
|---|---|---|---|
| 1 | `joint_pos` (6 joints) | 6 | rad |
| 2 | `joint_vel` | 6 | rad/s |
| 3 | `disk_pos_local` | 3 | m, frame local (= base_link) |
| 4 | `ball_pos_local` | 3 | m, frame local (= base_link) |
| 5 | `direction` = ball_local − disk_local | 3 | m |
| 6 | `distance` = ‖direction‖ | 1 | m |
| 7 | `ball_vel_w` | 3 | m/s, **frame monde** |
| 8 | `prev_disk_signed_dist > 0` | 1 | booléen (côté du plan du disque) |
| 9 | `actions` (dernière action) | 6 | normalisé `[-1,1]` |
| 10 | `pass_through_count` | 1 | compteur de passages |

**Total = 6+6+3+3+3+1+3+1+6+1 = 33.**

> ⚠️ Point d'attention sim-to-real : le champ `ball_vel_w` est en **frame monde**, pas en frame
> local. Le déploiement réel doit fournir la vitesse de balle dans le même frame monde que la sim
> (ou un frame cohérent avec l'entraînement), sinon décalage.

---

## 5. Espace d'action (6-D) et sémantique

Action = **6 valeurs normalisées `[-1, 1]`**, une par joint. Ce n'est **pas** une position ou une
vitesse directe : c'est un **intégrateur de cible articulaire incrémental** avec sécurités
(`_pre_physics_step`) :

1. `desired_delta_q = clamp(action, -1, 1) · v_safe · dt`
2. `desired_cmd_vel = desired_delta_q / dt`, puis **clamp de sécurité** :
   - butée de vitesse pour ne pas dépasser les limites articulaires (freinage cinématique),
   - clamp d'accélération (`a_safe · dt`) par rapport à la vitesse commande précédente,
   - clamp de vitesse max (`v_safe`).
3. `joint_pos_target += cmd_vel · dt`, puis clamp aux limites de position.
4. La cible est appliquée via `set_joint_position_target` (contrôle en position par l'actionneur).

Limites de sécurité :
- `v_safe` (rad/s) = `(π, π, π, 2π, 2π, 2π)`
- `a_safe` (rad/s²) = `(4π, 4π, 4π, 8π, 8π, 8π)` ≈ `(12.57, 12.57, 12.57, 25.13, 25.13, 25.13)`
- Limites de position (rad) : joints ±2π sauf `elbow` ±π.

> 💡 C'est cette logique d'intégrateur incrémental + sécurités qui doit être **répliquée à
> l'identique côté ROS** pour reproduire le comportement de la politique. Un tremblement du robot
> vient souvent d'un décalage entre cette intégration/sécurité en sim et le contrôleur réel.

---

## 6. Récompense (`compute_rewards`)

```
rew = rew_dist + rew_action + rew_pass + rew_termination
```

| Terme | Formule | Rôle |
|---|---|---|
| `rew_dist` | `exp(-2·d) − d` | rapprocher la balle du centre du disque |
| `rew_action` | `-Σ coeff_j · action_j²` | pénalité d'effort par joint (anti-tremblement) |
| `rew_pass` | `+400` si passage détecté | récompense de succès |
| `rew_termination` | `-100` si terminé (échec) | pénalité de fin prématurée |

Pénalité d'action **par joint**, avec **warm-up** de 150 000 pas
(`smoothstep` de `coeff_start` → `coeff_end`) :

| Joint | coeff début → fin |
|---|---|
| shoulder_pan | 0.85 → 2.55 |
| shoulder_lift | 0.95 → 2.85 |
| elbow | 0.30 → 1.00 |
| wrist_1 | 0.45 → 1.35 |
| wrist_2 | 0.35 → 1.05 |
| wrist_3 | 0.25 → 0.75 |

---

## 7. Reset et cycle d'épisode

- **Reset robot** : non systématique à chaque épisode (`reset_robot_on_episode_reset = False`).
  Pose de départ tirée aléatoirement autour de la pose init (bruit par joint dans `_reset_robot_idx`),
  puis reset aléatoire du robot avec proba 5 % à chaque respawn de balle
  (`random_robot_reset_on_ball_reset_probability = 0.05`).
- **Reset balle** (`_reset_ball_idx`) : position et vitesse tirées dans des plages :

| Grandeur | Plage |
|---|---|
| spawn X | (−0.6, −0.2) m |
| spawn Y | (1.2, 2.1) m |
| spawn Z | (0.5, 1.2) m |
| bruit de position | activé, σ = 0.01 m |
| vitesse X | (−0.7, 0.6) m/s |
| vitesse Y | (−5.0, −3.5) m/s (vers le robot) |
| vitesse Z | (−0.1, 1.5) m/s |

- **Succès** : `reset_on_success = False` (l'épisode continue), la balle est **re-spawnée** après un
  passage (`reset_ball_on_success = True`), maintenue au centre du disque en attendant
  (`ball_respawn_hold_at_disk_center = True`).
- **Terminaison** : contact sur le bras, balle au sol (`z < 0.05`), ou timeout (4 s).

---

## 8. Récapitulatif sim-to-real

1. **Frame observation = `base_link`** (identité). Ne pas utiliser `base` côté ROS (inversion X/Y).
2. **`ball_vel_w` en frame monde** — reproduire le même frame de vitesse.
3. **Disque** : offset `(-0.5, 0, 0)` et normale `(0, 0, -1)` dans `wrist_3_link`, rayon 0.10 m.
4. **Action = intégrateur incrémental** de cible articulaire avec clamps vitesse/accélération/position
   à répliquer exactement.
5. **Ordre des joints** identique à la liste ci-dessus.

---

## 9. Log brut de l'audit (référence)

```
=== Disk trigger === path=/World/envs/env_0/Robot/ur3e/wrist_3_link/Hoop/node_/Disk
offset=(-0.5, -9.269278108958723e-09, -1.7763568394002505e-15)
normal=(8.881784064652351e-15, 3.422854157396879e-08, -0.9999999999999994)
radius=0.10000000149011612
=== Joint names ===
['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
[INFO] Loading model checkpoint from: .../2026-06-30_19-02-25_ppo_torch/checkpoints/best_agent.pt
=== FRAME CHECK (temporary) ===
body_names: ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
  base_link world quat (w,x,y,z): [1.0, 0.0, 0.0, 0.0]
  base: NOT PRESENT
  ball_pos_local (m): [-12.0, 8.0, 0.0]   # instant du reset, avant 1er pas physique -> non significatif
  disk_pos_local (m): [-12.0, 8.0, 0.0]   # idem
  VERDICT: base_link est identité (1,0,0,0) -> le déploiement doit utiliser base_link
=== END FRAME CHECK ===
```
