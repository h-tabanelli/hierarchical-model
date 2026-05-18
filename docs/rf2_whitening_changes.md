# RF 2-Layer Model — Modifications minimales

## Résumé

Ajout d'un second layer RF avec suppression affine des ordres 0 et 1, activé via `head_mode=latent_rf_spectral`.

---

## 1. `estimators.py`

### 1a. `_apply_rf_activation` — nouvelles variantes de ReLU

**Avant :** une seule entrée `"relu"` qui faisait `relu(u) - 1/√(2π) - 0.5·u` (soustraction analytique ordres 0 et 1).

**Après :** quatre familles :

| Nom(s)                                        | Formule                              | Rôle                            |
|----------------------------------------------|--------------------------------------|---------------------------------|
| `"id"`, `"identity"`, `"linear"`             | `u`                                  | identité                        |
| `"relu_raw"`, `"relu_uncentered"`, `"relu_plain"` | `relu(u)`                       | ReLU brut, garde ordres 0 et 1  |
| `"relu_mean0"`, `"relu_center0"`, `"relu_zero_mean"` | `relu(u) - 1/√(2π)`         | retire ordre 0 seulement        |
| `"relu_l1"`, `"relu_center01"`               | `relu(u) - 1/√(2π) - 0.5·u`         | ancienne `"relu"`, ordres 0+1 analytiques (Gaussien) |

Le layer 1 continue d'utiliser `relu_l1` / `relu_center01`.  
Le layer 2 utilise `relu_raw` (le retrait affine est fait par régression, cf. §1c).

### 1b. Fonctions supprimées

- `estimate_Bhat_from_H_stream_nonisotropic` — retrait non-isotrope de Bhat (remplacé par le pipeline vecteur-affine).

### 1c. Nouvelles fonctions principales

#### `fit_rf_vector_affine_removed_head_from_H_stream`

**C'est la fonction centrale du second layer.**

**Principe :** retrait affine *vectoriel* dans l'espace RF2.

Étant donné `H : (bs, d_in)` :
```
U = H W^T           (bs, m)
S = sigma(U)        (bs, m)     # sigma = relu_raw
```

**Passe 1** — régression affine `S ≈ a + U B^T` par moindres carrés :
```
B = C_su · C_uu^{-1}    (m,m)
a = E[S] - B E[U]       (m,)
```
où `C_uu = Cov(U,U)` et `C_su = Cov(S,U)` calculées empiriquement sur le stream d'entraînement.  
Ridge `ε·I` ajouté sur `C_uu` pour régulariser.

**Features corrigées :**
```
R = S - a - U B^T
```

**Passe 2** — estimateur linéaire :
```
ahat = (1/n) Σ_μ y_μ R_μ
```

Retourne `(rf_layer, a_aff, B_aff, ahat)`.

#### `compute_rf_vector_affine_removed_features_from_H`

Applique la transformation `R = S - a - U B^T` à un batch `H`.

#### `compute_h2hat_from_H_and_rf_vector_affine_removed_linear_head`

Calcule le scalaire `h2hat = R @ ahat`.

#### `fit_rf_empirical_order01_linear_head_from_H_stream` *(approche intermédiaire, gardée)*

Retrait empirique *par dimension* :
```
a_j = E[relu(u_j)]
b_j = E[relu(u_j)·u_j] / E[u_j^2]
r_j = relu(u_j) - a_j - b_j·u_j
```
Non utilisée dans le pipeline actif (remplacée par la version vectorielle).

---

## 2. `cluster_tools/make_tasks_2layers.py`

Nouveaux arguments CLI et champs JSON de tâche :

```python
--head_mode   # ajout de "latent_rf_spectral" dans les choices
--rf2_width   # int, défaut 4096
--rf2_activation  # str, défaut "relu_raw"
--rf2_affine_ridge  # float, défaut 1e-6
--rf2_use_whiten / --no_rf2_use_whiten  # défaut True (non utilisé dans la branche active)
--calibrate_output / --no_calibrate_output  # défaut False
```

Ces champs sont sérialisés dans le JSON de tâche.

---

## 3. `cluster_tools/run_task_2layers.py`

### Suppressions

- `_raw_pred_from_Hhat_nonisotropic` : supprimée.
- Variables `hhat_mean`, `hhat_cov` : supprimées.

### Ajouts dans `_run_one_alpha_model`

**Nouveaux paramètres :** `rf2_width`, `rf2_activation`, `rf2_affine_ridge`, `calibrate_output`.

**`_make_Hhat_stream_factory(use_whiten_for_rf2)`** : helper interne qui construit un générateur de `(H, y)` — soit en passant par le whitening (si `use_whiten_for_rf2=True`), soit en donnant `H` brut.

**Branch `head_mode == "latent_rf_spectral"` :**

```python
# Pas de whitening avant RF2 dans cette version
Hhat_stream_factory = _make_Hhat_stream_factory(use_whiten_for_rf2=False)

rf2_layer, rf2_affine_a, rf2_affine_B, ahat_rf2 = (
    estimators.fit_rf_vector_affine_removed_head_from_H_stream(
        stream_fn_factory=Hhat_stream_factory,
        d_in=p, rf_width=rf2_width,
        n_total=n, rf_activation=rf2_activation,
        rf_seed=seed + 424242, device=device, ridge=1e-6,
    )
)

# Collecte H2_train = h2hat scalaire pour chaque sample
# puis KRR RBF sur H2_train, y2_train
```

### `calibrate_output`

Le flag rend la calibration affine de sortie optionnelle (elle était systématique avant).

---

## Réponse à la question sur l'ordre de la 2e layer

**Pour le premier layer** : les ordres 0 et 1 sont retirés **analytiquement** sous hypothèse gaussienne de la pré-activation (coefficients d'Hermite exacts : `- 1/√(2π)` et `- 0.5·u`).

**Pour le second layer** : la pré-activation `U = H W^T` n'est plus gaussienne (c'est une combinaison linéaire de features RF, donc non-standard). L'approche retenue est donc **empirique** — mais sous forme **vectorielle** (et non scalaire par dimension) :

- On estime `a ∈ R^m` et `B ∈ R^{m×m}` par régression sur le stream d'entraînement.
- `B` capture les corrélations croisées entre dimensions de `U`, ce qu'une approche scalaire `b_j = Cov(s_j, u_j)/Var(u_j)` ne ferait pas.
- En pratique `B` est proche d'une matrice diagonale si les pré-activations sont peu corrélées, mais le retrait est exact dans le cas général.

Une version intermédiaire `fit_rf_empirical_order01_linear_head_from_H_stream` avait été codée (retrait scalaire par dimension, identique à l'approche Gaussienne mais avec coefficients estimés). Elle est conservée dans le code mais n'est pas utilisée dans le pipeline actif.
