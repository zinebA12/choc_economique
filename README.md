# Choc économique — comparaison de méthodes numériques et neuronales pour une équation de diffusion-réaction

## Contexte et objectif

Ce projet étudie la propagation d'un **choc** (une perturbation localisée, par exemple un choc économique) qui se **diffuse et s'atténue au cours du temps**, modélisée par l'équation de diffusion-réaction 1D :

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} - \alpha \, u$$

où `D` est le coefficient de diffusion et `α` le taux d'atténuation (decay) du choc. Cette équation possède une **solution analytique connue**, ce qui permet de l'utiliser comme référence pour évaluer plusieurs méthodes de résolution :

1. **Volumes Finis (VF)** — méthode numérique classique de référence
2. **MLP** — réseau de neurones entraîné uniquement sur des données (apprentissage supervisé)
3. **PINN** (Physics-Informed Neural Network) — réseau de neurones contraint à respecter l'équation physique

L'objectif est de comparer ces approches en termes de **précision**, de **temps de calcul**, et surtout de **capacité de généralisation** à des situations non vues pendant l'entraînement (nouvelles conditions initiales, extrapolation temporelle).

## Structure du dépôt

Les dossiers sont organisés selon l'ordre logique de lecture du projet :

| # | Dossier | Contenu |
|---|---------|---------|
| 1 | [`VF_Solution/`](VF_Solution/) | Solution analytique de référence et résolution par Volumes Finis ; comparaison et quantification de l'erreur numérique |
| 2 | [`VF_Raffinement_Maillages/`](VF_Raffinement_Maillages/) | Étude de convergence : effet du raffinement du maillage (`Nx`) sur les erreurs L1 / L2 / L∞ |
| 3 | [`MLP/`](MLP/) | Génération du jeu de données, comparaison d'architectures, entraînement du MLP |
| 4 | [`PINN/`](PINN/) | Entraînement du PINN (Adam puis fine-tuning L-BFGS), comparaison aux 3 autres méthodes |
| 5 | [`Scenarios/`](Scenarios/) | Étude de sensibilité multi-scénarios, tests hors distribution (OOD : double choc, extrapolation temporelle), tableau final de comparaison |

Fichiers à la racine :
- `fonction.py` — fonctions et classes partagées par tous les notebooks (solution analytique, solveur VF, architectures `MLP`/`PINN`, calcul des erreurs, chargement des modèles entraînés)
- `mlp_reference.pth` / `pinn_reference.pth` — poids des modèles déjà entraînés (permettent de relancer les notebooks sans ré-entraîner)

## Installation et exécution

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

Les notebooks se lancent dans l'ordre indiqué ci-dessus (Jupyter ou VS Code). Les modèles pré-entraînés (`mlp_reference.pth`, `pinn_reference.pth`) permettent de sauter les phases d'entraînement longues et de directement visualiser les résultats via `fonction.charger_mlp()` / `fonction.charger_pinn()`.



Zineb A.
