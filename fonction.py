import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import copy
import time

# ============================================================
# Paramètres par défaut
# ============================================================

A = 0.2
x0 = 0.0
sigma = 1.0

D = 1.0
alpha = 0.1

L = 10
T = 5
Nx = 200
clf = 0.45


# ============================================================
# Solution analytique
# ============================================================

def u_analytique(x, t, A=A, x0=x0, sigma=sigma, D=D, alpha=alpha):

    denom = np.sqrt(1 + 4 * D * t / sigma**2)

    exp_part = np.exp(
        -(x - x0)**2 / (sigma**2 + 4 * D * t)
    )

    return A / denom * exp_part * np.exp(-alpha * t)


# ============================================================
# Méthode des volumes finis
# ============================================================

def u_volume_finie(
    A=A,
    x0=x0,
    sigma=sigma,
    D=D,
    alpha=alpha,
    L=L,
    T=T,
    Nx=Nx,
    clf=clf,
    ic_func=None,
    bc_func=None
):

    dx = 2 * L / Nx

    x_VF = -L + (np.arange(Nx) + 0.5) * dx

    dt = clf * dx**2 / D

    Nt = int(T / dt) + 1

    U_num = np.zeros((Nt, Nx))

    t_VF = np.linspace(0, T, Nt)

    # Condition initiale
    if ic_func is not None:
        U_num[0, :] = ic_func(x_VF)
    else:
        U_num[0, :] = A * np.exp(
            -(x_VF - x0)**2 / sigma**2
        )

    # Coefficient de stabilité
    r = D * dt / dx**2

    for i in range(Nt - 1):

        # Points intérieurs
        for j in range(1, Nx - 1):

            U_num[i+1, j] = (
                U_num[i, j]
                + r * (
                    U_num[i, j+1]
                    - 2 * U_num[i, j]
                    + U_num[i, j-1]
                )
                - alpha * dt * U_num[i, j]
            )

        # Conditions aux limites
        if bc_func is not None:

            U_num[i+1, 0] = bc_func(
                x_VF[0],
                t_VF[i+1]
            )

            U_num[i+1, -1] = bc_func(
                x_VF[-1],
                t_VF[i+1]
            )

        else:

            U_num[i+1, 0] = u_analytique(
                x_VF[0],
                t_VF[i+1],
                A,
                x0,
                sigma,
                D,
                alpha
            )

            U_num[i+1, -1] = u_analytique(
                x_VF[-1],
                t_VF[i+1],
                A,
                x0,
                sigma,
                D,
                alpha
            )

    return x_VF, t_VF, U_num


# ============================================================
# Calcul des erreurs
# ============================================================

def erreurs_L2_Linf(x, U_num, U_exact, dx):

    diff = U_num - U_exact

    L2 = np.sqrt(
        dx * np.sum(diff**2)
    )

    Linf = np.max(
        np.abs(diff)
    )

    return L2, Linf

# --- Configuration centralisée des hyperparamètres PINN "scénario" / OOD ---
# Regroupés ICI, avant toute définition de fonction qui les utilise, pour
# que les valeurs "annoncées" dans CFG soient bien celles réellement
# utilisées partout dans le notebook (harmonisation).
CFG = dict(
    epochs_sc=1000,      # nb d'époques Adam pour un PINN "scénario"/OOD
    lr_sc=1e-3,
    lambda_pde_sc=3.0,
    N_ic_sc=600,          # points de condition initiale
    N_bc_sc=300,          # points de bord (Dirichlet)
    N_r_sc=500,           # points de collocation "physiques" (Adam)
    N_lbfgs_sc=2000,      # points de collocation fixes pour L-BFGS
    lbfgs_iter_sc=200,    # itérations max de L-BFGS
    patience_sc=200,      # patience de l'arrêt anticipé
)

def snapshot_state_dict(model):
    """
    Renvoie une copie RÉELLE et indépendante de l'état d'un modèle
    (tous les tenseurs sont détachés du graphe de calcul et clonés en
    mémoire, cf. .detach().clone()). Utilisée pour mémoriser le
    "meilleur" modèle rencontré pendant l'entraînement (MLP et PINN) :
    contrairement à une simple référence à model.state_dict(), les poids
    sauvegardés ici ne changent plus, même si `model` continue ensuite
    d'être entraîné.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ============================================================
# Architectures partagées (nécessaires pour recharger un state_dict
# depuis n'importe quel notebook)
# ============================================================

class MLP(nn.Module):
    """MLP : 2 entrées (x, t) -> 1 sortie. 3 couches cachées, 32 neurones, tanh."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    """Même architecture que MLP, sortie passée par softplus (u >= 0)."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))


def pinn_residual(model, x, t, D, alpha):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u_pred = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - D * u_xx + alpha * u_pred


def pinn_loss(model, X_data, y_data, X_phys, lambda_pde, alpha, D):
    loss_data = nn.MSELoss()(model(X_data), y_data)
    loss_phys = torch.mean(pinn_residual(model, X_phys[:, 0:1], X_phys[:, 1:2], D, alpha) ** 2)
    return loss_data + lambda_pde * loss_phys, loss_data, loss_phys


def pinn_loss_scenario(model, X_data, y_data, X_phys, D, alpha, lambda_pde=1.0):
    loss_data = nn.MSELoss()(model(X_data), y_data)
    loss_phys = torch.mean(pinn_residual(model, X_phys[:, 0:1], X_phys[:, 1:2], D, alpha) ** 2)
    return loss_data + lambda_pde * loss_phys, loss_data, loss_phys


# ============================================================
# Chargement des modèles entraînés (utilisé par tout notebook qui a
# besoin d'un MLP ou d'un PINN déjà entraîné, sans le réentraîner)
# ============================================================

def charger_mlp(path='../mlp_reference.pth'):
    """Recharge le MLP entraîné + ses stats de normalisation."""
    ckpt = torch.load(path, weights_only=False)
    model = MLP(hidden_dim=ckpt['hidden_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    stats = {k: ckpt[k] for k in ('x_mean', 'x_std', 't_mean', 't_std',
                                   'u_mean', 'u_std', 'eps')}
    # .get(..., np.nan) : compatible avec d'anciens checkpoints sauvegardés
    # avant l'ajout de ce champ (pas de KeyError, juste un "—" à l'affichage).
    stats['t_train_mlp'] = ckpt.get('t_train_mlp', np.nan)
    return model, stats


def predire_mlp(model, stats, x, t):
    """Prédit u(x,t) avec le MLP, en gérant normalisation + dénormalisation."""
    xn = (np.asarray(x) - stats['x_mean']) / stats['x_std']
    tn = (np.asarray(t) - stats['t_mean']) / stats['t_std']
    X = torch.tensor(np.column_stack([xn.ravel(), tn.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        ul = model(X).numpy().flatten() * stats['u_std'] + stats['u_mean']
    return (np.exp(ul) - stats['eps']).reshape(np.asarray(x).shape)


def charger_pinn(path='../pinn_reference.pth'):
    """Recharge le PINN entraîné (Adam + L-BFGS).

    Renvoie désormais (model, meta) : meta contient les temps d'entraînement
    t_adam / t_lbfgs mesurés dans PINN.ipynb, pour pouvoir les réutiliser
    dans un autre notebook (ex. le tableau final de Scenarios.ipynb) sans
    avoir à ré-entraîner le PINN.
    """
    ckpt = torch.load(path, weights_only=False)
    model = PINN(hidden_dim=ckpt['hidden_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    meta = {
        't_adam': ckpt.get('t_adam', np.nan),
        't_lbfgs': ckpt.get('t_lbfgs', np.nan),
    }
    return model, meta