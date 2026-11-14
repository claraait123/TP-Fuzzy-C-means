import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fuzzy_c_means(X, K=2, m=2.0, eps=1e-4, max_iter=100, seed=0):
    """
    X        : (N, d) données
    K        : nombre de clusters
    m        : coefficient de flou (> 1)
    eps      : seuil d'arrêt sur ||U(k+1) - U(k)||
    Retourne : U (N,K), centres C (K,d), historique de J_m
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    U = rng.random((N, K))
    U /= U.sum(axis=1, keepdims=True)

    history = []
    for k in range(max_iter):
        # centres : c_j = sum(u_ij^m x_i) / sum(u_ij^m)
        Um = U ** m
        C = (Um.T @ X) / Um.sum(axis=0)[:, None]

        # distances ||x_i - c_j||
        D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        D = np.fmax(D, 1e-10)

        # fonction objectif J_m
        history.append(np.sum(Um * D**2))

        # maj u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1))
        inv = D ** (-2.0 / (m - 1))
        U_new = inv / inv.sum(axis=1, keepdims=True)

        if np.linalg.norm(U_new - U) < eps:
            U = U_new
            break
        U = U_new

    return U, C, history

def segmenter_image(path, K=2, m=2.0, eps=1e-4):
    """
    path     : chemin de l'image (convertie en niveaux de gris, valeurs dans [0,1])
    K        : nombre de clusters (classes floues)
    m        : coefficient de flou (> 1)
    eps      : seuil d'arrêt sur ||U(k+1) - U(k)||
    Retourne : U (N,K) degrés d'appartenance, centres C (K,1) triés par ordre croissant,
               dure (h,w) segmentation dure par argmax
    Affiche  : image originale, une heatmap d'appartenance par classe, segmentation dure
    Sauvegarde : fcm_result.png
    """

    img = np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    h, w = img.shape
    X = img.reshape(-1, 1)

    U, C, J = fuzzy_c_means(X, K=K, m=m, eps=eps)
    print(f"{len(J)} itérations, centres = {C.ravel()}")

    ordre = np.argsort(C.ravel())
    U, C = U[:, ordre], C[ordre]

    cartes = [U[:, j].reshape(h, w) for j in range(K)]
    dure = U.argmax(axis=1).reshape(h, w)

    # Affichage
    fig, ax = plt.subplots(1, K + 2, figsize=(4 * (K + 2), 4))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Image originale")
    for j in range(K):
        im = ax[j + 1].imshow(cartes[j], cmap="jet", vmin=0, vmax=1)
        ax[j + 1].set_title(f"Appartenance classe {j} (c={C[j,0]:.2f})")
        plt.colorbar(im, ax=ax[j + 1], fraction=0.046)
    ax[-1].imshow(dure, cmap="gray")
    ax[-1].set_title("Segmentation dure (argmax)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("fcm_resultat.png", dpi=150)
    plt.show()
    return U, C, dure

    
X = np.concatenate([np.random.normal(0.2, 0.05, 50), np.random.normal(0.8, 0.05, 50)]).reshape(-1, 1)
U, C, J = fuzzy_c_means(X, K=2)
print("Centres :", C.ravel())
print("Itérations :", len(J), "| J final :", J[-1])
print("Somme des lignes de U :", U.sum(axis=1)[:3])

segmenter_image("img/milky-way-nvg.jpg", K=2, m=2.0, eps=1e-4)
