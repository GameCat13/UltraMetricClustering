import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from distancefunc import *
def visualize_matrix(matrix):
    """
    Визуализирует матрицу в виде тепловой карты.

    Параметры:
    ----------
    matrix : ndarray
        Матрица для визуализации.
    """
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Тепловая карта матрицы")
    plt.show()


def print_ultrametric_matrices(matrices):
    for method, matrix in matrices.items():
        print(f"Ультраметрическая матрица расстояний {method}:")
        print(matrix)

def calculate_and_print_distances(matrix, matrices, distance_func):
    for method, ultrametric_matrix in matrices.items():
        distance = distance_func(matrix, ultrametric_matrix)
        print(f"Расстояние между матрицами ({distance_func.__name__}) {method}: {distance}")

        relative_err = relative_error(matrix, ultrametric_matrix, distance_func.__name__)
        print(f"Относительная ошибка {method}: {relative_err}")


def plot_dendrograms(methods, Z_matrices):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (method, Z) in enumerate(Z_matrices.items()):
        row, col = divmod(i, 3)
        dendrogram(Z, ax=axes[row, col])
        axes[row, col].set_title(f'Дендрограмма {method}')
        axes[row, col].set_xlabel('Объекты')
        axes[row, col].set_ylabel('Расстояние')
    plt.tight_layout()
    plt.show()