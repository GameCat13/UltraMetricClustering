import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from distancefunc import *
from tabulate import tabulate
def visualize_matrix(matrix, text=''):
    """
    Визуализирует матрицу в виде тепловой карты.

    Параметры:
    ----------
    matrix : ndarray
        Матрица для визуализации.
    """
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Тепловая карта матрицы " + text)
    plt.show()


def print_ultrametric_matrices(matrices):
    for method, matrix in matrices.items():
        print(f"Ультраметрическая матрица расстояний {method}:")
        print(matrix)

def calculate_and_print_distances(matrix, matrices, distance_func):
    for method, ultrametric_matrix in matrices.items():
        distance = matrix_distance(matrix, ultrametric_matrix, distance_func)
        print(f"Расстояние между матрицами ({distance_func}) {method}: {distance}")

        relative_err = relative_error(matrix, ultrametric_matrix, distance_func)
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


def calculate_and_print_all_distances(matrix, matrices):
    """
    Вычисляет расстояния между матрицами для всех норм и выводит результаты в виде таблицы.
    :param matrix: Исходная матрица (numpy array).
    :param matrices: Словарь ультраметрических матриц (dict).
    """
    # Создаём список для хранения результатов
    results = []

    # Вычисляем расстояния для каждой ультраметрической матрицы и каждой нормы
    for method, ultrametric_matrix in matrices.items():
        l1_distance = matrix_distance(matrix, ultrametric_matrix, 'L1')
        l2_distance = matrix_distance(matrix, ultrametric_matrix, 'L2')
        linf_distance = matrix_distance(matrix, ultrametric_matrix, np.inf)
        results.append([method, l1_distance, l2_distance, linf_distance])

    # Выводим результаты в виде таблицы
    headers = ["Метод", "L1 (Манхэттенское)", "L2 (Евклидово)", "L∞ (Чебышевское)"]
    print(tabulate(results, headers=headers, tablefmt="pretty", floatfmt=".8f"))