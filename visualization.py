import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from distancefunc import *
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков
from input import*


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


def plot_dendrograms(Z_matrices):
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


def plot_points(points):
    """
    Визуализирует точки на графике в зависимости от их размерности.

    Параметры:
    -----------
    points : ndarray
        Массив точек размером (n_points, n_dimensions), где n_dimensions — это 1, 2 или 3.
    """
    # Проверка входных данных
    if not isinstance(points, np.ndarray):
        raise ValueError("Входные данные должны быть массивом NumPy.")
    if points.ndim != 2:
        raise ValueError("Массив точек должен быть двумерным (n_points, n_dimensions).")

    n_points, n_dimensions = points.shape

    if n_dimensions == 1:
        # Одномерный случай
        plt.figure(figsize=(8, 2))
        plt.scatter(points[:, 0], np.zeros(n_points), c='blue', alpha=0.6)
        plt.title("Одномерные точки")
        plt.xlabel("X")
        plt.yticks([])  # Скрываем ось Y, так как она не нужна
        plt.grid(True)

    elif n_dimensions == 2:
        # Двумерный случай
        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.6)
        plt.title("Двумерные точки")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')  # Чтобы масштабы осей были одинаковыми

    elif n_dimensions == 3:
        # Трехмерный случай
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.6)
        ax.set_title("Трехмерные точки")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    else:
        raise ValueError("Размерность точек должна быть 1, 2 или 3.")

    plt.show()


def display_experiment_info(experiment_dir, seed):
    """
    Отображает информацию по эксперименту и строит дендрограммы.

    Параметры:
    -----------
    experiment_dir : str
        Папка, где сохранены результаты экспериментов.
    seed : int
        Сид эксперимента.
    """
    # Загружаем результаты эксперимента
    results = load_experiment(experiment_dir, seed)

    # Выводим исходную матрицу расстояний
    print("Исходная матрица расстояний:")
    print(results['distance_matrix'])

    # Выводим ультраметричные матрицы для каждого метода
    print("\nУльтраметричные матрицы:")
    for method, matrix in results['ultrametric_matrices'].items():
        print(f"{method}:")
        print(matrix)

    # Выводим матрицы связей Z для каждого метода
    print("\nМатрицы связей Z:")
    for method, Z in results['linkage_matrices'].items():
        print(f"{method}:")
        print(Z)

    # Выводим расстояния для каждого метода
    print("\nРасстояния (метод, L1, L2, Linf):")
    for row in results['distances']:
        print(row)

    # Строим дендрограммы
    print("\nСтроим дендрограммы...")
    plot_dendrograms(results['linkage_matrices'])