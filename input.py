import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

# Ввод ультраметрической матрицы расстояний
def input_matrix_manually():
    """
    Ввод матрицы вручную.

    Возвращает:
    ----------
    matrix : ndarray
        Введённая пользователем матрица.
    """
    n = int(input("Введите размерность матрицы (n): "))
    print(f"Введите элементы матрицы {n}x{n} построчно, разделяя числа пробелами:")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Строка {i + 1}: ").split()))
        matrix.append(row)
    return np.array(matrix)

# Генерация ультраметрической матрицы расстояний
def generate_ultrametric(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = np.random.uniform(0, 100)
            matrix[j, i] = matrix[i, j]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                matrix[i, j] = min(matrix[i, j], max(matrix[i, k], matrix[k, j]))
    return matrix


def generate_random_distance_matrix(n_points, n_dimensions=2, random_seed=None):
    """
    Генерирует случайную матрицу расстояний.

    :param n_points: Количество точек.
    :param n_dimensions: Размерность пространства (по умолчанию 2D).
    :param random_seed: Сид для воспроизводимости результатов.
    :return: Матрица расстояний (n_points x n_points).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Генерация случайных точек в n-мерном пространстве
    points = np.random.rand(n_points, n_dimensions)

    # Вычисление попарных расстояний
    distances = pdist(points, metric='euclidean')

    # Преобразование в квадратную матрицу расстояний
    distance_matrix = squareform(distances)

    return distance_matrix

def load_matrix_from_file():
    """
    Загрузка матрицы из файла, обрабатывая возможные ошибки.

    Возвращает:
        ndarray: Загруженная матрица или None в случае ошибки.
    """
    filename = input("Введите имя файла: ")
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return None

    try:
        matrix = np.loadtxt(filename, delimiter=' ', dtype=np.float64) # delimiter=' ' - предполагаем, что разделитель - пробел
        # Проверка на квадратность матрицы и наличие нулей на главной диагонали
        rows, cols = matrix.shape
        if rows != cols:
            print("Ошибка: матрица не является квадратной")
            return None
        if not np.allclose(np.diag(matrix), 0):
            print("Ошибка: на главной диагонали не нули")
            return None

        return matrix

    except ValueError as e:
        print(f"Ошибка при загрузке матрицы: {e}. Проверьте формат файла.")
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return None
