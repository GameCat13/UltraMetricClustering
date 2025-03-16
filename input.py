import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
import pickle

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


def generate_binary_points(n_points, n_dimensions, random_seed=None):
    """
    Генерирует случайные бинарные точки.

    Параметры:
    -----------
    n_points : int
        Количество точек.
    n_dimensions : int
        Размерность пространства.
    random_seed : int, optional
        Сид для воспроизводимости результатов.

    Возвращает:
    -----------
    points : ndarray
        Массив бинарных точек размером (n_points, n_dimensions).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Генерация бинарных точек (0 или 1)
    points = np.random.randint(2, size=(n_points, n_dimensions))
    return points
def generate_random_distance_matrix(n_points, n_dimensions=2, metric='euclidean', integer_values=False, random_seed=None):
    """
    Генерирует случайную матрицу расстояний и точки в круге (2D) или шаре (3D и выше) с радиусом от 0 до 100.

    Параметры:
    -----------
    n_points : int
        Количество точек.
    n_dimensions : int, optional
        Размерность пространства (по умолчанию 2D).
    metric : str, optional
        Метрика для вычисления расстояний. По умолчанию 'euclidean'.
        Другие варианты: 'cityblock', 'cosine', 'chebyshev', 'hamming' и т.д.
    integer_values : bool, optional
        Если True, расстояния округляются до целых чисел. По умолчанию False.
    random_seed : int, optional
        Сид для воспроизводимости результатов. По умолчанию None.

    Возвращает:
    -----------
    distance_matrix : ndarray
        Матрица расстояний размером n_points x n_points.
    points : ndarray
        Массив точек размером n_points x n_dimensions.
    random_seed : int
        Сид, использованный для генерации.
    """
    # Проверка входных данных
    if n_points <= 0:
        raise ValueError("Количество точек должно быть положительным числом.")
    if n_dimensions <= 0:
        raise ValueError("Размерность пространства должна быть положительным числом.")
    if metric not in ['euclidean', 'cityblock', 'cosine', 'chebyshev', 'hamming']:
        raise ValueError("Неподдерживаемая метрика.")

    # Установка сида для воспроизводимости
    rng = np.random.default_rng(random_seed)  # Используем Generator
    if random_seed is None:
        random_seed = rng.integers(0, 2**32 - 1, dtype=np.uint32)  # Генерация случайного сида
    # Генерация точек
    if metric == 'hamming':
        # Генерация бинарных точек для метрики Хэмминга
        points = generate_binary_points(n_points, n_dimensions, random_seed)
    else:
        # Генерация случайных точек в круге (2D) или шаре (3D и выше) с радиусом от 0 до 100
        if n_dimensions == 2:
            # Генерация точек в круге
            radius = rng.uniform(0, 100, n_points)  # Случайный радиус от 0 до 100
            angle = rng.uniform(0, 2 * np.pi, n_points)  # Случайный угол
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.column_stack((x, y))
        else:
            # Генерация точек в шаре
            radius = rng.uniform(0, 100, n_points) ** (1/n_dimensions)  # Случайный радиус от 0 до 100
            direction = rng.normal(size=(n_points, n_dimensions))  # Случайное направление
            direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]  # Нормализация
            points = radius[:, np.newaxis] * direction

    # Вычисление попарных расстояний
    distances = pdist(points, metric=metric)

    # Преобразование в квадратную матрицу расстояний
    distance_matrix = squareform(distances)

    # Округление до целых чисел, если требуется
    if integer_values:
        distance_matrix = np.round(distance_matrix)

    return distance_matrix, points, random_seed

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


def load_experiment(experiment_dir, seed=None):
    """
    Загружает результаты эксперимента по сиду или все эксперименты, если сид не указан.

    Параметры:
    -----------
    experiment_dir : str
        Папка, где сохранены результаты экспериментов.
    seed : int, optional
        Сид эксперимента. Если None, загружаются все эксперименты.

    Возвращает:
    -----------
    results : dict или list
        Если указан seed, возвращает словарь с результатами одного эксперимента.
        Если seed не указан, возвращает список словарей с результатами всех экспериментов.
    """
    if seed is not None:
        # Загружаем один эксперимент
        file_path = os.path.join(experiment_dir, f"experiment_{seed}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Эксперимент с сидом {seed} не найден.")

        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Загружаем все эксперименты
        results = []
        for file_name in os.listdir(experiment_dir):
            if file_name.startswith("experiment_") and file_name.endswith(".pkl"):
                file_path = os.path.join(experiment_dir, file_name)
                with open(file_path, 'rb') as f:
                    results.append(pickle.load(f))
        return results