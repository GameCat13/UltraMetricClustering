import numpy as np
from scipy.cluster.hierarchy import dendrogram
# Проверка ультраметрической матрицы расстояний
def is_ultrametric(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # Проверяем только верхний треугольник
            for k in range(n):
                if k == i or k == j:
                    continue  # Пропускаем случаи, когда k совпадает с i или j
                if matrix[i, j] > max(matrix[i, k], matrix[j, k]):
                    return False
    return True

def check_distance_matrix(matrix):
    """
    Проверяет, является ли данная матрица корректной матрицей расстояний.

    Args:
        matrix: NumPy array, предполагаемая матрица расстояний.

    Returns:
        Tuple: (bool, str), где bool - True, если матрица корректна, False - иначе.
               str - сообщение с описанием ошибки, если матрица некорректна. Возвращает пустую строку, если матрица корректна.
    """

    rows, cols = matrix.shape
    if rows != cols:
        return False, "Матрица не является квадратной."

    if not np.allclose(matrix, matrix.T): # Проверяем симметричность с учетом возможных погрешностей вычислений
        return False, "Матрица не является симметричной."

    if not np.all(np.diag(matrix) == 0):
        return False, "Диагональ матрицы не содержит нулей."

    if not np.all(matrix >= 0):
        return False, "Матрица содержит отрицательные значения."

    return True, ""


# Функция для построения ультраметрической матрицы
def build_block_matrix(Z, n):
    # Инициализация матрицы расстояний
    distance_matrix = np.zeros((n, n))

    # Список для хранения кластеров
    clusters = [[i] for i in range(n)]

    # Проходим по всем строкам Z
    for row in Z:
        cluster1, cluster2, distance, _ = row
        cluster1, cluster2 = int(cluster1), int(cluster2)

        # Получаем объекты из объединяемых кластеров
        objects1 = clusters[cluster1]
        objects2 = clusters[cluster2]

        # Обновляем расстояния между всеми объектами из двух кластеров
        for i in objects1:
            for j in objects2:
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Обновляем расстояния внутри нового кластера (если нужно)
        # В данном случае это не требуется, так как расстояния внутри кластеров уже заданы

        # Объединяем кластеры
        new_cluster = objects1 + objects2
        clusters.append(new_cluster)

    return distance_matrix

def build_ultrametric_matrices(methods, n):
    matrices = {}
    for method in methods:
        matrices[method] = build_block_matrix(methods[method], n)
    return matrices


def l1_norm(matrix1, matrix2):
    difference = matrix1 - matrix2
    return np.linalg.norm(difference, ord=1)

def l2_norm(matrix1, matrix2):
    difference = matrix1 - matrix2
    return np.linalg.norm(difference, ord=2)

def linf_norm(matrix1, matrix2):
    difference = matrix1 - matrix2
    return np.linalg.norm(difference, ord=np.inf)
