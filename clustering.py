import numpy as np
import time

# Класс для представления узла дерева
class ClusterNode:
    def __init__(self, id, left=None, right=None, size=1):
        self.id = id  # Уникальный идентификатор узла
        self.left = left  # Левый дочерний узел
        self.right = right  # Правый дочерний узел
        self.size = size  # Количество объектов в кластере

    def __repr__(self):
        # Если это листовой узел (кластер из одного объекта)
        if self.left is None and self.right is None:
            return str(self.id)

        # Если один из дочерних узлов — это список, объединяем их в один список
        left_repr = str(self.left)
        right_repr = str(self.right)

        # Если левый или правый узел — это список, извлекаем элементы
        left_elements = left_repr[1:-1].split(", ") if left_repr.startswith("[") else [left_repr]
        right_elements = right_repr[1:-1].split(", ") if right_repr.startswith("[") else [right_repr]

        # Объединяем элементы в один список
        combined_elements = left_elements + right_elements
        return f"[{', '.join(combined_elements)}]"

# Функция для вычисления расстояния «Минимакс» между двумя кластерами
def minimax_distance(cluster1, cluster2, distance_matrix):
    """
    Вычисляет минимаксное расстояние между двумя кластерами.
    :param cluster1: Первый кластер (список индексов объектов).
    :param cluster2: Второй кластер (список индексов объектов).
    :param distance_matrix: Матрица расстояний между объектами.
    :return: Минимаксное расстояние между кластерами.
    """
    distances = distance_matrix[np.ix_(cluster1, cluster2)]
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return (min_dist + max_dist) / 2

def custom_linkage(distance_matrix, method):
    """
    Выполняет иерархическую кластеризацию с использованием пользовательского метода.
    :param distance_matrix: Матрица расстояний между объектами.
    :param method: Функция для вычисления расстояния между кластерами.
    :return: Матрица связей Z.
    """
    n = distance_matrix.shape[0]
    clusters = [ClusterNode(i) for i in range(n)]  # Начальные кластеры (каждый объект — отдельный узел)
    Z = []  # Матрица связей
    next_cluster_id = n  # Уникальный идентификатор для новых кластеров
    step = 0  # Счётчик шагов

    # Инициализируем текущую матрицу расстояний между кластерами
    current_distances = np.copy(distance_matrix)

    while len(clusters) > 1:
        step += 1
        print(f"\nШаг {step}:")
        # Замеряем время начала шага
        start_time = time.time()

        min_dist = float('inf')
        best_i, best_j = -1, -1

        # Ищем пару кластеров с минимальным расстоянием
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Получаем индексы объектов в кластерах
                cluster1_indices = _get_cluster_indices(clusters[i])
                cluster2_indices = _get_cluster_indices(clusters[j])

                # Вычисляем расстояние между кластерами
                dist = method(cluster1_indices, cluster2_indices, distance_matrix)
                if dist < min_dist:
                    min_dist = dist
                    best_i, best_j = i, j

        # Объединяем кластеры
        new_cluster = ClusterNode(
            id=next_cluster_id,
            left=clusters[best_i],
            right=clusters[best_j],
            size=clusters[best_i].size + clusters[best_j].size
        )
        next_cluster_id += 1

        # Добавляем шаг в матрицу связей
        Z.append([clusters[best_i].id, clusters[best_j].id, min_dist, new_cluster.size])

        # Выводим информацию о шаге до удаления кластеров
        print(f"Объединяем кластеры {clusters[best_i].id} и {clusters[best_j].id} в новый кластер {new_cluster.id}")
        print(f"Расстояние между кластерами: {min_dist}")

        # Удаляем объединенные кластеры и добавляем новый
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in (best_i, best_j)]
        clusters.append(new_cluster)

        # Пересчитываем расстояния до нового кластера с помощью формулы Ланса-Уильямса
        new_distances = np.zeros((len(clusters), len(clusters)))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if i == len(clusters) - 1 or j == len(clusters) - 1:
                    # Вычисляем расстояние до нового кластера
                    cluster1_indices = _get_cluster_indices(clusters[i])
                    cluster2_indices = _get_cluster_indices(clusters[j])
                    new_distances[i, j] = method(cluster1_indices, cluster2_indices, distance_matrix)
                else:
                    # Используем старое расстояние
                    new_distances[i, j] = current_distances[i, j]
        current_distances = new_distances
        # Выводим текущие кластеры и матрицу связей Z
        print(f"Текущие кластеры: {clusters}")
        print(f"Матрица связей Z: [[{Z[0][0]}, {Z[0][1]}, {Z[0][2]:.4f}, {Z[0][3]}]]")
        # Замеряем время окончания шага и выводим время выполнения
        end_time = time.time()
        print(f"Время выполнения шага: {end_time - start_time:.4f} секунд")
    return np.array(Z)

def _get_cluster_indices(node):
    """
    Рекурсивно получает все индексы объектов в кластере.
    :param node: Узел дерева.
    :return: Список индексов объектов.
    """
    if node.left is None and node.right is None:
        return [node.id]
    return _get_cluster_indices(node.left) + _get_cluster_indices(node.right)
