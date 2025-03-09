import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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

def _get_cluster_indices(cluster):
    """Рекурсивно получаем все индексы объектов в кластере."""
    if cluster.left is None and cluster.right is None:
        return [cluster.id]
    return _get_cluster_indices(cluster.left) + _get_cluster_indices(cluster.right)

def median_distance(cluster1_indices, cluster2_indices, distance_matrix):
    """Вычисляет медианное расстояние между двумя кластерами."""
    distances = []
    for i in cluster1_indices:
        for j in cluster2_indices:
            distances.append(distance_matrix[i, j])
    return np.median(distances)

def custom_linkage(distance_matrix, method=median_distance, verbose=False):
    """
    Выполняет иерархическую кластеризацию с использованием пользовательского метода.
    :param distance_matrix: Матрица расстояний между объектами.
    :param method: Функция для вычисления расстояния между кластерами.
    :param verbose: Если True, выводит подробную информацию о каждом шаге.
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
        if verbose:
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
                if dist <= min_dist:
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
        if verbose:
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
        if verbose:
            print(f"Текущие кластеры: {clusters}")
            print(f"Матрица связей Z: [[{Z[step-1][0]}, {Z[step-1][1]}, {Z[step-1][2]:.4f}, {Z[step-1][3]}]]")
            # Замеряем время окончания шага и выводим время выполнения
            end_time = time.time()
            print(f"Время выполнения шага: {end_time - start_time:.4f} секунд")
    return np.array(Z)


# Пример матрицы расстояний
distance_matrix = np.array([
    [0, 1, 2, 3],
    [1, 0, 4, 5],
    [2, 4, 0, 6],
    [3, 5, 6, 0]
])


# Размер матрицы
n = 10

# Генерируем случайные целочисленные данные для матрицы расстояний
# Используем диапазон от 1 до 20 (можно изменить по желанию)
np.random.seed(42)  # Для воспроизводимости результата
data = np.random.randint(1, 20, size=(n, n))

# Делаем матрицу симметричной
distance_matrix = (data + data.T) // 2  # Используем целочисленное деление

# Обнуляем диагональ (расстояние от точки до самой себя равно 0)
np.fill_diagonal(distance_matrix, 0)

print("Матрица расстояний 10x10 (целочисленная):")
print(distance_matrix)

# Выполняем кластеризацию с медианным методом
Z = custom_linkage(distance_matrix, method=median_distance, verbose=True)

print(Z)

# Строим дендрограмму
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Дендрограмма для метода «Минимакс»")
plt.xlabel("Индексы объектов")
plt.ylabel("Расстояние")
plt.show()