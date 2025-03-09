import numpy as np

# Вектор Z, полученный после кластеризации
Z = np.array([
    [2., 3., 1., 2.],
    [0., 1., 1., 2.],
    [4., 5., 2.5, 4.]
])

# Количество объектов
n = 4


# Функция для построения блочной матрицы расстояний
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


# Построение блочной матрицы расстояний
block_matrix = build_block_matrix(Z, n)

print("Блочная матрица расстояний:")
print(block_matrix)