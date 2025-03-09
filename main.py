import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import time
from input import *
from save import *
from clustering import *
from visualization import visualize_matrix
from utils import *
from distancefunc import *



# Проверка монотонности
def is_monotonic(Z):
    for i in range(len(Z) - 1):
        if Z[i + 1, 2] < Z[i, 2]:
            return False
    return True
def main():
    # Пример ультраметрической матрицы
    matrix = np.array([
        [0, 2, 3, 3],
        [2, 0, 3, 3],
        [3, 3, 0, 1],
        [3, 3, 1, 0]
    ])

    print("Выберите способ создания матрицы:")
    print("1. Ввести матрицу вручную")
    print("2. Сгенерировать матрицу")
    print("3. Загрузить матрицу из файла")
    choice = input("Ваш выбор (1/2/3): ")

    if choice == "1":
        matrix = input_matrix_manually()
    elif choice == "2":
        n = int(input("Введите размерность матрицы (n): "))
        start_time = time.time()
        matrix = generate_random_distance_matrix(n)
        while is_ultrametric(matrix):
            matrix = generate_random_distance_matrix(n)
        end_time = time.time()
        print(f"Время поиска матрицы: {end_time - start_time:.4f} секунд")
    elif choice == "3":
        matrix = load_matrix_from_file()
        matrix = np.array(matrix, dtype=np.float64) # принудительное преобразование к float64
        #while not is_ultrametric(matrix):
        #    if is_ultrametric(matrix) == False:
        #        print("матрица не ультраметрична!")
        #    matrix = load_matrix_from_file()
        if matrix is None:
            return
    else:
        print("Неверный выбор!")
        return

    print("\nМатрица успешно создана/загружена:")
    print(matrix)

    if choice == "1" or choice == "2":
        save_choice = input("Хотите сохранить матрицу в файл? (y/n): ").lower()
        if save_choice == "y":
            save_matrix(matrix)

    print(matrix)
    # Тепловая карта
    # visualize_matrix(matrix)

    # Кастом линкейдж для построения дендрограммы
    start_time = time.time()
    Z_minmax = custom_linkage(matrix, method=minimax_distance)
    # Замеряем время окончания шага и выводим время выполнения
    end_time = time.time()
    print(f"время кластеризации minimax: {end_time - start_time:.4f} секунд")

    # Кастом линкейдж для построения дендрограммы
    start_time = time.time()
    initial_clusters = [ClusterNode(i) for i in range(matrix.shape[0])]
    #Z_recur_minimax = recursive_minimax_linkage(initial_clusters, matrix, verbose=True)
    # Замеряем время окончания шага и выводим время выполнения
    end_time = time.time()
    print(f"время кластеризации Rminimax: {end_time - start_time:.4f} секунд")

    # Кастом линкейдж для построения дендрограммы
    #start_time = time.time()
    #Z_median = custom_linkage(matrix, method=median_distance)
    # Замеряем время окончания шага и выводим время выполнения
    #end_time = time.time()
    #print(f"время кластеризации median: {end_time - start_time:.4f} секунд")

    start_time = time.time()
    condensed_matrix = squareform(matrix)
    Z_complete = linkage(condensed_matrix, method='complete')
    print("Финальная матрица связей complete Z:")
    print(Z_complete)
    Z_average = linkage(condensed_matrix, method='average')
    Z_ward = linkage(condensed_matrix, method='ward')
    end_time = time.time()
    print(f"время кластеризации: {end_time - start_time:.4f} секунд")
    #print("Является ли кластеризация монотонной?", is_monotonic(Z_custom))
    # Выводим финальную матрицу связей Z
    print("Финальная матрица связей Z:")
    print(Z_minmax)

    # Построение ультраметрической матрицы
    n = matrix.shape[0]  # Количество объектов
    print(n)
    ultrametric_matrix = build_block_matrix(Z_minmax, n)

    print("Ультраметрическая матрица minmax:")
    print(ultrametric_matrix)

    # Строим дендрограмму
    plt.figure(figsize=(10, 5))
    dendrogram(Z_minmax)
    plt.title("Дендрограмма для метода «Минимакс»")
    plt.xlabel("Индексы объектов")
    plt.ylabel("Расстояние")
    plt.show()



    n = matrix.shape[0]
    # Создаём ультраметрическую матрицу расстояний
    ultrametric_matrix_minmax = create_ultrametric_distance_matrix(Z_minmax, n)
    #ultrametric_matrix_recur_minmax = create_ultrametric_distance_matrix(Z_recur_minimax, n)
    #ultrametric_matrix_median = create_ultrametric_distance_matrix(Z_median, n)
    ultrametric_matrix_complete = create_ultrametric_distance_matrix(Z_complete, n)
    ultrametric_matrix_average = create_ultrametric_distance_matrix(Z_average, n)
    ultrametric_matrix_ward = create_ultrametric_distance_matrix(Z_ward, n)

    print("Ультраметрическая матрица расстояний minmax:")
    print(ultrametric_matrix_minmax)

    #print("Ультраметрическая матрица расстояний recur minmax:")
    #print(ultrametric_matrix_recur_minmax)

    #print("Ультраметрическая матрица расстояний median:")
    #print(ultrametric_matrix_median)

    print("Ультраметрическая матрица расстояний complete:")
    print(ultrametric_matrix_complete)

    print("Ультраметрическая матрица расстояний average:")
    print(ultrametric_matrix_average)

    print("Ультраметрическая матрица расстояний ward:")
    print(ultrametric_matrix_ward)

    distance = matrix_distance(matrix,ultrametric_matrix_minmax)
    print("Расстояние между матрицами (Frobenius) minmax:", distance)

    distance = relative_error(matrix,ultrametric_matrix_minmax)
    print("Относительная ошибка minmax:", distance)

    #distance = matrix_distance(matrix, ultrametric_matrix_median)
    #print("Расстояние между матрицами (Frobenius) median:", distance)

    #distance = relative_error(matrix, ultrametric_matrix_median)
    #print("Относительная ошибка median:", distance)

    distance = matrix_distance(matrix, ultrametric_matrix_complete)
    print("Расстояние между матрицами (Frobenius) complete:", distance)

    distance = relative_error(matrix, ultrametric_matrix_complete)
    print("Относительная ошибка complete:", distance)

    distance = matrix_distance(matrix, ultrametric_matrix_average)
    print("Расстояние между матрицами (Frobenius) average:", distance)

    distance = relative_error(matrix, ultrametric_matrix_average)
    print("Относительная ошибка average:", distance)

    distance = matrix_distance(matrix, ultrametric_matrix_ward)
    print("Расстояние между матрицами (Frobenius) ward:", distance)

    distance = relative_error(matrix, ultrametric_matrix_ward)
    print("Относительная ошибка ward:", distance)

    # Создаем фигуру с двумя подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Первая дендрограмма (minmax)
    dendrogram(Z_minmax, ax=axes[0, 0])
    axes[0, 0].set_title('Дендрограмма minmax')
    axes[0, 0].set_xlabel('Объекты')
    axes[0, 0].set_ylabel('Расстояние')

    # Вторая дендрограмма (complete)
    dendrogram(Z_complete, ax=axes[0, 1])
    axes[0, 1].set_title('Дендрограмма complete')
    axes[0, 1].set_xlabel('Объекты')
    axes[0, 1].set_ylabel('Расстояние')

    # Третья дендрограмма (average)
    dendrogram(Z_average, ax=axes[1, 0])
    axes[1, 0].set_title('Дендрограмма average')
    axes[1, 0].set_xlabel('Объекты')
    axes[1, 0].set_ylabel('Расстояние')

    # Четвертая дендрограмма (ward)
    dendrogram(Z_ward, ax=axes[1, 1])
    axes[1, 1].set_title('Дендрограмма ward')
    axes[1, 1].set_xlabel('Объекты')
    axes[1, 1].set_ylabel('Расстояние')

    # Настраиваем отступы между подграфиками
    plt.tight_layout()

    # Показываем график
    plt.show()

if __name__ == "__main__":
    main()
