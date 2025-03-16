import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import time
from input import *
from save import *
from clustering import *
from visualization import *
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
        matrix = generate_random_distance_matrix(n_points=n, integer_values=False)
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
    #start_time = time.time()
    #initial_clusters = [ClusterNode(i) for i in range(matrix.shape[0])]
    #Z_recur_minimax = recursive_minimax_linkage(initial_clusters, matrix, verbose=True)
    # Замеряем время окончания шага и выводим время выполнения
    #end_time = time.time()
    #print(f"время кластеризации Rminimax: {end_time - start_time:.4f} секунд")

    # Кастом линкейдж для построения дендрограммы
    start_time = time.time()
    Z_median = custom_linkage(matrix, method=median_distance)
    # Замеряем время окончания шага и выводим время выполнения
    end_time = time.time()
    print(f"время кластеризации median: {end_time - start_time:.4f} секунд")

    start_time = time.time()
    condensed_matrix = squareform(matrix)
    Z_complete = linkage(condensed_matrix, method='complete')
    #print("Финальная матрица связей complete Z:")
    #print(Z_complete)
    Z_single = linkage(condensed_matrix, method='single')
    Z_average = linkage(condensed_matrix, method='average')
    Z_ward = linkage(condensed_matrix, method='ward')
    end_time = time.time()
    print(f"время кластеризации: {end_time - start_time:.4f} секунд")
    #print("Является ли кластеризация монотонной?", is_monotonic(Z_custom))
    # Выводим финальную матрицу связей Z
    #print("Финальная матрица связей Z:")
    #print(Z_minmax)

    # Методы и матрицы
    methods = {
        'minmax': Z_minmax,
        'median': Z_median,
        'single': Z_single,
        'complete': Z_complete,
        'average': Z_average,
        'ward': Z_ward
    }

    # Построение ультраметрической матрицы
    n = matrix.shape[0]  # Количество объектов
    print(n)

    # Строим дендрограмму
    #plt.figure(figsize=(10, 5))
    #dendrogram(Z_minmax)
    #plt.title("Дендрограмма для метода «Минимакс»")
    #plt.xlabel("Индексы объектов")
    #plt.ylabel("Расстояние")
    #plt.show()



    n = matrix.shape[0]
    # Создаём ультраметрическую матрицу расстояний
    ultrametric_matrices = build_ultrametric_matrices(methods, n)
    print_ultrametric_matrices(ultrametric_matrices)

    # Использование разных норм
    norms = {
        'L1': 'L1',
        'L2': 'L2',
        'L_inf': np.inf
    }

    for norm_name, norm_func in norms.items():
        print(f"Используем норму: {norm_name}")
        calculate_and_print_distances(matrix, ultrametric_matrices, norm_func)

    calculate_and_print_all_distances(matrix, ultrametric_matrices)

    plot_dendrograms(methods)

if __name__ == "__main__":
    main()
