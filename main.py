import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import time
from input import input_matrix_manually, generate_ultrametric, load_matrix_from_file
from save import save_matrix
from clustering import custom_linkage, minimax_distance
from visualization import visualize_matrix
from utils import is_ultrametric


def main():
    # Пример ультраметрической матрицы
    ultrametric_matrix = np.array([
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
        ultrametric_matrix = input_matrix_manually()
    elif choice == "2":
        start_time = time.time()
        ultrametric_matrix = generate_ultrametric()
        while not is_ultrametric(ultrametric_matrix):
            ultrametric_matrix = generate_ultrametric()
        end_time = time.time()
        print(f"Время поиска матрицы: {end_time - start_time:.4f} секунд")
    elif choice == "3":
        ultrametric_matrix = load_matrix_from_file()
        while not is_ultrametric(ultrametric_matrix):
            if is_ultrametric(ultrametric_matrix) == False:
                print("матрица не ультраметрична!")
        if ultrametric_matrix is None:
            return
    else:
        print("Неверный выбор!")
        return

    print("\nМатрица успешно создана/загружена:")
    print(ultrametric_matrix)

    if choice == "1" or choice == "2":
        save_choice = input("Хотите сохранить матрицу в файл? (y/n): ").lower()
        if save_choice == "y":
            save_matrix(ultrametric_matrix)

    print(ultrametric_matrix)
    # Тепловая карта
    visualize_matrix(ultrametric_matrix)

    # Кастом линкейдж для построения дендрограммы
    Z_custom = custom_linkage(ultrametric_matrix, method=minimax_distance)

    # Выводим финальную матрицу связей Z
    print("Финальная матрица связей Z:")
    print(Z_custom)

    # Строим дендрограмму
    plt.figure(figsize=(10, 5))
    dendrogram(Z_custom)
    plt.title("Дендрограмма для метода «Минимакс»")
    plt.xlabel("Индексы объектов")
    plt.ylabel("Расстояние")
    plt.show()

if __name__ == "__main__":
    main()
