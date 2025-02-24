
from scipy.cluster.hierarchy import dendrogram, linkage
from save import save_matrix
from visualization import visualize_matrix
from utils import is_ultrametric
import matplotlib.pyplot as plt
from input import input_matrix_manually, generate_ultrametric, load_matrix_from_file
import time
def main():
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

  # Преобразуем матрицу расстояний в сжатую форму (condensed form)
  from scipy.spatial.distance import squareform
  condensed_matrix = squareform(ultrametric_matrix)

  # Выполняем иерархическую кластеризацию методом Complete
  Z = linkage(condensed_matrix, method='complete')

  # Визуализируем дендрограмму
  plt.figure(figsize=(10, 5))
  dendrogram(Z)
  plt.title('Дендрограмма иерархической кластеризации (Complete)')
  plt.xlabel('Индекс точки')
  plt.ylabel('Расстояние')
  plt.show()

if __name__ == "__main__":
    main()