import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





# Пример данных
data = [[0,0], [0,3] , [4,3]]

# Преобразуем вложенный список в массив NumPy
np_array = np.array(data)
# Проверяем размерность
cols,rows = np_array.shape
if rows == 2 and cols > 0: # Условие: 2 строки и хотя бы один столбец
  # Разделяем данные на координаты x и y
  x = np.array([point[0] for point in data])
  y = np.array([point[1] for point in data])
  # Построение графика
  plt.plot(x, y, 'o') # 'o' указывает на отображение точками

  # Настройка графика (необязательно)
  plt.xlabel("Ось X")
  plt.ylabel("Ось Y")
  plt.title("График точек")
  plt.grid(True) # Добавление сетки

  # Отображение графика
  plt.show()

  Z = sch.linkage(data, method='single', optimal_ordering=True)
  np.set_printoptions(suppress=True)  # подавляет научную нотацию
  print(Z)

  # Обрезка дендрограммы для получения 2 кластеров
  clusters = fcluster(Z, 2, criterion='maxclust')
  print(clusters)
  # Построение дендрограммы
  plt.figure(figsize=(10, 7))
  sch.dendrogram(Z, p=5, truncate_mode='level')
  plt.show()