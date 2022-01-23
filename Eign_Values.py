import numpy as np


class Find_eign_values:
  def __init__(self, A):
    if A.shape[0] != A.shape[1]:
      raise ValueError('None square matrix')
    else:
      self.n = A.shape[0]
    for i, _ in enumerate(A):
      row = A[i, i:]
      col = A[i:, i]
      for j, _ in enumerate(row):
        if row[j] != col[j]:
          raise ValueError('None symmetric matrix')
    self.A = A
    self.vals = np.linalg.eigh(self.A)[0]

  """Поиск максимального собственного значения МПИ"""
  def find_max(self, epoches, accuracy):
    x = np.random.rand(self.n)
    max_value = round(self.vals.max(), accuracy)
    eign_max = 0

    for epoch in range(epoches):
      if round(eign_max, accuracy) == max_value:
        print('макс. собственное значение: ', eign_max)
        print('кол-во иттераций: ', epoch + 1)
        break
      y = np.dot(self.A, x)
      y_norm = np.dot(y, y)
      eign_max = np.dot(x, y) / np.dot(x, x)
      x = y / y_norm
    return eign_max

  """Поиск минимального собственного значения МОИ"""
  def find_min(self, epoches, const, accuracy):
    b = np.random.rand(self.n)
    min_value = round(self.vals.min(), accuracy)
    eign_min = 0
    m = const
    diag = np.ones(self.n)
    E = np.diag(diag)
    mE = m * E


    for epoch in range(epoches):
      if round(eign_min, accuracy) == min_value:
        print('мин. собственное значение: ', eign_min)
        print('кол-во иттераций: ', epoch + 1)
        break
      dot = self.A - mE
      dot = np.linalg.inv(dot)
      b = np.dot(dot, b)
      b_norm = np.dot(b, b)
      b = b / b_norm

      Ab = np.dot(self.A, b)
      eign_min = np.dot(b, Ab) / np.dot(b, b)
    return eign_min