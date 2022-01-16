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

  """Поиск максимального собственного значения МПИ"""
  def find_max(self, epoches):
    x = np.random.rand(self.n)
    mas = []

    for epoch in range(epoches):
      y = np.dot(self.A, x)
      y_norm = np.dot(y, y)
      eign_max = np.dot(x, y) / np.dot(x, x)
      x = y / y_norm
    return eign_max

  """Поиск минимального собственного значения МОИ"""
  def find_min(self, epoches, m):
    b = np.random.rand(self.n)
    m = 0.01
    diag = np.ones(self.n)
    E = np.diag(diag)
    mE = m * E


    for epoch in range(epoches):
      dot = self.A - mE
      dot = np.linalg.inv(dot)
      b = np.dot(dot, b)
      b_norm = np.dot(b, b)
      b = b / b_norm

      Ab = np.dot(self.A, b)
      eign_min = np.dot(b, Ab) / np.dot(b, b)
    return eign_min