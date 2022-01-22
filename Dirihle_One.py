import numpy as np
import math


class Dirihle_sub:

    def __init__(self, h, min, max):
        self.h = h  # задаем кол-во разбиений
        self.min = min  # минимальное значение в сетке
        self.max = max  # максимальное значение в сетке
        self.U = []  # сетка значений
        self.F = []  # значение ф-и в точках сетки

    """Создаем сетку значений и ф-ю в каждой точке сетки"""

    def init_net(self):
        x = np.linspace(self.min, self.max, self.h)
        y = np.linspace(self.min, self.max, self.h)

        for i, el_x in enumerate(x):
            for j, el_y in enumerate(y):
                if el_x == self.min or el_y == self.min:
                    self.U.append(0)
                elif el_x == self.max:
                    self.U.append(el_y)
                elif el_y == self.max:
                    self.U.append(el_x)
                else:
                    self.U.append(0)
                self.F.append(1 / (self.h ** 2) * 10 * math.exp(el_x + el_y))

    """Создаем расчетную сетку для внутренних значений"""

    def create_compute_net(self):
        R = self.h ** 2
        g = np.zeros(R)
        g[:R] -= np.dot(self.U[:R], 4)
        g[:R - 1] += self.U[1:R]
        g[1:R] += self.U[:R - 1]
        g[:R - self.h] += self.U[self.h:R]
        g[self.h:R] += self.U[:R - self.h]

        self.U = np.array(self.U).reshape(self.h, self.h)
        self.F = self.F - g
        self.F = np.array(self.F).reshape(self.h, self.h)

    """Апроксимируем вторую производную и решаем систему"""

    def solve_sistem(self):
        R = (self.h - 2) ** 2
        A = np.zeros((R, R))
        A[np.arange(R), np.arange(R)] = -4
        A[np.arange(R - 1), np.arange(1, R)] = 1
        A[np.arange(1, R), np.arange(R - 1)] = 1
        A[np.arange(R - self.h), np.arange(self.h, R)] = 1
        A[np.arange(self.h, R), np.arange(R - self.h)] = 1
        U = self.U[1:self.h - 1, 1:self.h - 1]
        near_y = []
        for i, row in enumerate(U):
            for j, col in enumerate(row):
                y_mas = []
                if j == 0:
                    prev_y = 0
                else:
                    prev_y = 1
                if j == 2:
                    next_y = 0
                else:
                    next_y = 1
                y_mas.append(prev_y)
                y_mas.append(next_y)
                near_y.append(y_mas)
        for i in range(1, R - 1):
            if near_y[i][0] == 0:
                A[i][i - 1] = 0
            else:
                A[i][i - 1] = 1
            if near_y[i][1] == 0:
                A[i][i + 1] = 0
            else:
                A[i][i + 1] = 1

        U = self.U[1:self.h - 1, 1:self.h - 1].reshape(R)
        F = self.F[1:self.h - 1, 1:self.h - 1].reshape(R)
        Q, r = self.QR_decompose(A, R)
        F = Q.T.dot(F)
        U[R - 1] = F[R - 1] / r[R - 1, R - 1]
        for i in np.arange(R - 1)[::-1]:
            U[i] = (F[i] - np.dot(r[i, i + 1:], U[i + 1:])) / r[i, i]

        self.U[1:self.h - 1, 1:self.h - 1] = U.reshape(self.h - 2, self.h - 2)
        return self.U

    """алгоритм QR-разложения"""

    def QR_decompose(self, A, R):
        Q = np.zeros((R, R))
        R = np.zeros((R, R))

        A = A.T
        Q[0, :] = A[0, :]
        for i, col_a in enumerate(A[1:, :]):
            sum = 0
            for col_b in Q[:i + 1, :]:
                dot_ab = np.dot(col_a, col_b)
                norm_b = np.dot(col_b, col_b)
                sum += (dot_ab / norm_b) * col_b
            col_a = col_a - sum
            Q[i + 1, :] = col_a

        for i, row in enumerate(Q):
            Q[i, :] = row / math.sqrt(np.dot(row, row))

        for i, row in enumerate(Q):
            for j, col in enumerate(A):
                R[i, j] = np.dot(row, col)
        return Q.T, R