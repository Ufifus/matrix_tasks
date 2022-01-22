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

    """Апроксимируем вторую производную и решаем систему методом верхней релаксации"""

    def solve_sistem(self, w, epoches):
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

        L, D, r = self.LDR_decompose(A, R)
        B = D + w * L

        for epoch in range(epoches):
            R_n = F - np.dot(A, U)
            error = np.dot(np.linalg.inv(B), R_n)
            U = U + w * error

        self.U[1:self.h - 1, 1:self.h - 1] = U.reshape(self.h - 2, self.h - 2)
        return self.U

    """Разложение матрицы на нижнюю верхнюю и диагональную"""

    def LDR_decompose(self, A, R):
        """Lower matrix"""
        L = np.zeros((R, R))
        for i, row in enumerate(A):
            L[i, :i] = row[:i]

        """Upper matrix"""
        Rm = np.zeros((R, R))
        for i, row in enumerate(A):
            Rm[i, i + 1:] = row[i + 1:]

        """Diagonal matrix"""
        D = np.zeros((R, R))
        for i, row in enumerate(A):
            D[i, i] = row[i]

        return L, D, Rm