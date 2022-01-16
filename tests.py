import Dirihle_One, Dirihle_Two, Eign_Values
import time
import numpy as np

h = 5  # кол-во разбиений
min = 0  # мин значение
max = 1  # макс значение

start_time = time.time()


"""1. Решаем Дирихле методом конечных разностей"""
test = Dirihle_One.Dirihle_sub(h, min, max)
test.init_net()
test.create_compute_net()
result = test.solve_sistem()
print('Дирихле конечными')
print(result)

print('время работы 1 метода')
print("--- %s seconds ---" % (time.time() - start_time))


"""2. Решаем Дирихле методом верхней релаксации"""
test = Dirihle_Two.Dirihle_sub(h, min, max)
test.init_net()
test.create_compute_net()
result = test.solve_sistem(1.5, 10)   #Выбираем константу и кол-во идераций
print('Дирихле МВР')
print(result)

print("время работы 2 метода")
print("--- %s seconds ---" % (time.time() - start_time))


"""3. Поиск собственных значений"""
n = 5
A = np.random.rand(n, n)
S = np.dot(A.T, A)
epoches = 10  #кол-во иттераций

test = Eign_Values.Find_eign_values(S)
print('Максимальное и минимальное собственные значения')
print(test.find_max(epoches))
print(test.find_min(epoches, 0.1))