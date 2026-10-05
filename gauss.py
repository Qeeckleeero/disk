"""
Метод Гаусса с выбором главного элемента.

Решает СЛАУ A*x = b (см. НМ, п. 1.2):
  1) прямой ход - приводим расширенную матрицу (A|b) к треугольному виду,
     на каждом шаге выбирая в столбце наибольший по модулю элемент
     ("главный") и переставляя его строку на текущую позицию;
  2) обратный ход - находим x, поднимаясь по треугольной системе снизу вверх.

Использование в других лабораторных:
    from gauss import solve_gauss
    x = solve_gauss(A, b)
"""

import numpy as np


def forward_elimination(augmented):
    """Прямой ход: приводит (A|b) к треугольному виду с единицами на диагонали."""
    matrix = augmented.astype(float)
    n = matrix.shape[0]

    for k in range(n):
        # ищем главный элемент - наибольший по модулю в столбце k, начиная со строки k
        pivot_row = k + np.argmax(np.abs(matrix[k:, k]))
        pivot_value = matrix[pivot_row, k]

        if pivot_value == 0:
            raise np.linalg.LinAlgError("матрица вырождена")

        matrix[[k, pivot_row]] = matrix[[pivot_row, k]]  # переставляем строки
        matrix[k] /= pivot_value  # получаем единицу на диагонали

        for i in range(k + 1, n):  # обнуляем столбец k под диагональю
            matrix[i] -= matrix[i, k] * matrix[k]

    return matrix


def back_substitution(triangular):
    """Обратный ход: находит x по треугольной матрице, полученной в forward_elimination."""
    n = triangular.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = triangular[i, -1] - triangular[i, i + 1:n] @ x[i + 1:n]

    return x


def solve_gauss(A, b):
    """Решает A*x = b методом Гаусса с выбором главного элемента."""
    augmented = np.column_stack((A, b)).astype(float)
    triangular = forward_elimination(augmented)
    return back_substitution(triangular)


if __name__ == "__main__":
    # пример из пособия (п. 1.2.3)
    A = [[1, 2, 3, 4], [3, 5, 1, 7], [8, 2, 0, -2], [6, 6, 6, 9]]
    b = [22, 38, 16, 60]
    print(solve_gauss(A, b))
