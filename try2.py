from typing import List,Tuple

class Tensor:
    def __init__(self,data:List[List[float]]):
        if (not isinstance(data,list)) or (len(data) == 0):
            raise ValueError('data must be a non-empty two-dimensional list.')
        row_length = len(data[0])
        for row in data:
            if len(row) != row_length:
                raise ValueError('data must have the same number of cols.')
        self.data = [[num for num in row] for row in data]

    def shape(self) -> Tuple[int, int]:
        rows = len(self.data)
        if rows > 0:
            cols = len(self.data[0])
        else:
            cols = 0
        return rows, cols

    def transpose(self) -> 'Tensor':
        rows, cols = self.shape()
        transposed_data = [[0.0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed_data[j][i] = self.data[i][j]
        return Tensor(transposed_data)

    def add(self, other:'Tensor') -> 'Tensor':
        if self.shape() != other.shape():
            raise ValueError('The two tensors have different shapes.')
        rows, cols = self.shape()
        result_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result_data[i][j] = self.data[i][j] + other.data[i][j]
        return Tensor(result_data)

    def sub(self, other:'Tensor') -> 'Tensor':
        if self.shape() != other.shape():
            raise ValueError('The two tensors have different shapes.')
        rows, cols = self.shape()
        result_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result_data[i][j] = self.data[i][j] - other.data[i][j]
        return Tensor(result_data)

    def mul(self, scalar: float) -> 'Tensor':
        rows, cols = self.shape()
        result_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result_data[i][j] = self.data[i][j] * scalar
        return Tensor(result_data)

    def matmul(self, other:'Tensor') -> 'Tensor':
        self_rows, self_cols = self.shape()
        other_rows, other_cols = other.shape()
        if self_cols != other_rows:
            raise ValueError('cols must match the number of rows of the other tensor.')
        result_data = [[0.0 for _ in range(other_cols)] for _ in range(self_rows)]
        for i in range(self_rows):
            for j in range(other_cols):
                total = 0.0
                for k in range(self_cols):
                    total += self.data[i][k] * other.data[k][j]
                result_data[i][j] = total
        return Tensor(result_data)

    def mean(self, axis: int) -> List[float]:
        if axis not in [0,1]:
            raise ValueError('axis must be 0 or 1.')
        rows, cols = self.shape()
        mean_list =[]
        if axis == 0:
            for i in range(cols):
                col_sum = 0.0
                for j in range(rows):
                    col_sum += self.data[j][i]
                mean_list.append(col_sum / rows)
        else:
            for i in range(rows):
                row_sum = 0.0
                for j in range(cols):
                    row_sum += self.data[i][j]
                mean_list.append(row_sum / cols)
        return mean_list

    def std(self, axis: int) -> List[float]:
        if axis not in [0,1]:
            raise ValueError('axis must be 0 or 1.')
        rows, cols = self.shape()
        std_list = []
        if axis == 0:
            for j in range(cols):
                col_sum = 0.0
                for i in range(rows):
                    col_sum += self.data[i][j]
                col_mean = col_sum / rows
                variance = 0.0
                for i in range(rows):
                    variance += (self.data[i][j] - col_mean) ** 2
                variance = variance / rows
                std_list.append(variance ** 0.5)
        else:
            for i in range(rows):
                row_sum = 0.0
                for j in range(cols):
                    row_sum += self.data[i][j]
                row_mean = row_sum / cols
                variance = 0.0
                for j in range(cols):
                    variance += (self.data[i][j] - row_mean) ** 2
                variance = variance / cols
                std_list.append(variance ** 0.5)
        return std_list

    def min(self, axis: int) -> List[float]:
        if axis not in [0,1]:
            raise ValueError('axis must be 0 or 1.')
        rows, cols = self.shape()
        min_list = []
        if axis == 0:
            for j in range(cols):
                col_min = self.data[0][j]
                for i in range(1,rows):
                    if self.data[i][j] < col_min:
                        col_min = self.data[i][j]
                min_list.append(col_min)
        else:
            for i in range(rows):
                row_min = self.data[i][0]
                for j in range(1,cols):
                    if self.data[i][j] < row_min:
                        row_min = self.data[i][j]
                min_list.append(row_min)
        return min_list

    def max(self, axis: int) -> List[float]:
        if axis not in [0,1]:
            raise ValueError('axis must be 0 or 1.')
        rows, cols = self.shape()
        max_list = []
        if axis == 0:
            for j in range(cols):
                col_max = self.data[0][j]
                for i in range(1,rows):
                    if self.data[i][j] > col_max:
                        col_max = self.data[i][j]
                max_list.append(col_max)
        else:
            for i in range(rows):
                row_max = self.data[i][0]
                for j in range(1,cols):
                    if self.data[i][j] > row_max:
                        row_max = self.data[i][j]
                max_list.append(row_max)
        return max_list

    def standardize(self) -> 'Tensor':
        rows, cols = self.shape()
        col_means = self.mean(axis=0)
        col_stds = self.std(axis=0)
        standardized_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if col_stds[j] == 0.0:
                    standardized_data[i][j] = 0.0
                else:
                    standardized_data[i][j] = (self.data[i][j] -col_means[j]) / col_stds[j]
        return Tensor(standardized_data)



