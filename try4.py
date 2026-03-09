from typing import List,Tuple,Union

class Tensor:
    def __init__(self,data:List[List[float]]):
        if (not isinstance(data,list)) or (len(data) == 0):
            raise ValueError('data must be a non-empty two-dimensional list.')
        row_length = len(data[0])
        for row in data:
            if len(row) != row_length:
                raise ValueError('data must have the same number of cols.')
        self.data: List[List[float]] = [[num for num in row] for row in data]

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

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape()})"

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return self.add(other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return self.sub(other)

    def __mul__(self, scalar:float) -> 'Tensor':
        return self.mul(scalar)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def __getitem__(self,key:Union[Tuple[int,slice]]) -> 'Tensor':
        if isinstance(key,tuple) and len(key) == 2:
            row_slice,col_slice = key
            sliced_data = [row[col_slice] for row in self.data[row_slice]]
            return Tensor(sliced_data)
        elif isinstance(key,int):
            return Tensor([self.data[key]])
        else:
            raise ValueError('key must be either a tuple of two slices or an integer')

A_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B_data = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
A = Tensor(A_data)
B = Tensor(B_data)

C = A + B
D = B - A
E = A * 2
F = A @ B.transpose()
H = A[0:2, 1:3] * 3

print("C = A + B：", C)
print("D = B - A：", D)
print("E = A * 2：", E)
print("F = A @ B.T()：", F)
print("H = A[0:2, 1:3] * 3：", H)