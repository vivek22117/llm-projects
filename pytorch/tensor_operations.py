
import torch

# "*" and "@"

# "*" means element wise multiplication  (Multiplies matching positions)
# For "*" the tensor must have exact same shape

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])


# this calculates [[1*10, 2*20], [3*30, 4*40]]
element_wise_mul = a * b
print(element_wise_mul)

# "@" Matrix Multiplications  (Rule: the number of columns in m1 = number of rows in m2)

m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
m2 = torch.tensor([[7, 8], [9, 10], [11, 34]])  # shape (3, 2)

# for m1 and m2 the inner dimensions matches (3)
# when we do matrix multiplication of m1 and m2 the result will match with outer dimensions (2, 2)

# Calculation [[1*7 + 2*9 + 3*11], [1*8 + 2*10 + 3*34]
#               [4*7 + 2*9 + 3*11], [4*8 + 5*10 + 6*34]]
matrix_mul_result = m1 @ m2 # shape (2, 2)
print(matrix_mul_result)
# tensor([[ 58, 130],
#         [139, 286]])


# Reduction operation on tensor (sum, mean, max) Reduces the matrix size
# Rows represents: Student Scores in diff assignments
# Columns represents: Assignments Scores
input_tensor = torch.tensor([[10., 20., 30.], [40., 50., 60.]])

# calculates 10 + 20 + 30 + 40 + 50 + 60 / 6
average_score = input_tensor.mean()
print(f"Overall means: {average_score}")  # Overall means: 35.0


# the "dim" argument helps us to control on which direction we want to perform reduction/calculations

# dim=0 Collapses the rows, Operates "Vertically"
avg_per_assignments = input_tensor.mean(dim=0)
print(f"Average of each assignments {avg_per_assignments}")  # Average of each assignments tensor([25., 35., 45.])

# dim=1 or dim=-1 Collapses the Columns, Operates "Horizontally"
avg_per_students = input_tensor.mean(dim=-1)
print(f"Average of each student {avg_per_students}")  # Average of each student tensor([20., 50.])



# Reading data from matrix/tensors
x = torch.arange(12).reshape(3, 4)
print(f"Input tensor {x}")
# Input tensor tensor([[ 0,  1,  2,  3],
#                      [ 4,  5,  6,  7],
#                      [ 8,  9, 10, 11]])
col_2 = x[:, 2]

print(f"Get all the rows but only from column at index 2 {col_2}")
# Get all the rows but only from column at index 2 tensor([ 2,  6, 10])


# ARGMAX, finds the indx of the highest value.
scores = torch.tensor([
    [10, 0, 30, 5, 12],
    [1, 56, 8, 32, 11]
])

best_indices = torch.argmax(scores, dim=1)
print(f"Best index of the best score for each row {best_indices}") # Best index of the best score for each row tensor([2, 1])


