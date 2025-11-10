# check if X^T * X is invertible
# def check_inverse_rank(matrix):
#     rank = matrix_rank(matrix, tol=1e-12)
#     print("matrix rank is : "+ str(rank))
#     print("matrix size is : "+ str(matrix.shape))

#     if matrix.shape[0] == matrix.shape[1]:
#        if rank == matrix.shape[0]:
#            print("matrix is invertible")
#        else:
#            print("matrix is not invertible")
#     else:
#        print("matrix is not square, hence not invertible")

#     return (rank == matrix.shape[0]) and (matrix.shape[0] == matrix.shape[1])

# def check_inverse_det(matrix, tol=1e-12):
#     deter = det(matrix)
#     print("determinant is : " + str(deter))
#     if abs(deter) < tol:
#         print("matrix is invertible")
#     else:
#         print("matrix is not invertible")