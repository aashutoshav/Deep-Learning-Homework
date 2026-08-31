import numpy as np
from typing import Callable, Tuple
import torch
from jaxtyping import Float, Int  # type: ignore


def create_zero_vector(size: int = 10) -> Float[np.ndarray, " size"]:
    """
    Exercise 1: Create a zero vector of specified size.

    Args:
        size: Size of the vector (default: 10)

    Returns:
        Zero vector of specified size
    """
    ### BEGIN SOLUTION
    return np.zeros(size)
    ### END SOLUTION


def create_diagonal_matrix(
    size: int = 10, diag_value: int = -1
) -> Int[np.ndarray, "size size"]:
    """
    Exercise 2: Create an int64 matrix with diagonal values set to specified value.

    Args:
        size: Size of the square matrix (default: 10)
        diag_value: Value to set on diagonal (default: -1)

    Returns:
        int64 matrix with specified diagonal values
    """
    ### BEGIN SOLUTION
    matrix=np.eye(size,dtype=np.int64)
    matrix[matrix==1]=diag_value
    return matrix
    ### END SOLUTION


def create_checkerboard(size: int = 10) -> Float[np.ndarray, "size size"]:
    """
    Exercise 3: Create a checkerboard pattern matrix. matrix[0, 0] should be 0.

    Args:
        size: Size of the square matrix (default: 10)

    Returns:
        Matrix with checkerboard pattern (0s and 1s)
    """
    ### BEGIN SOLUTION
    checkerboard = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = 0
            else:
                checkerboard[i, j] = 1
    return checkerboard
    ### END SOLUTION


def place_random_ones(
    matrix_shape: Tuple[int, int] = (8, 8), num_ones: int = 5
) -> Float[np.ndarray, "height width"]:
    """
    Exercise 4: Randomly place specified number of 1's in a zero matrix.

    Args:
        matrix_shape: Shape of the matrix (default: (8, 8))
        num_ones: Number of 1's to place (default: 5)

    Returns:
        Matrix with randomly placed 1's
    """
    ### BEGIN SOLUTION
    matrix = np.zeros(matrix_shape)
    rows, cols = matrix_shape
    total_elements = rows * cols
    
    if num_ones > total_elements:
        raise ValueError("Number of ones exceeds the total number of elements in the matrix.")
    
    random_indices = np.random.choice(total_elements, num_ones, replace=False)
    row_indices = random_indices // cols
    col_indices = random_indices % cols
    matrix[row_indices, col_indices] = 1
    return matrix
    ### END SOLUTION


def channel_last_to_first(
    image: Float[np.ndarray, "height width channels"],
) -> Float[np.ndarray, "channels height width"]:
    """
    Exercise 5: Convert channel-last image to channel-first.

    Args:
        image: Image tensor in channel-last format (H, W, C)

    Returns:
        Image tensor in channel-first format (C, H, W)
    """
    ### BEGIN SOLUTION
    return np.transpose(image, (2, 0, 1))
    ### END SOLUTION


def rgb_to_bgr(
    image: Float[np.ndarray, "3 height width"],
) -> Float[np.ndarray, "3 height width"]:
    """
    Exercise 6: Convert RGB image to BGR by swapping color channels.

    Args:
        image: RGB image in channel-first format (3, H, W)

    Returns:
        BGR image in channel-first format (3, H, W)
    """
    ### BEGIN SOLUTION
    bgr_image = image.copy()
    bgr_image = bgr_image[::-1]
    return bgr_image
    ### END SOLUTION


def negate_range_inplace(
    array: Float[np.ndarray, " n"], min_val: float = 3, max_val: float = 8
) -> Float[np.ndarray, " n"]:
    """
    Exercise 7: Negate all elements between min_val and max_val, in place.

    Args:
        array: 1D array to modify
        min_val: Minimum value of range (inclusive, default: 3)
        max_val: Maximum value of range (inclusive, default: 8)

    Returns:
        Modified array (same object, modified in place)
    """
    ### BEGIN SOLUTION
    mask=(array>=min_val)&(array<=max_val)
    array[mask]*=-1
    return array
    ### END SOLUTION


def convert_dtype(array: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """
    Exercise 8: Convert array to specified dtype.

    Args:
        array: Input array
        target_dtype: Target data type

    Returns:
        np.ndarray converted to target dtype
    """
    ### BEGIN SOLUTION
    return array.astype(target_dtype)
    ### END SOLUTION


def subtract_row_means(
    matrix: Float[np.ndarray, "rows cols"],
) -> Float[np.ndarray, "rows cols"]:
    """
    Exercise 9: Subtract the mean of each row from the matrix.

    Args:
        matrix: Input matrix

    Returns:
        Matrix with row means subtracted
    """
    ### BEGIN SOLUTION
    row_means=np.mean(matrix,axis=1,keepdims=True)
    return matrix-row_means
    ### END SOLUTION


def sort_by_column(
    matrix: Float[np.ndarray, "rows cols"], column_idx: int = 1
) -> Float[np.ndarray, "rows cols"]:
    """
    Exercise 10: Sort matrix by specified column.

    Args:
        matrix: Input matrix
        column_idx: Column index to sort by (default: 1 for second column)

    Returns:
        Matrix sorted by specified column
    """
    ### BEGIN SOLUTION
    sorted_indices=np.argsort(matrix[:, column_idx])
    return matrix[sorted_indices]
    ### END SOLUTION


def one_hot_encode(
    indices: Int[np.ndarray, " n"], num_classes: int = 10
) -> Int[np.ndarray, "n num_classes"]:
    """
    Exercise 11: Convert integer array to one-hot encoding.

    Args:
        indices: np.ndarray of integer indices
        num_classes: Number of classes (default: 10)

    Returns:
        One-hot encoded matrix of shape (len(indices), num_classes)
    """
    ### BEGIN SOLUTION
    one_hot=np.zeros((len(indices),num_classes))
    one_hot[np.arange(len(indices)),indices]=1
    return one_hot
    ### END SOLUTION


def broadcast_multiply_rows(
    matrix: Float[np.ndarray, "rows cols"], vector: Float[np.ndarray, " rows"]
) -> Float[np.ndarray, "rows cols"]:
    """
    Exercise 12: Multiply nth row of matrix with nth element of vector using broadcasting.

    Args:
        matrix: Input matrix
        vector: Vector for multiplication

    Returns:
        Matrix with rows multiplied by corresponding vector elements
    """
    ### BEGIN SOLUTION
    vector_reshaped=vector[:,np.newaxis]
    return matrix*vector_reshaped
    ### END SOLUTION


def pad_with_zeros(
    array: Float[np.ndarray, "height width"], pad_width: int = 1
) -> Float[np.ndarray, "padded_height padded_width"]:
    """
    Exercise 13: Pad array with zeros without using np.pad.

    Args:
        array: Input array
        pad_width: Width of padding (default: 1)

    Returns:
        Padded array
    """
    ### BEGIN SOLUTION
    h,w=array.shape
    padded_array=np.zeros((h+2*pad_width,w+2*pad_width))
    padded_array[pad_width:pad_width+h,pad_width:pad_width+w]=array
    return padded_array
    ### END SOLUTION


def numerical_gradient(
    func: Callable[[Float[np.ndarray, " n"]], float],
    x: Float[np.ndarray, " n"],
    h: float = 1e-5,
) -> Float[np.ndarray, " n"]:
    """
    Exercise 13.a: Implement numerical gradient computation using central difference.

    Args:
        func: Function that takes numpy array and returns scalar
        x: Point at which to evaluate gradient
        h: Step size for finite difference (default: 1e-5)

    Returns:
        Approximate gradient at x
    """
    ### BEGIN SOLUTION
    gradient = np.zeros_like(x)
    h = 1e-5
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        gradient[i] = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
    return gradient
    ### END SOLUTION


def analytical_gradient_f1(x: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
    """
    Exercise 13.b: Analytical gradient of f(x) = x^T x.

    Args:
        x: Input vector

    Returns:
        Analytical gradient: 2x
    """
    ### BEGIN SOLUTION
    return 2*x
    ### END SOLUTION


def analytical_gradient_f2(x: Float[np.ndarray, "3"]) -> Float[np.ndarray, "3"]:
    """
    Exercise 13.b: Analytical gradient of f(x) = sin(x_1) + cos(x_2) + cos(x_3).

    Args:
        x: Input vector

    Returns:
        Analytical gradient: [cos(x_1), -sin(x_2), -sin(x_3)]
    """
    ### BEGIN SOLUTION
    return np.array([np.cos(x[0]),-np.sin(x[1]),-np.sin(x[2])])
    ### END SOLUTION


def analytical_gradient_f3(
    x: Float[np.ndarray, " n"], A: Float[np.ndarray, "n n"]
) -> Float[np.ndarray, " n"]:
    """
    Exercise 13.b: Analytical gradient of f(x) = x^T A x.

    Args:
        x: Input vector
        A: Matrix A

    Returns:
        Analytical gradient: (A + A^T) x
    """
    ### BEGIN SOLUTION
    return (A + A.T) @ x
    ### END SOLUTION


def complex_function(
    x: Float[np.ndarray, " n"], A: Float[np.ndarray, "n n"], b: Float[np.ndarray, " n"]
) -> float:
    """
    Exercise 13.c: Complex vector function for numerical differentiation.

    f(x) = (x^T A x) * sin(x^T b) + exp(-x^T x)

    Args:
        x: Input vector
        A: Matrix A
        b: Vector b

    Returns:
        Function value
    """
    ### BEGIN SOLUTION
    term1=(x.T@A@x)*np.sin(x.T@b)
    term2=np.exp(-x.T@x)
    return term1+term2
    ### END SOLUTION


# Test functions for exercises 14.b
def f1(x: Float[np.ndarray, " n"]) -> float:
    """Test function 1: f(x) = x^T x"""
    return np.dot(x, x)


def f2(x: Float[np.ndarray, "3"]) -> float:
    """Test function 2: f(x) = sin(x_1) + cos(x_2) + cos(x_3)"""
    return np.sin(x[0]) + np.cos(x[1]) + np.cos(x[2])


def f3(x: Float[np.ndarray, " n"], A: Float[np.ndarray, "n n"]) -> float:
    """Test function 3: f(x) = x^T A x"""
    return np.dot(x, np.dot(A, x))
