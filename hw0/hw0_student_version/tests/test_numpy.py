import pytest
import unittest
import numpy as np

from main import (
    create_zero_vector,
    create_diagonal_matrix,
    create_checkerboard,
    place_random_ones,
    channel_last_to_first,
    rgb_to_bgr,
    negate_range_inplace,
    convert_dtype,
    subtract_row_means,
    sort_by_column,
    one_hot_encode,
    broadcast_multiply_rows,
    pad_with_zeros,
    numerical_gradient,
    analytical_gradient_f1,
    analytical_gradient_f2,
    analytical_gradient_f3,
    f1,
    f2,
    f3,
)


class TestBasicArrayOperations(unittest.TestCase):
    """Test basic array creation and manipulation functions."""

    def test_create_zero_vector(self):
        """Test zero vector creation."""
        # Test default size
        vector = create_zero_vector()
        assert vector.shape == (10,)
        assert np.all(vector == 0)

        # Test custom size
        vector = create_zero_vector(5)
        assert vector.shape == (5,)
        assert np.all(vector == 0)

    def test_create_diagonal_matrix(self):
        """Test diagonal matrix creation."""
        # Test default parameters
        matrix = create_diagonal_matrix()
        assert matrix.shape == (10, 10)
        assert matrix.dtype == np.int64
        assert np.all(np.diag(matrix) == -1)
        assert np.all(matrix[np.triu_indices(10, k=1)] == 0)  # Upper triangle
        assert np.all(matrix[np.tril_indices(10, k=-1)] == 0)  # Lower triangle

        # Test custom parameters
        matrix = create_diagonal_matrix(5, 3)
        assert matrix.shape == (5, 5)
        assert np.all(np.diag(matrix) == 3)
        assert np.all(matrix[np.triu_indices(5, k=1)] == 0)
        assert np.all(matrix[np.tril_indices(5, k=-1)] == 0)

    def test_create_checkerboard(self):
        """Test checkerboard pattern creation."""
        matrix = create_checkerboard(4)
        expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        assert np.array_equal(matrix, expected)

    def test_place_random_ones(self):
        """Test random ones placement."""
        np.random.seed(42)
        matrix = place_random_ones((3, 3), 3)
        assert matrix.shape == (3, 3)
        assert np.sum(matrix) == 3  # Exactly 3 ones
        assert np.all((matrix == 0) | (matrix == 1))  # Only 0s and 1s


class TestImageProcessing(unittest.TestCase):
    """Test image processing functions."""

    def test_channel_last_to_first(self):
        """Test channel convention conversion."""
        # Create a test image (H=2, W=3, C=3)
        image = np.random.randn(2, 3, 3)
        converted = channel_last_to_first(image)

        assert converted.shape == (3, 2, 3)  # (C, H, W)
        # Check that data is preserved
        assert np.array_equal(converted[0], image[:, :, 0])
        assert np.array_equal(converted[1], image[:, :, 1])
        assert np.array_equal(converted[2], image[:, :, 2])

    def test_rgb_to_bgr(self):
        """Test RGB to BGR conversion."""
        # Create RGB image (3, H, W)
        rgb_image = np.random.randn(3, 4, 4)
        bgr_image = rgb_to_bgr(rgb_image)

        assert bgr_image.shape == rgb_image.shape
        # Check channel swapping: BGR = [R, G, B] -> [B, G, R]
        assert np.array_equal(bgr_image[0], rgb_image[2])  # B = R
        assert np.array_equal(bgr_image[1], rgb_image[1])  # G = G
        assert np.array_equal(bgr_image[2], rgb_image[0])  # R = B


class TestArrayManipulation(unittest.TestCase):
    """Test array manipulation functions."""

    def test_negate_range_inplace(self):
        """Test in-place negation of range."""
        array = np.arange(11).astype(float)  # [0, 1, 2, ..., 10]
        original_id = id(array)

        result = negate_range_inplace(array, 3, 8)

        # Check that it's the same object (in-place)
        assert id(result) == original_id

        # Check expected values
        expected = np.array([0, 1, 2, -3, -4, -5, -6, -7, -8, 9, 10])
        assert np.array_equal(result, expected)

    def test_convert_dtype(self):
        """Test dtype conversion."""
        array = np.zeros(5, dtype=np.float64)
        converted = convert_dtype(array, np.uint8)

        assert converted.dtype == np.uint8
        assert np.array_equal(converted, np.zeros(5, dtype=np.uint8))

    def test_subtract_row_means(self):
        """Test row mean subtraction."""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = subtract_row_means(matrix)

        # Check that row means are approximately zero
        row_means = np.mean(result, axis=1)
        assert np.allclose(row_means, 0, atol=1e-10)

    def test_sort_by_column(self):
        """Test matrix sorting by column."""
        matrix = np.array([[1, 3], [2, 1], [3, 2]])
        sorted_matrix = sort_by_column(matrix, 1)  # Sort by second column

        expected = np.array([[2, 1], [3, 2], [1, 3]])
        assert np.array_equal(sorted_matrix, expected)

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        indices = np.array([0, 2, 1, 3])
        one_hot = one_hot_encode(indices, 4)

        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        assert np.array_equal(one_hot, expected)

    def test_broadcast_multiply_rows(self):
        """Test broadcasting multiplication."""
        matrix = np.ones((3, 4))
        vector = np.array([1, 2, 3])
        result = broadcast_multiply_rows(matrix, vector)

        expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        assert np.array_equal(result, expected)

    def test_pad_with_zeros(self):
        """Test zero padding."""
        array = np.ones((2, 2))
        padded = pad_with_zeros(array, 1)

        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        assert np.array_equal(padded, expected)


class TestGradientComparison(unittest.TestCase):
    """Test comparison between numerical and analytical gradients."""

    def test_gradient_consistency(self):
        """Test that numerical and analytical gradients are close."""
        x = np.array([1.0, 1.0, 1.0])

        # Test f1
        num_grad1 = numerical_gradient(f1, x)
        ana_grad1 = analytical_gradient_f1(x)
        assert np.allclose(num_grad1, ana_grad1, atol=1e-4)

        # Test f2
        num_grad2 = numerical_gradient(f2, x)
        ana_grad2 = analytical_gradient_f2(x)
        assert np.allclose(num_grad2, ana_grad2, atol=1e-4)

        # Test f3
        A = np.random.randn(3, 3)
        num_grad3 = numerical_gradient(lambda x: f3(x, A), x)
        ana_grad3 = analytical_gradient_f3(x, A)
        assert np.allclose(num_grad3, ana_grad3, atol=1e-4)
