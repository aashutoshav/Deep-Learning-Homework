import numpy as np
from jaxtyping import Float, Int  # type: ignore
from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU => Softmax

        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self) -> None:
        """
        initialize weights of the single layer regression network. No bias term included.
        - W1: The weight matrix of the linear layer of shape (input_size, num_classes)
        - gradients: The gradient dictionary of the linear layer of shape (input_size, num_classes)

        :return: None
        """
        np.random.seed(1024)
        self.weights["W1"] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients["W1"] = np.zeros((self.input_size, self.num_classes))

    def forward(
        self,
        X: Float[np.ndarray, "batch_size input_size"],
        y: Int[np.ndarray, "batch_size"],
        mode: str = "train",
    ) -> tuple[float, float]:
        """
        Compute loss, accuracy, and gradients. Your implementation should be vectorized.
            1) Forward pass: Compute the model's output and the Cross-Entropy loss of the probabilities w.r.t. the labels.
                - Layers: Linear without bias -> ReLU -> Softmax
            2) Backward pass: Compute the gradients w.r.t. the loss

            Helpful hints:
            1) Don't forget to store the gradients in self.gradients
            1) You may find the intermediate outputs before ReLU useful for the backward pass.

        :param X: a batch of images (batch_size, 28x28)
        :param y: labels of images in the batch (batch_size,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        accuracy = None
        ### TODO: BEGIN SOLUTION ###
        raise NotImplementedError('TODO: Implement this function')
        ### END SOLUTION ###
        if mode != "train":
            return loss, accuracy

        # Backward pass
        ### TODO: BEGIN SOLUTION ###
        raise NotImplementedError('TODO: Implement this function')
        ### END SOLUTION ###
        return loss, accuracy
