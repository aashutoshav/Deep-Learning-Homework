from ._base_network import _baseNetwork
import numpy as np
from jaxtyping import Float, Int  # type: ignore

np.random.seed(1024)


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network

        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (input_size, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights["b1"] = np.zeros(self.hidden_size)
        self.weights["b2"] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights["W1"] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights["W2"] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients["W1"] = np.zeros((self.input_size, self.hidden_size))
        self.gradients["b1"] = np.zeros(self.hidden_size)
        self.gradients["W2"] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients["b2"] = np.zeros(self.num_classes)

    def forward(
        self,
        X: Float[np.ndarray, "batch_size input_size"],
        y: Int[np.ndarray, "batch_size"],
        mode: str = "train",
    ) -> tuple[float, float]:
        """
        The forward pass of the two-layer net. The network consists of the following layers:
        - Linear layer: input_size --> hidden_size
        - Sigmoid activation function
        - Linear layer: hidden_size --> num_classes
        - Softmax activation function

        Forward Pass:
        1) Compute the model output:
            1) Call the sigmoid function between the two layers
            2) The output of the second layer should be passed to softmax
                function before computing the cross entropy loss
        2) Compute Cross-Entropy Loss and batch accuracy using the predicted probabilities

        Backward Pass
        1) Compute gradients of each weight and bias by chain rule
        2) Store the gradients in self.gradients
        HINT: You will need to compute gradients backwards, i.e, compute
            gradients of W2 and b2 first, then compute it for W1 and b1.
            You may also want to implement the analytical derivative of
            the sigmoid function in self.sigmoid_dev first

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients, otherwise just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        accuracy = None

        # Forward Pass
        ### TODO: BEGIN SOLUTION ###
        N = X.shape[0]
        z1 = X @ self.weights["W1"] + self.weights["b1"]
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.weights["W2"] + self.weights["b2"]
        probs = self.softmax(z2)
        loss = self.cross_entropy_loss(probs, y)
        accuracy = self.compute_accuracy(probs, y)
        ### TODO: END SOLUTION ###
        if mode != "train":
            return loss, accuracy

        # Backward Pass
        ### TODO: BEGIN SOLUTION ###
        d_z2 = probs.copy()
        d_z2[np.arange(N), y] -= 1.0
        d_z2 /= N

        self.gradients["W2"] = a1.T @ d_z2
        self.gradients["b2"] = np.sum(d_z2, axis=0)

        d_a1 = d_z2 @ self.weights["W2"].T
        d_z1 = d_a1 * self.sigmoid_dev(z1)

        self.gradients["W1"] = X.T @ d_z1
        self.gradients["b1"] = np.sum(d_z1, axis=0)
        ### END SOLUTION ###

        return loss, accuracy
