import numpy as np
from jaxtyping import Float, Int  # type: ignore


class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self, X, y, mode="train") -> tuple[float, float]:
        raise NotImplementedError("Subclass must implement forward method")

    def softmax(
        self, scores: Float[np.ndarray, "batch_size num_classes"]
    ) -> Float[np.ndarray, "batch_size num_classes"]:
        """
        Compute softmax probabilities given the raw output from the model
        Make sure to use the numerically stable softmax implementation.

        :param scores: raw scores from the model (batch_size, num_classes)
        :return:
            prob: softmax probabilities (batch_size, num_classes)
        """
        prob = None
        ### TODO: BEGIN SOLUTION ###
        shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        prob = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        ### END SOLUTION ###
        return prob

    def cross_entropy_loss(
        self,
        probs: Float[np.ndarray, "batch_size num_classes"],
        y: Int[np.ndarray, "batch_size"],
    ) -> float:
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels.
        Don't forget to return the mean of the loss over the batch.

        This is slightly different from nn.CrossEntropyLoss in PyTorch in that
        nn.CrossEntropyLoss expects unnormalized logits as input
        whereas our cross_entropy_loss expects normalized probabilities as input.

        :param probs: Predicted probabilities of shape (batch_size, num_classes).
        :param y: Labels of instances in the batch
        :return: The Cross-Entropy Loss averaged over the batch
        """
        loss = None
        ### TODO: BEGIN SOLUTION ###
        N = probs.shape[0]
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-15))
        ### END SOLUTION ###
        return loss

    def compute_accuracy(
        self, probs: Float[np.ndarray, "N num_classes"], y: Int[np.ndarray, "N"]
    ) -> float:
        """
        Compute the accuracy of current batch

        :param probs: Predicted probabilities of shape (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        accuracy = None
        ### TODO: BEGIN SOLUTION ###
        preds = np.argmax(probs, axis=-1)
        accuracy = float(np.mean(preds == y))
        ### END SOLUTION ###
        return accuracy

    def sigmoid(self, x: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        """
        Compute sigmoid(x)

        :param x: some input array of any shape
        :return:
            out: sigmoid(x), should have the same shape as x
        """
        out = None
        ### TODO: BEGIN SOLUTION ###
        out = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
        ### END SOLUTION ###
        return out

    def sigmoid_dev(self, x: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        """
        The analytical derivative of sigmoid(x)

        :param x: some input array of any shape
        :return: The derivative of sigmoid(x), should have the same shape as x
        """
        out = None
        ### TODO: BEGIN SOLUTION ###
        s = self.sigmoid(x)
        out = s * (1.0 - s)
        ### END SOLUTION ###
        return out

    def ReLU(self, x: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        """
        Compute ReLU(x)

        :param x: some input array of any shape
        :return:
            out: ReLU(x), should have the same shape as x
        """
        out = None
        ### TODO: BEGIN SOLUTION ###
        out = np.maximum(0.0, x)
        ### END SOLUTION ###
        return out

    def ReLU_dev(self, x: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        """
        Compute the gradient of ReLU(x)

        :param x: some input array of any shape
        :return:
            out: gradient of ReLU(x), should have the same shape as x
        """
        out = None
        ### TODO: BEGIN SOLUTION ###
        out = (x > 0).astype(float)
        ### END SOLUTION ###
        return out
