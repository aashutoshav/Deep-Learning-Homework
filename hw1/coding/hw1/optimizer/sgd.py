from ._base_optimizer import _BaseOptimizer
from models._base_network import _baseNetwork


class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model: _baseNetwork) -> None:
        """
        1) Apply regularization to the model gradients
        2) Update the model weights using the regularized gradients (don't forget to use self.learning_rate)

        :param model: The model to be updated
        :return: None, but the model.weights should be updated
        """
        ### TODO: BEGIN SOLUTION ###
        self.apply_regularization(model)
        for key in model.weights:
            model.weights[key] -= self.learning_rate * model.gradients[key]
        ### END SOLUTION ###
