from models._base_network import _baseNetwork


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model: _baseNetwork) -> None:
        pass

    def apply_regularization(self, model: _baseNetwork) -> None:
        """
        Apply L2 penalty to the model by updating the gradient dictionary
        Don't forget to use the regularization coefficient.
        Don't apply regularization to the bias terms.

        :param model: The model with gradients
        :return: None, but the model.gradients should be updated
        """
        ### TODO: BEGIN SOLUTION ###
        for key in model.weights:
            if not key.startswith("b"):
                model.gradients[key] += self.reg * model.weights[key]
        ### END SOLUTION ###
