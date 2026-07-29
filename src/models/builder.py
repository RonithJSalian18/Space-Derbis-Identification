from .factory import ModelFactory


def get_model(architecture_name="cnn", learning_rate=None):
    """
    Factory wrapper function to instantiate and compile model by name.
    Delegates to ModelFactory for architecture construction and compilation.
    """
    return ModelFactory.create_model(
        architecture_name=architecture_name,
        learning_rate=learning_rate
    )
