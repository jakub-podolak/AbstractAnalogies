from abc import ABC

class EasyInferenceModel(ABC):
    """
    All models should support the forward method for compatibility
    """
    def forward(self, text: str):
        """
        Return output string
        """
        ...
