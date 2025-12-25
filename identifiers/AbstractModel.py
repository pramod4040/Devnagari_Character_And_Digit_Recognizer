from abc import ABC, abstractmethod

class AbstractDigitModel(ABC):
    """
    Abstract base class for Devanagari digit recognition models.
    This allows for different model implementations while maintaining a consistent interface.
    """
    
    @abstractmethod
    def __init__(self, weight_path: str, class_names: list):
        """
        Initialize the model.
        
        Args:
            weight_path (str): Path to the model weights
            class_names (list): List of class names for classification
        """
        pass
    
    @abstractmethod
    def create_model(self):
        """
        Create the underlying model architecture.
        This method should set up the model structure.
        """
        pass
    
    @abstractmethod
    def load_weights(self):
        """
        Load pre-trained weights into the model.
        """
        pass
    
    # @abstractmethod
    # def preprocess_image(self, image_path: str):
    #     """
    #     Preprocess an image for model inference.
        
    #     Args:
    #         image_path (str): Path to the input image
            
    #     Returns:
    #         Preprocessed image in the format expected by the model
    #     """
    #     pass
    
    @abstractmethod
    def predict(self, image):
        """
        Make a prediction on the preprocessed image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Prediction result
        """
        pass
