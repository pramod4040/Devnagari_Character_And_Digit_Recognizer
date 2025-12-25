import os
from typing import Dict, Type, Optional
from .AbstractModel import AbstractDigitModel
from .EnsembleModel import EnsembleModelCharacter
from .MoeModelOne import MoeModelOne
from .MoeModelNew import MoeModelNew
from .MoeModelOptimized import EnhancedMoeModel
from .MoeModelV2Good import MoeModelV2Good
from .MoeModelWithoutGating400 import MoeModelWithoutGating400
from .CustomCnn32 import CustomCNNModel

class ModelFactory:
    """Factory class for creating digit recognition models"""
    
    _models: Dict[str, Type[AbstractDigitModel]] = {
        'ensemble': EnsembleModelCharacter,
        'moe': MoeModelOne,
        'newMoe': MoeModelNew,
        'enhancedMoeModel': EnhancedMoeModel,
        'BestEnsembleModel': MoeModelV2Good,
        'moeModelWithoutGating400': MoeModelWithoutGating400,
        'customCNNModel': CustomCNNModel
        # Add more model types here as they are implemented
    }

    @classmethod
    def get_models(cls) -> Dict[str, Type[AbstractDigitModel]]:
        return cls._models
    
    @classmethod
    def create_model(cls, 
                    model_type: str, 
                    weight_path: Optional[str] = None,
                    class_names: Optional[list] = None) -> AbstractDigitModel:
        """
        Create a model instance based on the specified type and parameters.
        
        Args:
            model_type (str): Type of model to create ('ensemble', etc.)
            weight_path (str, optional): Path to model weights. If None, uses default path
            class_names (list, optional): List of class names. If None, uses default names
            
        Returns:
            AbstractDigitModel: Instance of the requested model
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._models.keys())}")
            
        # Use default weight path if none provided
        if weight_path is None:
            weight_path = os.path.join(
                os.getcwd(), 
                "identifiers", 
                "trained_model", 
                "model_weight_ensemble.h5"
            )
        
        # Use default class names if none provided
        if class_names is None:
            from .EnsembleModel import get_class_names
            class_names = get_class_names()
            
        # Create and return the model instance
        model_class = cls._models[model_type]
        print("Creating model is about to done!")
        return model_class(weight_path, class_names)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[AbstractDigitModel]):
        """
        Register a new model type
        
        Args:
            name (str): Name to register the model under
            model_class (Type[AbstractDigitModel]): Model class to register
        """
        cls._models[name] = model_class
