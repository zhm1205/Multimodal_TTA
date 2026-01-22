"""
Component registry module
Used for dynamically registering and retrieving components like models, datasets, attack methods, etc.
"""

from typing import Dict, Type, Any, Callable
import importlib


class Registry:
    """Component registry"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str, cls: Type = None) -> Callable:
        """
        Register component
        
        Can be used as decorator:
        @registry.register("component_name")
        class Component:
            pass
            
        Or called directly:
        registry.register("component_name", Component)
        """
        def _register(cls: Type) -> Type:
            if name in self._registry:
                print(f"Warning: {name} is already registered in {self.name}")
            self._registry[name] = cls
            return cls
        
        if cls is not None:
            return _register(cls)
        else:
            return _register
    
    def get(self, name: str) -> Type:
        """get registered component"""
        if name not in self._registry:
            raise KeyError(f"{name} is not registered in {self.name}")
        return self._registry[name]
    
    def has(self, name: str) -> bool:
        """check if component is registered"""
        return name in self._registry
    
    def list_all(self) -> list:
        """list all registered components"""
        return list(self._registry.keys())
    
    def clear(self) -> None:
        """clear registry"""
        self._registry.clear()


# create global registry
MODELS = Registry("models")
DATASETS = Registry("datasets")
DATASET_BUILDERS = Registry("dataset_builders")
EVALUATION_STRATEGIES = Registry("evaluation_strategies")
CRITERIA = Registry("criteria")
PROVIDERS = Registry("providers")
PLUGINS = Registry("plugins")

def register_provider(name: str):
    """Decorator to register a UE training-free provider."""
    return PROVIDERS.register(name)

def register_model(name: str):
    """register model"""
    return MODELS.register(name)

def register_dataset(name: str):
    """register dataset"""
    return DATASETS.register(name)

def register_criterion(name: str):
    """register criterion"""
    return CRITERIA.register(name)


def register_dataset_builder(name: str):
    """register dataset builder"""
    return DATASET_BUILDERS.register(name)


def register_evaluation_strategy(name: str):
    """register evaluation strategy"""
    return EVALUATION_STRATEGIES.register(name)

def register_plugin(name: str):
    """register plugin"""
    return PLUGINS.register(name)

def get_model(name: str) -> Type:
    """get model class"""
    return MODELS.get(name)

def get_dataset(name: str) -> Type:
    """get dataset class"""
    return DATASETS.get(name)

def get_criterion(name: str) -> Type:
    """get criterion class"""
    return CRITERIA.get(name)

def get_provider(name: str) -> Type:
    """get provider class"""
    return PROVIDERS.get(name)

def get_dataset_builder(name: str) -> Type:
    """get dataset builder class"""
    return DATASET_BUILDERS.get(name)

def get_evaluation_strategy(name: str) -> Type:
    """get evaluation strategy class"""
    return EVALUATION_STRATEGIES.get(name)

def get_plugin(name: str) -> Type:
    """get plugin class"""
    return PLUGINS.get(name)


def list_all_components() -> Dict[str, list]:
    """List all registered components"""
    return {
        "models": MODELS.list_all(),
        "datasets": DATASETS.list_all(),
        "dataset_builders": DATASET_BUILDERS.list_all(),
        "evaluation_strategies": EVALUATION_STRATEGIES.list_all(),
        "criteria": CRITERIA.list_all(),
        "providers": PROVIDERS.list_all(),
        "plugins": PLUGINS.list_all(),
    }


def list_models() -> list:
    """List all registered models"""
    return MODELS.list_all()

def list_datasets() -> list:
    """List all registered datasets"""
    return DATASETS.list_all()


def list_dataset_builders() -> list:
    """List all registered dataset builders"""
    return DATASET_BUILDERS.list_all()


def list_evaluation_strategies() -> list:
    """List all registered evaluation strategies"""
    return EVALUATION_STRATEGIES.list_all()

def list_criteria() -> list:
    """List all registered criteria"""
    return CRITERIA.list_all()

def list_providers() -> list:
    """List all registered providers"""
    return PROVIDERS.list_all()

def list_plugins() -> list:
    """List all registered plugins"""
    return PLUGINS.list_all()