from typing import Dict, Type
from .component import ComponentG, ComponentD
from .vae import MlpVae, MlpSharingVae, CnnVae, CnnSharingVae, ResNetVae
from .classifier import (
    MlpClassifier,
    MlpSharingClassifier,
    CnnClassifier,
    ResNetClassifier,
    ResNetSharingClassifier,
)


G: Dict[str, Type[ComponentG]] = {
    'mlp_vae': MlpVae,
    'mlp_sharing_vae': MlpSharingVae,
    'cnn_vae': CnnVae,
    'cnn_sharing_vae': CnnSharingVae,
    'resnet_vae': ResNetVae,
}
D: Dict[str, Type[ComponentD]] = {
    'mlp_classifier': MlpClassifier,
    'mlp_sharing_classifier': MlpSharingClassifier,
    'cnn_classifier': CnnClassifier,
    'resnet_classifier': ResNetClassifier,
    'resnet_sharing_classifier': ResNetSharingClassifier,
}
