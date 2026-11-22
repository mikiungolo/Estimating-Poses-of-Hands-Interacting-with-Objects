"""
HOI Model: Shared Backbone + Hand/Object Heads.
"""
from .backbone import SharedBackbone
from .heads import (
    HandJointsHead,
    HandMANOHead,
    HandTranslationHead,
    ObjectEmbedding,
    ObjectPoseHead,
)
from .model import HOIModel

__all__ = [
    "HOIModel",
    "SharedBackbone",
    "HandMANOHead",
    "HandJointsHead",
    "HandTranslationHead",
    "ObjectPoseHead",
    "ObjectEmbedding",
]
