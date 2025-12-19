"""
Federated Learning Package for Distributed Cyber Defense
=========================================================
Phase 3: Federated War Mode

This package implements privacy-preserving distributed learning:
- Local models train on local traffic (never share raw data)
- Exchange gradients/model updates only
- Global defense brain emerges from collective intelligence
- Each subnet becomes a neuron in the defense organism

Architecture:
- FederatedClient: Local training node
- FederatedServer: Central aggregation coordinator  
- SecureAggregator: Privacy-preserving gradient aggregation
- AdversarialTrainer: GAN-based evasion resistance
"""

from .federated_client import FederatedClient, LocalTrainer
from .federated_server import FederatedServer, ModelAggregator
from .secure_aggregator import SecureAggregator, DifferentialPrivacy
from .adversarial_trainer import AdversarialTrainer, AttackerGAN, DefenderDiscriminator

__all__ = [
    # Federated Learning
    'FederatedClient',
    'LocalTrainer',
    'FederatedServer', 
    'ModelAggregator',
    'SecureAggregator',
    'DifferentialPrivacy',
    
    # Adversarial Training
    'AdversarialTrainer',
    'AttackerGAN',
    'DefenderDiscriminator'
]
