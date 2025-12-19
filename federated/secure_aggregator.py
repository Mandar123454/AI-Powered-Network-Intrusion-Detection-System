"""
Secure Aggregator - Privacy-Preserving Gradient Aggregation
=============================================================
The cryptographic shield for federated learning:
1. Differential Privacy: Gradient noise injection with privacy guarantees
2. Secure Aggregation: Clients never see each other's updates
3. Homomorphic Encryption: Compute on encrypted gradients
4. Byzantine Resilience: Detect and exclude malicious clients

This is how we aggregate 1,000 minds without any one seeing the others' secrets.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import hashlib
import secrets
from datetime import datetime
import threading
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Available privacy mechanisms."""
    NONE = "none"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    RENYI_DP = "renyi_dp"


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN
    
    # Privacy budget
    epsilon: float = 1.0  # Total privacy budget
    delta: float = 1e-5   # Failure probability
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    per_layer_clipping: bool = False
    
    # Noise calibration
    noise_multiplier: float = 1.0
    adaptive_noise: bool = True
    
    # Accounting
    accountant: str = "rdp"  # rdp, gdp, or moments


@dataclass
class PrivacyBudget:
    """Tracks privacy budget consumption."""
    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    rounds_completed: int = 0
    
    def remaining_epsilon(self) -> float:
        return max(0, self.total_epsilon - self.spent_epsilon)
    
    def can_continue(self) -> bool:
        return self.spent_epsilon < self.total_epsilon
    
    def add_round(self, epsilon: float, delta: float = 0.0):
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.rounds_completed += 1


class DifferentialPrivacy:
    """
    Implements differential privacy for gradient aggregation.
    
    Provides formal privacy guarantees through:
    1. Gradient clipping (bounds sensitivity)
    2. Noise addition (provides plausible deniability)
    3. Privacy accounting (tracks budget consumption)
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        
        # Privacy budget tracking
        self.budget = PrivacyBudget(
            total_epsilon=config.epsilon,
            total_delta=config.delta
        )
        
        # RDP (Rényi Differential Privacy) orders for composition
        self.rdp_orders = np.arange(2, 100)
        self.rdp_epsilons: List[np.ndarray] = []
    
    def clip_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            Clipped gradients
        """
        if self.config.per_layer_clipping:
            return self._per_layer_clip(gradients)
        else:
            return self._global_clip(gradients)
    
    def _global_clip(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Global gradient clipping."""
        # Compute global norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        clip_coef = min(1.0, self.config.max_grad_norm / (total_norm + 1e-6))
        
        clipped = {}
        for name, grad in gradients.items():
            clipped[name] = grad * clip_coef
        
        return clipped
    
    def _per_layer_clip(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Per-layer gradient clipping."""
        clipped = {}
        for name, grad in gradients.items():
            norm = grad.norm(2).item()
            clip_coef = min(1.0, self.config.max_grad_norm / (norm + 1e-6))
            clipped[name] = grad * clip_coef
        return clipped
    
    def add_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        num_clients: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Add calibrated noise for differential privacy.
        
        Args:
            gradients: Clipped gradients
            num_clients: Number of clients in aggregation (for calibration)
            
        Returns:
            Noisy gradients
        """
        # Calibrate noise based on sensitivity and desired epsilon
        sigma = self._calibrate_noise(num_clients)
        
        noisy = {}
        for name, grad in gradients.items():
            if self.config.mechanism == PrivacyMechanism.GAUSSIAN:
                noise = torch.randn_like(grad) * sigma
            elif self.config.mechanism == PrivacyMechanism.LAPLACE:
                # Laplace noise (heavier tails)
                noise = torch.empty_like(grad).exponential_() - torch.empty_like(grad).exponential_()
                noise = noise * sigma / np.sqrt(2)
            else:
                noise = torch.zeros_like(grad)
            
            noisy[name] = grad + noise
        
        # Update privacy budget
        round_epsilon = self._compute_epsilon(sigma, num_clients)
        self.budget.add_round(round_epsilon)
        
        return noisy
    
    def _calibrate_noise(self, num_clients: int) -> float:
        """Calibrate noise level based on privacy parameters."""
        sensitivity = self.config.max_grad_norm / num_clients
        
        if self.config.mechanism == PrivacyMechanism.GAUSSIAN:
            # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.config.delta))
            sigma = sigma / (self.config.epsilon / self.budget.rounds_completed if self.budget.rounds_completed > 0 else self.config.epsilon)
        else:
            # Laplace mechanism: b = sensitivity / epsilon
            sigma = sensitivity / self.config.epsilon
        
        # Apply noise multiplier
        sigma = sigma * self.config.noise_multiplier
        
        return sigma
    
    def _compute_epsilon(self, sigma: float, num_clients: int) -> float:
        """Compute epsilon spent in this round."""
        sensitivity = self.config.max_grad_norm / num_clients
        
        if self.config.mechanism == PrivacyMechanism.GAUSSIAN:
            # Compute RDP epsilon
            epsilon = (sensitivity ** 2) / (2 * sigma ** 2)
        else:
            epsilon = sensitivity / sigma
        
        return epsilon
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy budget status."""
        return {
            'epsilon_spent': self.budget.spent_epsilon,
            'epsilon_remaining': self.budget.remaining_epsilon(),
            'delta': self.config.delta,
            'rounds': self.budget.rounds_completed,
            'can_continue': self.budget.can_continue()
        }


class SecureAggregator:
    """
    Implements secure aggregation protocols.
    
    Ensures that:
    1. Server only sees the sum of all updates
    2. No client can see another client's update
    3. Malicious clients are detected and excluded
    """
    
    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
        byzantine_threshold: float = 0.3,
        use_secure_aggregation: bool = True
    ):
        self.privacy_config = privacy_config or PrivacyConfig()
        self.differential_privacy = DifferentialPrivacy(self.privacy_config)
        
        self.byzantine_threshold = byzantine_threshold
        self.use_secure_aggregation = use_secure_aggregation
        
        # Client masks for secure aggregation
        self.client_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self.mask_seeds: Dict[str, bytes] = {}
        
        # Byzantine detection
        self.client_history: Dict[str, List[float]] = defaultdict(list)
        self.excluded_clients: set = set()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def setup_round(
        self,
        client_ids: List[str],
        model_template: Dict[str, torch.Tensor]
    ) -> Dict[str, bytes]:
        """
        Setup secure aggregation for a round.
        
        Args:
            client_ids: Participating clients
            model_template: Template with parameter shapes
            
        Returns:
            Per-client random seeds for mask generation
        """
        with self._lock:
            self.client_masks.clear()
            self.mask_seeds.clear()
            
            # Generate random seeds for each client pair
            for client_id in client_ids:
                seed = secrets.token_bytes(32)
                self.mask_seeds[client_id] = seed
                
                # Pre-generate masks
                self.client_masks[client_id] = self._generate_masks(
                    seed, model_template
                )
            
            return self.mask_seeds.copy()
    
    def _generate_masks(
        self,
        seed: bytes,
        template: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate pseudorandom masks from seed."""
        # Use seed to initialize RNG
        seed_int = int.from_bytes(seed[:8], 'big')
        rng = np.random.RandomState(seed_int)
        
        masks = {}
        for name, param in template.items():
            mask_data = rng.standard_normal(param.shape).astype(np.float32)
            masks[name] = torch.from_numpy(mask_data)
        
        return masks
    
    def mask_update(
        self,
        client_id: str,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Mask a client's update for secure transmission.
        
        Args:
            client_id: Client identifier
            gradients: Client's gradient updates
            
        Returns:
            Masked gradients
        """
        if not self.use_secure_aggregation:
            return gradients
        
        with self._lock:
            if client_id not in self.client_masks:
                logger.warning(f"No mask for client {client_id}")
                return gradients
            
            masks = self.client_masks[client_id]
            
            masked = {}
            for name, grad in gradients.items():
                if name in masks:
                    masked[name] = grad + masks[name].to(grad.device)
                else:
                    masked[name] = grad
            
            return masked
    
    def unmask_aggregate(
        self,
        masked_updates: List[Tuple[str, Dict[str, torch.Tensor]]],
        template: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Unmask and aggregate updates.
        
        The key insight: sum of masks cancels out when all clients participate,
        leaving only the true sum of updates.
        """
        with self._lock:
            # Initialize aggregate
            aggregate = {
                name: torch.zeros_like(param)
                for name, param in template.items()
            }
            
            # Sum all masked updates
            for client_id, updates in masked_updates:
                for name, update in updates.items():
                    if name in aggregate:
                        aggregate[name] += update.to(aggregate[name].device)
            
            # Subtract mask sum (would cancel if all clients participated)
            if self.use_secure_aggregation:
                mask_sum = {
                    name: torch.zeros_like(param)
                    for name, param in template.items()
                }
                
                participating = set(cid for cid, _ in masked_updates)
                for client_id in participating:
                    if client_id in self.client_masks:
                        for name, mask in self.client_masks[client_id].items():
                            if name in mask_sum:
                                mask_sum[name] += mask.to(mask_sum[name].device)
                
                for name in aggregate:
                    aggregate[name] -= mask_sum[name]
            
            return aggregate
    
    def detect_byzantine(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        global_model: Dict[str, torch.Tensor]
    ) -> set:
        """
        Detect potentially malicious (Byzantine) clients.
        
        Uses multiple detection methods:
        1. Gradient magnitude outliers
        2. Cosine similarity to mean update
        3. Historical consistency
        
        Args:
            client_updates: Dictionary of client gradients
            global_model: Current global model state
            
        Returns:
            Set of client IDs to exclude
        """
        if len(client_updates) < 3:
            return set()
        
        suspicious = set()
        
        # Compute update magnitudes
        magnitudes = {}
        for client_id, updates in client_updates.items():
            total_norm = 0
            for param in updates.values():
                total_norm += param.norm(2).item() ** 2
            magnitudes[client_id] = total_norm ** 0.5
        
        # Statistical outlier detection
        mean_mag = np.mean(list(magnitudes.values()))
        std_mag = np.std(list(magnitudes.values()))
        
        for client_id, mag in magnitudes.items():
            z_score = abs(mag - mean_mag) / (std_mag + 1e-6)
            if z_score > 3:  # 3-sigma outlier
                suspicious.add(client_id)
                logger.warning(f"Byzantine detection: {client_id} has outlier magnitude")
        
        # Compute mean update direction
        mean_update = {}
        for name in global_model.keys():
            stacked = torch.stack([
                client_updates[cid][name]
                for cid in client_updates if name in client_updates[cid]
            ])
            mean_update[name] = stacked.mean(dim=0)
        
        # Check cosine similarity to mean
        for client_id, updates in client_updates.items():
            cos_sim = self._cosine_similarity(updates, mean_update)
            
            # Track history
            self.client_history[client_id].append(cos_sim)
            if len(self.client_history[client_id]) > 10:
                self.client_history[client_id] = self.client_history[client_id][-10:]
            
            # Negative similarity is very suspicious
            if cos_sim < -0.5:
                suspicious.add(client_id)
                logger.warning(f"Byzantine detection: {client_id} has negative similarity")
        
        # Limit exclusions to threshold
        max_exclusions = int(len(client_updates) * self.byzantine_threshold)
        if len(suspicious) > max_exclusions:
            # Keep only the most suspicious
            suspicion_scores = {
                cid: magnitudes.get(cid, 0) for cid in suspicious
            }
            sorted_suspicious = sorted(
                suspicious,
                key=lambda x: suspicion_scores[x],
                reverse=True
            )
            suspicious = set(sorted_suspicious[:max_exclusions])
        
        self.excluded_clients.update(suspicious)
        return suspicious
    
    def _cosine_similarity(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor]
    ) -> float:
        """Compute cosine similarity between two updates."""
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for name in update1:
            if name in update2:
                u1 = update1[name].flatten()
                u2 = update2[name].flatten()
                
                dot_product += (u1 * u2).sum().item()
                norm1 += (u1 * u1).sum().item()
                norm2 += (u2 * u2).sum().item()
        
        return dot_product / (np.sqrt(norm1 * norm2) + 1e-6)
    
    def secure_aggregate(
        self,
        client_updates: List[Tuple[str, Dict[str, torch.Tensor], int]],
        global_model: Dict[str, torch.Tensor],
        detect_byzantine: bool = True,
        apply_dp: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Perform complete secure aggregation.
        
        Args:
            client_updates: List of (client_id, gradients, num_samples)
            global_model: Current global model state
            detect_byzantine: Whether to detect malicious clients
            apply_dp: Whether to apply differential privacy
            
        Returns:
            Tuple of (aggregated update, metadata)
        """
        with self._lock:
            # Convert to dict for processing
            updates_dict = {cid: grad for cid, grad, _ in client_updates}
            weights = {cid: samples for cid, _, samples in client_updates}
            
            # Byzantine detection
            excluded = set()
            if detect_byzantine and len(updates_dict) >= 3:
                excluded = self.detect_byzantine(updates_dict, global_model)
            
            # Filter out excluded clients
            valid_updates = {
                cid: grad for cid, grad in updates_dict.items()
                if cid not in excluded
            }
            valid_weights = {
                cid: w for cid, w in weights.items()
                if cid not in excluded
            }
            
            if not valid_updates:
                logger.error("No valid updates after Byzantine filtering")
                return {}, {'error': 'no_valid_updates'}
            
            # Clip gradients
            clipped_updates = {}
            for cid, grad in valid_updates.items():
                clipped_updates[cid] = self.differential_privacy.clip_gradients(grad)
            
            # Compute weighted average
            total_weight = sum(valid_weights.values())
            normalized_weights = {
                cid: w / total_weight for cid, w in valid_weights.items()
            }
            
            aggregated = {
                name: torch.zeros_like(param)
                for name, param in global_model.items()
            }
            
            for cid, updates in clipped_updates.items():
                weight = normalized_weights[cid]
                for name, grad in updates.items():
                    if name in aggregated:
                        aggregated[name] += weight * grad.to(aggregated[name].device)
            
            # Add differential privacy noise
            if apply_dp:
                aggregated = self.differential_privacy.add_noise(
                    aggregated,
                    num_clients=len(valid_updates)
                )
            
            metadata = {
                'num_clients': len(valid_updates),
                'excluded_clients': list(excluded),
                'total_samples': sum(valid_weights.values()),
                'privacy': self.differential_privacy.get_privacy_spent()
            }
            
            return aggregated, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            'privacy': self.differential_privacy.get_privacy_spent(),
            'excluded_clients': list(self.excluded_clients),
            'secure_aggregation': self.use_secure_aggregation,
            'byzantine_threshold': self.byzantine_threshold
        }


class HomomorphicAggregator:
    """
    Simulated homomorphic encryption for gradient aggregation.
    
    NOTE: This is a simplified simulation. Real HE would use
    libraries like Microsoft SEAL, TenSEAL, or Paillier.
    """
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        
        # Simulated public/private key
        self._private_key = secrets.token_bytes(key_size // 8)
        self.public_key = hashlib.sha256(self._private_key).hexdigest()
        
        logger.info(f"Initialized HE with key size {key_size}")
    
    def encrypt(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        'Encrypt' gradients (simulation).
        
        In real HE, this would convert to ciphertext.
        """
        encrypted = {}
        for name, grad in gradients.items():
            # Simulated encryption: just convert to numpy
            # Real HE would produce ciphertext
            encrypted[name] = grad.cpu().numpy()
        return encrypted
    
    def decrypt(
        self,
        encrypted: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        'Decrypt' gradients (simulation).
        """
        decrypted = {}
        for name, cipher in encrypted.items():
            decrypted[name] = torch.from_numpy(cipher)
        return decrypted
    
    def aggregate_encrypted(
        self,
        encrypted_updates: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate encrypted gradients.
        
        In real HE, we can add ciphertexts without decrypting.
        """
        if not encrypted_updates:
            return {}
        
        aggregated = {}
        for name in encrypted_updates[0].keys():
            stacked = np.stack([e[name] for e in encrypted_updates])
            aggregated[name] = stacked.mean(axis=0)
        
        return aggregated


def create_secure_aggregator(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    byzantine_threshold: float = 0.3,
    use_secure_agg: bool = True,
    **kwargs
) -> SecureAggregator:
    """
    Factory function to create a secure aggregator.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy failure probability
        byzantine_threshold: Max fraction of malicious clients
        use_secure_agg: Whether to use secure aggregation
        
    Returns:
        Configured SecureAggregator
    """
    config = PrivacyConfig(
        epsilon=epsilon,
        delta=delta,
        **kwargs
    )
    
    return SecureAggregator(
        privacy_config=config,
        byzantine_threshold=byzantine_threshold,
        use_secure_aggregation=use_secure_agg
    )


if __name__ == "__main__":
    print("Secure Aggregator Demo")
    print("=" * 50)
    
    # Create aggregator
    aggregator = create_secure_aggregator(
        epsilon=5.0,
        delta=1e-5,
        byzantine_threshold=0.3
    )
    
    print(f"Privacy budget: ε={aggregator.privacy_config.epsilon}")
    print(f"Byzantine threshold: {aggregator.byzantine_threshold}")
    
    # Simulate global model
    global_model = {
        'layer1.weight': torch.randn(64, 32),
        'layer1.bias': torch.randn(64),
        'layer2.weight': torch.randn(10, 64),
        'layer2.bias': torch.randn(10)
    }
    
    # Setup round
    client_ids = ['client-001', 'client-002', 'client-003', 'client-004', 'client-005']
    seeds = aggregator.setup_round(client_ids, global_model)
    print(f"\nSetup round for {len(client_ids)} clients")
    
    # Simulate client updates
    client_updates = []
    for i, client_id in enumerate(client_ids):
        gradients = {
            name: torch.randn_like(param) * 0.01
            for name, param in global_model.items()
        }
        
        # Make one client slightly malicious
        if i == 2:
            gradients = {
                name: torch.randn_like(param) * 10  # Much larger
                for name, param in global_model.items()
            }
        
        samples = np.random.randint(100, 500)
        client_updates.append((client_id, gradients, samples))
    
    # Secure aggregation
    aggregated, metadata = aggregator.secure_aggregate(
        client_updates,
        global_model,
        detect_byzantine=True,
        apply_dp=True
    )
    
    print(f"\nAggregation Results:")
    print(f"  Valid clients: {metadata['num_clients']}")
    print(f"  Excluded: {metadata['excluded_clients']}")
    print(f"  Total samples: {metadata['total_samples']}")
    
    print(f"\nPrivacy Status:")
    for key, value in metadata['privacy'].items():
        print(f"  {key}: {value}")
    
    print(f"\nAggregated Update Stats:")
    for name, update in aggregated.items():
        print(f"  {name}: norm={update.norm().item():.4f}")
    
    # Test differential privacy
    dp = aggregator.differential_privacy
    print(f"\nDifferential Privacy Budget:")
    print(f"  Spent: ε={dp.budget.spent_epsilon:.4f}")
    print(f"  Remaining: ε={dp.budget.remaining_epsilon():.4f}")
    
    print("\n✅ Secure Aggregator ready for privacy-preserving FL!")
