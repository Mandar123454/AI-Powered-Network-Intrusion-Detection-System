"""
Adversarial Trainer - GAN-Based Evasion Resistance
====================================================
The ultimate countermeasure: teach the model to defeat itself.

Architecture:
1. Attacker GAN (Generator): Creates synthetic evasion flows
   - Input: Normal traffic + attack intent
   - Output: Traffic that bypasses current detector
   
2. Defender Discriminator: Learns to catch everything
   - Input: Traffic flow (real or synthetic)
   - Output: Attack probability
   
This is AlphaZero-style self-play for security.
The attacker and defender evolve together, forever.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import logging
import time
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training."""
    # Architecture
    input_dim: int = 78
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    num_classes: int = 10
    
    # Training
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0001
    batch_size: int = 64
    
    # GAN training dynamics
    generator_steps: int = 1
    discriminator_steps: int = 1
    label_smoothing: float = 0.1
    
    # Adversarial strategies
    attack_types: List[str] = field(default_factory=lambda: [
        'evasion', 'mimicry', 'perturbation', 'polymorphic'
    ])
    
    # Robustness
    epsilon: float = 0.3  # Maximum perturbation
    alpha: float = 0.01   # Step size for PGD
    pgd_steps: int = 10   # Steps for PGD attack


class AttackerGAN(nn.Module):
    """
    The Attacker: Generates adversarial traffic that evades detection.
    
    This is the "Red Team" neural network. It learns to:
    1. Take normal traffic patterns
    2. Inject attack payloads
    3. Output traffic that looks normal but is malicious
    """
    
    def __init__(
        self,
        input_dim: int = 78,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
        num_attack_types: int = 10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_attack_types = num_attack_types
        
        # Encoder: Compress input traffic
        encoder_layers = []
        prev_dim = input_dim + num_attack_types  # Input + attack type embedding
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent transformation
        self.to_latent = nn.Linear(prev_dim, latent_dim * 2)  # Mean and log-var
        
        # Decoder: Generate adversarial traffic
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output layer with constraints
        self.output = nn.Linear(prev_dim, input_dim)
        
        # Attack type embedding
        self.attack_embedding = nn.Embedding(num_attack_types, num_attack_types)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(
        self,
        x: torch.Tensor,
        attack_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution."""
        attack_emb = self.attack_embedding(attack_type)
        combined = torch.cat([x, attack_emb], dim=-1)
        
        h = self.encoder(combined)
        latent = self.to_latent(h)
        
        mean, log_var = latent.chunk(2, dim=-1)
        return mean, log_var
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for VAE-style training."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to adversarial traffic."""
        h = self.decoder(z)
        return self.output(h)
    
    def forward(
        self,
        x: torch.Tensor,
        attack_type: torch.Tensor,
        epsilon: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial traffic.
        
        Args:
            x: Original traffic features
            attack_type: Type of attack to inject
            epsilon: Maximum perturbation
            
        Returns:
            Dictionary with adversarial traffic and latent info
        """
        mean, log_var = self.encode(x, attack_type)
        z = self.reparameterize(mean, log_var)
        
        # Generate perturbation
        perturbation = self.decode(z)
        
        # Clip perturbation to epsilon ball
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        
        # Apply perturbation
        adversarial = x + perturbation
        
        return {
            'adversarial': adversarial,
            'perturbation': perturbation,
            'mean': mean,
            'log_var': log_var,
            'latent': z
        }
    
    def generate_evasion(
        self,
        x: torch.Tensor,
        attack_type: int,
        epsilon: float = 0.3
    ) -> torch.Tensor:
        """Generate evasion attack for given traffic."""
        attack_tensor = torch.full(
            (x.size(0),), attack_type,
            dtype=torch.long, device=x.device
        )
        output = self.forward(x, attack_tensor, epsilon)
        return output['adversarial']


class DefenderDiscriminator(nn.Module):
    """
    The Defender: Detects both real attacks and synthetic evasions.
    
    This is the "Blue Team" neural network. It learns to:
    1. Detect known attack patterns
    2. Recognize GAN-generated evasions
    3. Generalize to novel attack variants
    """
    
    def __init__(
        self,
        input_dim: int = 78,
        hidden_dims: List[int] = [128, 64, 32],
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Feature extractor with spectral normalization
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.utils.spectral_norm(nn.Linear(prev_dim, hidden_dim)),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        # Real/Fake head (for GAN training)
        self.real_fake_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
        # Attack classification head
        self.attack_head = nn.Linear(prev_dim, num_classes)
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Classify traffic.
        
        Args:
            x: Traffic features
            return_all: Return all outputs
            
        Returns:
            Dictionary with classification results
        """
        features = self.features(x)
        
        real_fake = self.real_fake_head(features)
        attack_logits = self.attack_head(features)
        anomaly = self.anomaly_head(features)
        
        result = {
            'real_prob': real_fake,
            'attack_logits': attack_logits,
            'attack_probs': F.softmax(attack_logits, dim=-1),
            'anomaly_score': anomaly
        }
        
        if return_all:
            result['features'] = features
        
        return result
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get attack predictions."""
        output = self.forward(x)
        return output['attack_logits'].argmax(dim=-1)
    
    def get_threat_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get combined threat score."""
        output = self.forward(x)
        
        # Combine: max attack probability * (1 - real probability) * anomaly
        max_attack_prob = output['attack_probs'].max(dim=-1)[0]
        fake_prob = 1 - output['real_prob'].squeeze(-1)
        anomaly = output['anomaly_score'].squeeze(-1)
        
        # Threat is high if it looks like an attack AND looks synthetic
        threat = max_attack_prob * (0.5 + 0.5 * fake_prob) * (0.5 + 0.5 * anomaly)
        return threat


class PGDAttacker:
    """
    Projected Gradient Descent attacker.
    Generates worst-case adversarial examples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        steps: int = 10
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False
    ) -> torch.Tensor:
        """
        Generate PGD adversarial examples.
        
        Args:
            x: Original inputs
            y: Target labels (true for untargeted, desired for targeted)
            targeted: Whether to do targeted attack
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(self.steps):
            self.model.zero_grad()
            
            output = self.model(x_adv)
            logits = output['attack_logits']
            
            loss = F.cross_entropy(logits, y)
            
            if targeted:
                loss = -loss  # Minimize loss for targeted attack
            
            loss.backward()
            
            # Update adversarial example
            grad = x_adv.grad.sign()
            x_adv = x_adv.detach() + self.alpha * grad
            
            # Project to epsilon ball
            perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = (x + perturbation).detach().requires_grad_(True)
        
        return x_adv.detach()


class AdversarialTrainer:
    """
    Complete adversarial training system.
    
    Implements:
    1. GAN training (attacker vs defender)
    2. PGD adversarial training
    3. Curriculum learning (easy to hard attacks)
    4. Experience replay for stability
    """
    
    def __init__(
        self,
        config: Optional[AdversarialConfig] = None,
        device: str = 'cpu'
    ):
        self.config = config or AdversarialConfig()
        self.device = device
        
        # Create models
        self.attacker = AttackerGAN(
            input_dim=self.config.input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
            num_attack_types=self.config.num_classes
        ).to(device)
        
        self.defender = DefenderDiscriminator(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=self.config.num_classes
        ).to(device)
        
        # Optimizers
        self.attacker_optimizer = torch.optim.Adam(
            self.attacker.parameters(),
            lr=self.config.generator_lr,
            betas=(0.5, 0.999)
        )
        
        self.defender_optimizer = torch.optim.Adam(
            self.defender.parameters(),
            lr=self.config.discriminator_lr,
            betas=(0.5, 0.999)
        )
        
        # PGD attacker for additional robustness
        self.pgd_attacker = PGDAttacker(
            self.defender,
            epsilon=self.config.epsilon,
            alpha=self.config.alpha,
            steps=self.config.pgd_steps
        )
        
        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=10000)
        
        # Training history
        self.history = {
            'attacker_loss': [],
            'defender_loss': [],
            'defender_acc': [],
            'evasion_rate': []
        }
        
        # Curriculum state
        self.current_difficulty = 0.1
        self.max_difficulty = 1.0
    
    def train_step(
        self,
        real_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            real_data: Real traffic samples
            labels: Attack labels
            
        Returns:
            Training metrics
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        labels = labels.to(self.device)
        
        # Train Discriminator/Defender
        defender_loss, defender_acc = self._train_defender(
            real_data, labels, batch_size
        )
        
        # Train Generator/Attacker
        attacker_loss, evasion_rate = self._train_attacker(
            real_data, labels, batch_size
        )
        
        # Store in replay buffer
        for i in range(batch_size):
            self.replay_buffer.append((
                real_data[i].cpu().numpy(),
                labels[i].item()
            ))
        
        metrics = {
            'attacker_loss': attacker_loss,
            'defender_loss': defender_loss,
            'defender_acc': defender_acc,
            'evasion_rate': evasion_rate
        }
        
        for key, value in metrics.items():
            self.history[key].append(value)
        
        return metrics
    
    def _train_defender(
        self,
        real_data: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int
    ) -> Tuple[float, float]:
        """Train the defender/discriminator."""
        self.defender_optimizer.zero_grad()
        
        # Real samples
        real_output = self.defender(real_data)
        
        # Label smoothing
        real_labels = torch.full(
            (batch_size, 1), 1.0 - self.config.label_smoothing,
            device=self.device
        )
        
        # Real/fake loss
        real_loss = F.binary_cross_entropy(
            real_output['real_prob'], real_labels
        )
        
        # Classification loss
        class_loss = F.cross_entropy(real_output['attack_logits'], labels)
        
        # Generate fake samples
        attack_types = torch.randint(
            0, self.config.num_classes, (batch_size,),
            device=self.device
        )
        
        with torch.no_grad():
            fake_output = self.attacker(
                real_data, attack_types, self.config.epsilon
            )
            fake_data = fake_output['adversarial']
        
        # Fake samples
        fake_output = self.defender(fake_data)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        fake_loss = F.binary_cross_entropy(
            fake_output['real_prob'], fake_labels
        )
        
        # PGD adversarial training
        if np.random.random() < 0.5:  # 50% of time
            pgd_data = self.pgd_attacker.attack(real_data, labels)
            pgd_output = self.defender(pgd_data)
            pgd_loss = F.cross_entropy(pgd_output['attack_logits'], labels)
        else:
            pgd_loss = 0
        
        # Total loss
        total_loss = real_loss + fake_loss + class_loss
        if isinstance(pgd_loss, torch.Tensor):
            total_loss = total_loss + 0.5 * pgd_loss
        
        total_loss.backward()
        self.defender_optimizer.step()
        
        # Accuracy
        preds = real_output['attack_logits'].argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        
        return total_loss.item(), acc
    
    def _train_attacker(
        self,
        real_data: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int
    ) -> Tuple[float, float]:
        """Train the attacker/generator."""
        self.attacker_optimizer.zero_grad()
        
        # Generate adversarial samples
        attack_types = torch.randint(
            0, self.config.num_classes, (batch_size,),
            device=self.device
        )
        
        output = self.attacker(real_data, attack_types, self.config.epsilon)
        adversarial = output['adversarial']
        
        # Fool the defender (want high real_prob)
        defender_output = self.defender(adversarial)
        
        # Adversarial loss: want defender to think it's real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        adv_loss = F.binary_cross_entropy(
            defender_output['real_prob'], real_labels
        )
        
        # Attack success: want defender to misclassify
        wrong_labels = (labels + 1) % self.config.num_classes
        attack_loss = F.cross_entropy(
            defender_output['attack_logits'], wrong_labels
        )
        
        # VAE regularization
        kl_loss = -0.5 * torch.mean(
            1 + output['log_var'] - output['mean'].pow(2) - output['log_var'].exp()
        )
        
        # Perturbation penalty (want small perturbations)
        perturb_loss = output['perturbation'].abs().mean()
        
        total_loss = adv_loss + attack_loss + 0.01 * kl_loss + 0.1 * perturb_loss
        
        total_loss.backward()
        self.attacker_optimizer.step()
        
        # Evasion rate: how often defender is fooled
        preds = defender_output['attack_logits'].argmax(dim=-1)
        evasion_rate = (preds != attack_types).float().mean().item()
        
        return total_loss.item(), evasion_rate
    
    def train_epoch(
        self,
        data_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.attacker.train()
        self.defender.train()
        
        epoch_metrics = {
            'attacker_loss': 0,
            'defender_loss': 0,
            'defender_acc': 0,
            'evasion_rate': 0
        }
        num_batches = 0
        
        for batch_x, batch_y in data_loader:
            metrics = self.train_step(batch_x, batch_y)
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1
        
        # Average
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        if verbose:
            logger.info(
                f"Epoch: att_loss={epoch_metrics['attacker_loss']:.4f}, "
                f"def_loss={epoch_metrics['defender_loss']:.4f}, "
                f"def_acc={epoch_metrics['defender_acc']:.4f}, "
                f"evasion={epoch_metrics['evasion_rate']:.4f}"
            )
        
        # Update curriculum
        self._update_curriculum(epoch_metrics)
        
        return epoch_metrics
    
    def _update_curriculum(self, metrics: Dict[str, float]):
        """Update training difficulty based on performance."""
        if metrics['defender_acc'] > 0.9 and metrics['evasion_rate'] < 0.3:
            # Defender is winning, increase difficulty
            self.current_difficulty = min(
                self.max_difficulty,
                self.current_difficulty + 0.1
            )
            self.config.epsilon = 0.1 + 0.2 * self.current_difficulty
            
            logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
    
    def generate_adversarial_batch(
        self,
        x: torch.Tensor,
        attack_type: int
    ) -> torch.Tensor:
        """Generate batch of adversarial examples."""
        self.attacker.eval()
        with torch.no_grad():
            return self.attacker.generate_evasion(
                x.to(self.device),
                attack_type,
                self.config.epsilon
            )
    
    def evaluate_robustness(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model robustness against attacks."""
        self.defender.eval()
        self.attacker.eval()
        
        clean_correct = 0
        adversarial_correct = 0
        pgd_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Clean accuracy
                clean_output = self.defender(batch_x)
                clean_preds = clean_output['attack_logits'].argmax(dim=-1)
                clean_correct += (clean_preds == batch_y).sum().item()
                
                # GAN adversarial accuracy
                adv_x = self.attacker.generate_evasion(
                    batch_x, 0, self.config.epsilon
                )
                adv_output = self.defender(adv_x)
                adv_preds = adv_output['attack_logits'].argmax(dim=-1)
                adversarial_correct += (adv_preds == batch_y).sum().item()
                
                total += len(batch_y)
        
        # PGD accuracy (more expensive, sample)
        sample_x, sample_y = next(iter(data_loader))
        sample_x = sample_x.to(self.device)
        sample_y = sample_y.to(self.device)
        
        self.defender.train()  # Need gradients for PGD
        pgd_x = self.pgd_attacker.attack(sample_x, sample_y)
        self.defender.eval()
        
        with torch.no_grad():
            pgd_output = self.defender(pgd_x)
            pgd_preds = pgd_output['attack_logits'].argmax(dim=-1)
            pgd_acc = (pgd_preds == sample_y).float().mean().item()
        
        return {
            'clean_accuracy': clean_correct / max(total, 1),
            'adversarial_accuracy': adversarial_correct / max(total, 1),
            'pgd_accuracy': pgd_acc,
            'robustness_gap': (clean_correct - adversarial_correct) / max(total, 1)
        }
    
    def get_defender_model(self) -> DefenderDiscriminator:
        """Get trained defender for deployment."""
        return copy.deepcopy(self.defender)
    
    def save(self, path: str):
        """Save trainer state."""
        torch.save({
            'attacker': self.attacker.state_dict(),
            'defender': self.defender.state_dict(),
            'config': self.config.__dict__,
            'history': self.history,
            'difficulty': self.current_difficulty
        }, path)
        logger.info(f"Saved adversarial trainer to {path}")
    
    def load(self, path: str):
        """Load trainer state."""
        state = torch.load(path, map_location=self.device)
        self.attacker.load_state_dict(state['attacker'])
        self.defender.load_state_dict(state['defender'])
        self.history = state['history']
        self.current_difficulty = state['difficulty']
        logger.info(f"Loaded adversarial trainer from {path}")


def create_adversarial_trainer(
    input_dim: int = 78,
    num_classes: int = 10,
    device: str = 'cpu',
    **kwargs
) -> AdversarialTrainer:
    """
    Factory function to create adversarial trainer.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of attack classes
        device: Device to run on
        **kwargs: Additional AdversarialConfig parameters
        
    Returns:
        Configured AdversarialTrainer
    """
    config = AdversarialConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        **kwargs
    )
    
    return AdversarialTrainer(config, device)


if __name__ == "__main__":
    print("Adversarial Trainer Demo")
    print("=" * 50)
    
    # Create trainer
    trainer = create_adversarial_trainer(
        input_dim=78,
        num_classes=10
    )
    
    print(f"Attacker parameters: {sum(p.numel() for p in trainer.attacker.parameters()):,}")
    print(f"Defender parameters: {sum(p.numel() for p in trainer.defender.parameters()):,}")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 78).astype(np.float32)
    y = np.random.randint(0, 10, 1000).astype(np.int64)
    
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Train for a few epochs
    print("\nTraining adversarial system...")
    for epoch in range(3):
        metrics = trainer.train_epoch(data_loader, verbose=True)
    
    # Evaluate robustness
    print("\nEvaluating robustness...")
    robustness = trainer.evaluate_robustness(data_loader)
    
    print("\nRobustness Metrics:")
    for key, value in robustness.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate adversarial examples
    print("\nGenerating adversarial examples...")
    sample = torch.randn(5, 78)
    adversarial = trainer.generate_adversarial_batch(sample, attack_type=1)
    
    print(f"Original shape: {sample.shape}")
    print(f"Adversarial shape: {adversarial.shape}")
    print(f"Max perturbation: {(adversarial.cpu() - sample).abs().max():.4f}")
    
    # Get threat scores
    defender = trainer.defender.eval()
    with torch.no_grad():
        threat_scores = defender.get_threat_score(adversarial)
        print(f"Threat scores: {threat_scores.numpy()}")
    
    print("\n✅ Adversarial Trainer ready for GAN-based robustness training!")
