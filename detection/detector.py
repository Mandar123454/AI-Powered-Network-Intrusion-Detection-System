"""
AI-NIDS Detection Engine
Core detection logic integrating ML models with real-time analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


@dataclass
class DetectionResult:
    """Represents a single detection result."""
    is_attack: bool
    attack_type: str
    confidence: float
    severity: ThreatSeverity
    model_used: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    raw_features: Optional[Dict] = None
    shap_explanation: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_attack': self.is_attack,
            'attack_type': self.attack_type,
            'confidence': self.confidence,
            'severity': self.severity.name,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat(),
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol,
            'shap_explanation': self.shap_explanation,
            'metadata': self.metadata
        }


class DetectionEngine:
    """
    Main detection engine for AI-NIDS.
    Orchestrates ML models and provides real-time detection.
    """
    
    # Attack type to severity mapping
    SEVERITY_MAP = {
        'DDoS': ThreatSeverity.CRITICAL,
        'DoS': ThreatSeverity.HIGH,
        'Bot': ThreatSeverity.CRITICAL,
        'Infiltration': ThreatSeverity.CRITICAL,
        'Brute Force': ThreatSeverity.HIGH,
        'SSH-Patator': ThreatSeverity.HIGH,
        'FTP-Patator': ThreatSeverity.HIGH,
        'Web Attack': ThreatSeverity.HIGH,
        'XSS': ThreatSeverity.MEDIUM,
        'SQL Injection': ThreatSeverity.CRITICAL,
        'Heartbleed': ThreatSeverity.CRITICAL,
        'PortScan': ThreatSeverity.LOW,
        'Reconnaissance': ThreatSeverity.LOW,
        'Exploits': ThreatSeverity.CRITICAL,
        'Fuzzers': ThreatSeverity.MEDIUM,
        'Generic': ThreatSeverity.MEDIUM,
        'Shellcode': ThreatSeverity.CRITICAL,
        'Worms': ThreatSeverity.CRITICAL,
        'Backdoor': ThreatSeverity.CRITICAL,
        'Analysis': ThreatSeverity.LOW,
        'Unknown': ThreatSeverity.MEDIUM,
        'Benign': ThreatSeverity.INFO,
        'Normal': ThreatSeverity.INFO
    }
    
    def __init__(
        self,
        preprocessor=None,
        ensemble_model=None,
        xgboost_model=None,
        autoencoder_model=None,
        lstm_model=None,
        explainer=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize detection engine.
        
        Args:
            preprocessor: Data preprocessor instance
            ensemble_model: Ensemble detector instance
            xgboost_model: XGBoost classifier instance
            autoencoder_model: Autoencoder instance
            lstm_model: LSTM detector instance
            explainer: SHAP explainer instance
            config: Configuration dictionary
        """
        self.preprocessor = preprocessor
        self.ensemble_model = ensemble_model
        self.xgboost_model = xgboost_model
        self.autoencoder_model = autoencoder_model
        self.lstm_model = lstm_model
        self.explainer = explainer
        self.config = config or {}
        
        # Detection settings
        self.detection_threshold = self.config.get('detection_threshold', 0.5)
        self.enable_explanation = self.config.get('enable_explanation', True)
        self.batch_size = self.config.get('batch_size', 100)
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'attacks_detected': 0,
            'normal_traffic': 0,
            'by_attack_type': {},
            'by_severity': {s.name: 0 for s in ThreatSeverity}
        }
        
        # Detection queue for async processing
        self.detection_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        logger.info("Detection engine initialized")
    
    def load_models(
        self,
        preprocessor_path: str,
        xgboost_path: Optional[str] = None,
        autoencoder_path: Optional[str] = None,
        lstm_path: Optional[str] = None,
        ensemble_path: Optional[str] = None
    ) -> None:
        """
        Load models from disk.
        
        Args:
            preprocessor_path: Path to preprocessor
            xgboost_path: Path to XGBoost model
            autoencoder_path: Path to autoencoder model
            lstm_path: Path to LSTM model
            ensemble_path: Path to ensemble config
        """
        from ml.preprocessing import DataPreprocessor
        from ml.models import XGBoostClassifier, AnomalyAutoencoder, LSTMDetector, EnsembleDetector
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor.load(preprocessor_path)
        
        # Load individual models
        if xgboost_path:
            self.xgboost_model = XGBoostClassifier.load(xgboost_path)
            logger.info("Loaded XGBoost model")
        
        if autoencoder_path:
            self.autoencoder_model = AnomalyAutoencoder.load(autoencoder_path)
            logger.info("Loaded Autoencoder model")
        
        if lstm_path:
            self.lstm_model = LSTMDetector.load(lstm_path)
            logger.info("Loaded LSTM model")
        
        # Load or create ensemble
        if ensemble_path:
            models = {}
            if self.xgboost_model:
                models['xgboost'] = self.xgboost_model
            if self.autoencoder_model:
                models['autoencoder'] = self.autoencoder_model
            if self.lstm_model:
                models['lstm'] = self.lstm_model
            
            self.ensemble_model = EnsembleDetector.load(ensemble_path, models=models)
        elif self.xgboost_model or self.autoencoder_model or self.lstm_model:
            from ml.models import create_ensemble
            self.ensemble_model = create_ensemble(
                xgboost_model=self.xgboost_model,
                autoencoder_model=self.autoencoder_model,
                lstm_model=self.lstm_model
            )
        
        logger.info("All models loaded successfully")
    
    def detect(
        self,
        features: Union[np.ndarray, Dict, List[Dict]],
        metadata: Optional[Dict] = None
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Perform detection on input features.
        
        Args:
            features: Input features (array, dict, or list of dicts)
            metadata: Additional metadata
            
        Returns:
            Detection result(s)
        """
        # Convert input to array
        X, flow_metadata = self._prepare_input(features)
        
        # Single or batch detection
        if len(X) == 1:
            return self._detect_single(X[0], flow_metadata[0] if flow_metadata else metadata)
        else:
            return self._detect_batch(X, flow_metadata or [metadata] * len(X))
    
    def _prepare_input(
        self,
        features: Union[np.ndarray, Dict, List[Dict]]
    ) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Prepare input features for detection."""
        metadata = None
        
        if isinstance(features, dict):
            # Single flow as dictionary
            metadata = [self._extract_metadata(features)]
            X = self._dict_to_array([features])
            
        elif isinstance(features, list) and isinstance(features[0], dict):
            # List of flow dictionaries
            metadata = [self._extract_metadata(f) for f in features]
            X = self._dict_to_array(features)
            
        elif isinstance(features, np.ndarray):
            X = features if features.ndim == 2 else features.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported input type: {type(features)}")
        
        # Preprocess if we have a preprocessor
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        
        return X, metadata
    
    def _extract_metadata(self, flow: Dict) -> Dict:
        """Extract metadata from flow dictionary."""
        return {
            'source_ip': flow.get('Source IP', flow.get('src_ip')),
            'destination_ip': flow.get('Destination IP', flow.get('dst_ip')),
            'source_port': flow.get('Source Port', flow.get('src_port')),
            'destination_port': flow.get('Destination Port', flow.get('dst_port')),
            'protocol': flow.get('Protocol', flow.get('protocol'))
        }
    
    def _dict_to_array(self, flows: List[Dict]) -> np.ndarray:
        """Convert list of flow dictionaries to numpy array."""
        if self.preprocessor and self.preprocessor.feature_columns:
            feature_cols = self.preprocessor.feature_columns
        else:
            # Default feature columns
            feature_cols = list(flows[0].keys())
            # Remove non-numeric columns
            feature_cols = [c for c in feature_cols if c not in 
                          ['Source IP', 'Destination IP', 'Timestamp', 'Label', 'src_ip', 'dst_ip']]
        
        X = np.array([[flow.get(col, 0) for col in feature_cols] for flow in flows])
        return X
    
    def _detect_single(self, x: np.ndarray, metadata: Optional[Dict] = None) -> DetectionResult:
        """Detect on a single sample."""
        x = x.reshape(1, -1)
        
        # Get prediction from ensemble or best available model
        model_used = 'ensemble'
        
        if self.ensemble_model is not None:
            proba = self.ensemble_model.predict_proba(x)[0]
            prediction = self.ensemble_model.predict(x)[0]
        elif self.xgboost_model is not None:
            proba = self.xgboost_model.predict_proba(x)[0]
            prediction = self.xgboost_model.predict(x)[0]
            model_used = 'xgboost'
        elif self.autoencoder_model is not None:
            proba = self.autoencoder_model.predict_proba(x)
            prediction = self.autoencoder_model.predict(x)[0]
            model_used = 'autoencoder'
        else:
            raise ValueError("No model available for detection")
        
        # Determine if attack
        is_attack = prediction == 1 or (isinstance(proba, np.ndarray) and proba.ndim > 0 and proba[-1] > self.detection_threshold)
        
        # Get confidence
        if isinstance(proba, np.ndarray) and proba.ndim > 0:
            confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            confidence = float(proba)
        
        # Determine attack type (for multi-class models)
        attack_type = 'Unknown Attack' if is_attack else 'Normal'
        if hasattr(self.xgboost_model, 'label_encoder') and self.xgboost_model is not None:
            try:
                attack_type = self.xgboost_model.label_encoder.inverse_transform([prediction])[0]
            except:
                pass
        
        # Get severity
        severity = self._get_severity(attack_type, confidence)
        
        # Get SHAP explanation if enabled
        shap_explanation = None
        if is_attack and self.enable_explanation and self.explainer is not None:
            try:
                shap_explanation = self.explainer.explain_single(x)
                # Keep only top features
                shap_explanation = {
                    'top_contributors': shap_explanation.get('top_positive_contributors', [])[:5],
                    'base_value': shap_explanation.get('base_value'),
                    'contribution': shap_explanation.get('prediction_contribution')
                }
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # Update statistics
        self._update_stats(is_attack, attack_type, severity)
        
        # Build result
        result = DetectionResult(
            is_attack=is_attack,
            attack_type=attack_type,
            confidence=confidence,
            severity=severity,
            model_used=model_used,
            source_ip=metadata.get('source_ip') if metadata else None,
            destination_ip=metadata.get('destination_ip') if metadata else None,
            source_port=metadata.get('source_port') if metadata else None,
            destination_port=metadata.get('destination_port') if metadata else None,
            protocol=metadata.get('protocol') if metadata else None,
            shap_explanation=shap_explanation
        )
        
        return result
    
    def _detect_batch(self, X: np.ndarray, metadata: List[Dict]) -> List[DetectionResult]:
        """Detect on a batch of samples."""
        results = []
        
        for i, x in enumerate(X):
            meta = metadata[i] if i < len(metadata) else None
            result = self._detect_single(x, meta)
            results.append(result)
        
        return results
    
    def _get_severity(self, attack_type: str, confidence: float) -> ThreatSeverity:
        """Determine severity based on attack type and confidence."""
        base_severity = self.SEVERITY_MAP.get(attack_type, ThreatSeverity.MEDIUM)
        
        # Adjust based on confidence
        if confidence > 0.9:
            return base_severity
        elif confidence > 0.7:
            # One level lower
            severity_value = max(base_severity.value - 1, 1)
            return ThreatSeverity(severity_value)
        else:
            # Two levels lower
            severity_value = max(base_severity.value - 2, 1)
            return ThreatSeverity(severity_value)
    
    def _update_stats(self, is_attack: bool, attack_type: str, severity: ThreatSeverity) -> None:
        """Update detection statistics."""
        self.stats['total_detections'] += 1
        
        if is_attack:
            self.stats['attacks_detected'] += 1
            self.stats['by_attack_type'][attack_type] = self.stats['by_attack_type'].get(attack_type, 0) + 1
        else:
            self.stats['normal_traffic'] += 1
        
        self.stats['by_severity'][severity.name] += 1
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            **self.stats,
            'detection_rate': self.stats['attacks_detected'] / max(self.stats['total_detections'], 1)
        }
    
    def analyze_flow(self, flow: Dict) -> Dict:
        """
        Analyze a single network flow for intrusion detection.
        
        This is a simple wrapper for external API calls that returns
        a dictionary result even when no ML models are loaded.
        
        Args:
            flow: Dictionary containing flow data
            
        Returns:
            Dictionary with detection results
        """
        import random
        
        # Check if we have models loaded
        has_models = (
            self.ensemble_model is not None or
            self.xgboost_model is not None or
            self.autoencoder_model is not None or
            self.lstm_model is not None
        )
        
        if has_models:
            # Use the proper detect method
            try:
                result = self.detect(flow)
                return {
                    'is_threat': result.is_attack,
                    'attack_type': result.attack_type,
                    'severity': result.severity.name.lower() if hasattr(result.severity, 'name') else str(result.severity),
                    'confidence': result.confidence,
                    'description': f"Detected {result.attack_type} attack" if result.is_attack else "Normal traffic",
                    'model_used': result.model_used,
                    'source_ip': flow.get('src_ip'),
                    'destination_ip': flow.get('dst_ip')
                }
            except Exception as e:
                logger.warning(f"ML detection failed: {e}, using heuristic analysis")
        
        # Heuristic-based analysis when no models are available
        src_ip = flow.get('src_ip', '')
        dst_port = flow.get('dst_port', 0)
        protocol = flow.get('protocol', 'TCP')
        bytes_sent = flow.get('bytes_sent', 0)
        bytes_recv = flow.get('bytes_recv', 0)
        duration = flow.get('duration', 1)
        
        # Simple heuristic rules
        is_threat = False
        attack_type = 'Normal'
        severity = 'info'
        confidence = 0.95
        description = 'Normal network traffic'
        
        # Check for suspicious patterns
        suspicious_ports = [22, 23, 3389, 445, 1433, 3306, 5432]  # Common attack targets
        high_risk_ports = [4444, 5555, 6666, 31337]  # Known malware ports
        
        # Port scan detection (many different ports)
        if dst_port in high_risk_ports:
            is_threat = True
            attack_type = 'Suspicious Port Access'
            severity = 'high'
            confidence = 0.85
            description = f'Access to suspicious port {dst_port}'
        
        # High traffic anomaly
        elif bytes_sent > 10000000 or bytes_recv > 10000000:
            is_threat = True
            attack_type = 'Data Exfiltration'
            severity = 'high'
            confidence = 0.75
            description = 'Unusually high data transfer detected'
        
        # Brute force detection (many packets, low duration)
        elif duration < 1 and (bytes_sent > 10000 or bytes_recv > 10000):
            is_threat = True
            attack_type = 'Brute Force'
            severity = 'medium'
            confidence = 0.70
            description = 'Potential brute force attack pattern'
        
        # SSH/RDP on suspicious ports
        elif dst_port in [22, 3389] and bytes_sent > 5000:
            is_threat = True
            attack_type = 'Lateral Movement'
            severity = 'medium'
            confidence = 0.65
            description = 'Suspicious remote access activity'
        
        # If threat detected from external IP
        elif src_ip and not src_ip.startswith(('192.168.', '10.', '172.16.')):
            if dst_port in suspicious_ports:
                is_threat = random.random() > 0.7  # 30% chance to flag external access to sensitive ports
                if is_threat:
                    attack_type = 'Reconnaissance'
                    severity = 'low'
                    confidence = 0.60
                    description = f'External access attempt to port {dst_port}'
        
        return {
            'is_threat': is_threat,
            'attack_type': attack_type,
            'severity': severity,
            'confidence': confidence,
            'description': description,
            'model_used': 'heuristic',
            'source_ip': src_ip,
            'destination_ip': flow.get('dst_ip', '')
        }
    
    def analyze_batch(self, flows: List[Dict]) -> List[Dict]:
        """Analyze multiple flows."""
        return [self.analyze_flow(flow) for flow in flows]
    
    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self.stats = {
            'total_detections': 0,
            'attacks_detected': 0,
            'normal_traffic': 0,
            'by_attack_type': {},
            'by_severity': {s.name: 0 for s in ThreatSeverity}
        }
    
    # ===== Async Processing =====
    
    def start_async_detection(self) -> None:
        """Start asynchronous detection processing."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Started async detection worker")
    
    def stop_async_detection(self) -> None:
        """Stop asynchronous detection processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Stopped async detection worker")
    
    def _detection_worker(self) -> None:
        """Worker thread for async detection."""
        while self._running:
            try:
                # Get batch from queue
                batch = []
                while len(batch) < self.batch_size:
                    try:
                        item = self.detection_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break
                
                if batch:
                    # Process batch
                    features = [item['features'] for item in batch]
                    callbacks = [item.get('callback') for item in batch]
                    
                    results = self.detect(features)
                    
                    # Call callbacks
                    for result, callback in zip(results, callbacks):
                        if callback:
                            callback(result)
                            
            except Exception as e:
                logger.error(f"Detection worker error: {e}")
    
    def queue_detection(
        self,
        features: Union[np.ndarray, Dict],
        callback: Optional[callable] = None
    ) -> None:
        """
        Queue features for async detection.
        
        Args:
            features: Input features
            callback: Callback function for result
        """
        self.detection_queue.put({
            'features': features,
            'callback': callback
        })


def create_detection_engine(
    model_dir: str = 'models',
    config: Optional[Dict] = None
) -> DetectionEngine:
    """
    Factory function to create and configure DetectionEngine.
    
    Args:
        model_dir: Directory containing model files
        config: Configuration dictionary
        
    Returns:
        Configured DetectionEngine instance
    """
    import os
    
    engine = DetectionEngine(config=config)
    
    # Look for model files
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    xgboost_path = os.path.join(model_dir, 'xgboost_model.pkl')
    autoencoder_path = os.path.join(model_dir, 'autoencoder_model.pt')
    lstm_path = os.path.join(model_dir, 'lstm_model.pt')
    ensemble_path = os.path.join(model_dir, 'ensemble_config.json')
    
    # Load available models
    if os.path.exists(preprocessor_path):
        engine.load_models(
            preprocessor_path=preprocessor_path,
            xgboost_path=xgboost_path if os.path.exists(xgboost_path) else None,
            autoencoder_path=autoencoder_path if os.path.exists(autoencoder_path) else None,
            lstm_path=lstm_path if os.path.exists(lstm_path) else None,
            ensemble_path=ensemble_path if os.path.exists(ensemble_path) else None
        )
    else:
        logger.warning(f"Preprocessor not found at {preprocessor_path}. Engine initialized without models.")
    
    return engine
