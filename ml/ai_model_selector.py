"""
AI Model Selector for Defense Operations
==========================================
Intelligent selection of AI models based on attack type, severity, and context.
Provides reasoning for model selection to enhance transparency and trust.
"""

import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AIProvider(Enum):
    """AI Model Providers."""
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"
    META = "Meta"
    MISTRAL = "Mistral AI"
    COHERE = "Cohere"
    XAI = "xAI"
    DEEPMIND = "DeepMind"
    NVIDIA = "NVIDIA"
    HUGGINGFACE = "HuggingFace"
    LOCAL = "Local/Custom"


@dataclass
class AIModel:
    """Represents an AI model with its capabilities."""
    id: str
    name: str
    provider: AIProvider
    version: str
    specialties: List[str]
    speed_rating: float  # 1-10, higher is faster
    accuracy_rating: float  # 1-10, higher is more accurate
    cost_rating: float  # 1-10, higher is more expensive
    context_window: int  # tokens
    description: str
    icon: str
    color: str
    strengths: List[str]
    best_for: List[str]


# Comprehensive AI Model Registry
AI_MODELS: Dict[str, AIModel] = {
    # OpenAI Models
    "gpt-4o": AIModel(
        id="gpt-4o",
        name="GPT-4o",
        provider=AIProvider.OPENAI,
        version="2024",
        specialties=["reasoning", "code_analysis", "pattern_recognition", "multi_modal"],
        speed_rating=8.5,
        accuracy_rating=9.5,
        cost_rating=8.0,
        context_window=128000,
        description="OpenAI's most advanced multimodal model with excellent reasoning capabilities",
        icon="🧠",
        color="#10a37f",
        strengths=["Complex reasoning", "Code analysis", "Pattern detection", "Real-time analysis"],
        best_for=["SQL Injection", "Command Injection", "Complex multi-stage attacks"]
    ),
    "gpt-4-turbo": AIModel(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider=AIProvider.OPENAI,
        version="2024",
        specialties=["fast_inference", "code_analysis", "threat_detection"],
        speed_rating=9.0,
        accuracy_rating=9.0,
        cost_rating=7.0,
        context_window=128000,
        description="Fast and powerful model optimized for production workloads",
        icon="⚡",
        color="#10a37f",
        strengths=["Fast response", "Accurate detection", "Cost effective for high volume"],
        best_for=["Port Scan", "Brute Force", "Real-time threat detection"]
    ),
    "gpt-3.5-turbo": AIModel(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider=AIProvider.OPENAI,
        version="2024",
        specialties=["fast_inference", "basic_analysis", "high_throughput"],
        speed_rating=9.5,
        accuracy_rating=7.5,
        cost_rating=3.0,
        context_window=16385,
        description="Fast and cost-effective for high-volume basic threat analysis",
        icon="💨",
        color="#10a37f",
        strengths=["Ultra-fast", "Low cost", "High throughput"],
        best_for=["Low severity alerts", "Initial screening", "Volume processing"]
    ),
    "o1-preview": AIModel(
        id="o1-preview",
        name="o1 Preview",
        provider=AIProvider.OPENAI,
        version="2024",
        specialties=["deep_reasoning", "complex_analysis", "strategic_thinking"],
        speed_rating=5.0,
        accuracy_rating=10.0,
        cost_rating=10.0,
        context_window=128000,
        description="Advanced reasoning model for complex threat analysis and APT detection",
        icon="🔮",
        color="#10a37f",
        strengths=["Deep reasoning", "APT detection", "Zero-day analysis", "Strategic defense"],
        best_for=["APT", "Zero-day exploits", "Data Exfiltration", "Advanced persistent threats"]
    ),
    
    # Anthropic Models
    "claude-4-opus": AIModel(
        id="claude-4-opus",
        name="Claude 4 Opus",
        provider=AIProvider.ANTHROPIC,
        version="2025",
        specialties=["security_analysis", "code_review", "detailed_reasoning", "safety"],
        speed_rating=7.0,
        accuracy_rating=9.8,
        cost_rating=9.0,
        context_window=200000,
        description="Most capable Claude model with exceptional security analysis capabilities",
        icon="🛡️",
        color="#d4a574",
        strengths=["Security expertise", "Detailed analysis", "Safe responses", "Long context"],
        best_for=["Malware Communication", "Data Exfiltration", "Insider threats"]
    ),
    "claude-3.5-sonnet": AIModel(
        id="claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        provider=AIProvider.ANTHROPIC,
        version="2024",
        specialties=["balanced_performance", "security_analysis", "code_analysis"],
        speed_rating=8.5,
        accuracy_rating=9.2,
        cost_rating=6.0,
        context_window=200000,
        description="Balanced performance with excellent security analysis at reasonable cost",
        icon="🎵",
        color="#d4a574",
        strengths=["Balanced speed/accuracy", "Security focus", "Large context window"],
        best_for=["XSS", "Path Traversal", "Web application attacks"]
    ),
    "claude-3-haiku": AIModel(
        id="claude-3-haiku",
        name="Claude 3 Haiku",
        provider=AIProvider.ANTHROPIC,
        version="2024",
        specialties=["fast_inference", "real_time", "cost_effective"],
        speed_rating=9.8,
        accuracy_rating=8.0,
        cost_rating=2.0,
        context_window=200000,
        description="Ultra-fast model for real-time threat detection and high-volume processing",
        icon="🌸",
        color="#d4a574",
        strengths=["Fastest inference", "Very low cost", "Real-time capable"],
        best_for=["DDoS detection", "Real-time monitoring", "High-volume scanning"]
    ),
    
    # Google Models
    "gemini-2.5-pro": AIModel(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider=AIProvider.GOOGLE,
        version="2025",
        specialties=["multi_modal", "long_context", "reasoning", "code_analysis"],
        speed_rating=8.0,
        accuracy_rating=9.5,
        cost_rating=7.5,
        context_window=2000000,
        description="Google's most advanced model with massive context and multimodal capabilities",
        icon="💎",
        color="#4285f4",
        strengths=["Massive context", "Multimodal analysis", "Pattern recognition"],
        best_for=["Log analysis", "Network forensics", "Long-term pattern detection"]
    ),
    "gemini-2.0-flash": AIModel(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider=AIProvider.GOOGLE,
        version="2024",
        specialties=["fast_inference", "multi_modal", "real_time"],
        speed_rating=9.5,
        accuracy_rating=8.5,
        cost_rating=4.0,
        context_window=1000000,
        description="Fast multimodal model optimized for real-time security operations",
        icon="⚡",
        color="#4285f4",
        strengths=["Very fast", "Multimodal", "Cost effective"],
        best_for=["Real-time detection", "Image-based threats", "Quick analysis"]
    ),
    "gemini-1.5-pro": AIModel(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider=AIProvider.GOOGLE,
        version="2024",
        specialties=["long_context", "code_analysis", "reasoning"],
        speed_rating=7.5,
        accuracy_rating=9.0,
        cost_rating=6.0,
        context_window=2000000,
        description="Excellent for analyzing large codebases and extensive log files",
        icon="🔷",
        color="#4285f4",
        strengths=["Very long context", "Code understanding", "Pattern analysis"],
        best_for=["Large log analysis", "Codebase security review", "Historical analysis"]
    ),
    
    # Meta Models
    "llama-3.3-70b": AIModel(
        id="llama-3.3-70b",
        name="Llama 3.3 70B",
        provider=AIProvider.META,
        version="2024",
        specialties=["open_source", "customizable", "on_premise", "privacy"],
        speed_rating=7.0,
        accuracy_rating=8.5,
        cost_rating=2.0,
        context_window=128000,
        description="Powerful open-source model for on-premise security deployments",
        icon="🦙",
        color="#0668E1",
        strengths=["Open source", "On-premise deployment", "Customizable", "Privacy-focused"],
        best_for=["Sensitive environments", "Air-gapped networks", "Custom fine-tuning"]
    ),
    "llama-3.2-vision": AIModel(
        id="llama-3.2-vision",
        name="Llama 3.2 Vision",
        provider=AIProvider.META,
        version="2024",
        specialties=["multi_modal", "image_analysis", "open_source"],
        speed_rating=7.5,
        accuracy_rating=8.0,
        cost_rating=2.0,
        context_window=128000,
        description="Multimodal open-source model for visual threat detection",
        icon="👁️",
        color="#0668E1",
        strengths=["Image analysis", "Open source", "Visual threat detection"],
        best_for=["Phishing detection", "Visual malware", "Screenshot analysis"]
    ),
    
    # Mistral Models
    "mistral-large-2": AIModel(
        id="mistral-large-2",
        name="Mistral Large 2",
        provider=AIProvider.MISTRAL,
        version="2024",
        specialties=["european", "code_analysis", "reasoning", "multilingual"],
        speed_rating=8.0,
        accuracy_rating=9.0,
        cost_rating=5.0,
        context_window=128000,
        description="European AI model with strong code analysis and multilingual support",
        icon="🌊",
        color="#FF7000",
        strengths=["GDPR compliant", "Multilingual", "Strong reasoning", "Code focus"],
        best_for=["European compliance", "Multilingual threats", "Code vulnerabilities"]
    ),
    "codestral": AIModel(
        id="codestral",
        name="Codestral",
        provider=AIProvider.MISTRAL,
        version="2024",
        specialties=["code_analysis", "vulnerability_detection", "secure_coding"],
        speed_rating=9.0,
        accuracy_rating=9.0,
        cost_rating=4.0,
        context_window=32000,
        description="Specialized code model for vulnerability detection and secure coding",
        icon="💻",
        color="#FF7000",
        strengths=["Code expertise", "Vulnerability detection", "Fast analysis"],
        best_for=["SQL Injection", "Command Injection", "Code review"]
    ),
    
    # xAI Models
    "grok-2": AIModel(
        id="grok-2",
        name="Grok 2",
        provider=AIProvider.XAI,
        version="2024",
        specialties=["real_time", "internet_access", "reasoning"],
        speed_rating=8.5,
        accuracy_rating=8.5,
        cost_rating=6.0,
        context_window=128000,
        description="Real-time AI with internet access for threat intelligence",
        icon="🚀",
        color="#000000",
        strengths=["Real-time intel", "Internet access", "Current threat data"],
        best_for=["Threat intelligence", "IOC lookup", "Real-time threat feeds"]
    ),
    
    # Cohere Models
    "command-r-plus": AIModel(
        id="command-r-plus",
        name="Command R+",
        provider=AIProvider.COHERE,
        version="2024",
        specialties=["rag", "enterprise", "retrieval", "grounded"],
        speed_rating=8.0,
        accuracy_rating=8.5,
        cost_rating=5.0,
        context_window=128000,
        description="Enterprise-focused model with strong RAG capabilities for security documentation",
        icon="📚",
        color="#39594D",
        strengths=["RAG integration", "Enterprise ready", "Grounded responses"],
        best_for=["Security documentation", "Knowledge base queries", "Incident analysis"]
    ),
    
    # NVIDIA Models
    "nemotron-70b": AIModel(
        id="nemotron-70b",
        name="Nemotron 70B",
        provider=AIProvider.NVIDIA,
        version="2024",
        specialties=["inference_optimization", "hardware_accelerated", "enterprise"],
        speed_rating=9.5,
        accuracy_rating=8.5,
        cost_rating=5.0,
        context_window=128000,
        description="Hardware-optimized model for high-performance security inference",
        icon="🎮",
        color="#76B900",
        strengths=["GPU optimized", "High throughput", "Enterprise deployment"],
        best_for=["High-volume processing", "GPU infrastructure", "Real-time detection"]
    ),
    
    # Local/Custom Models
    "random-forest-nids": AIModel(
        id="random-forest-nids",
        name="Random Forest NIDS",
        provider=AIProvider.LOCAL,
        version="Custom",
        specialties=["network_traffic", "classification", "low_latency"],
        speed_rating=10.0,
        accuracy_rating=8.0,
        cost_rating=1.0,
        context_window=0,
        description="Custom-trained Random Forest model for network intrusion detection",
        icon="🌲",
        color="#228B22",
        strengths=["Ultra-low latency", "No cloud dependency", "Interpretable"],
        best_for=["Initial detection", "Real-time classification", "Edge deployment"]
    ),
    "xgboost-nids": AIModel(
        id="xgboost-nids",
        name="XGBoost NIDS",
        provider=AIProvider.LOCAL,
        version="Custom",
        specialties=["network_traffic", "boosting", "high_accuracy"],
        speed_rating=9.5,
        accuracy_rating=8.5,
        cost_rating=1.0,
        context_window=0,
        description="Gradient boosting model optimized for network anomaly detection",
        icon="🚀",
        color="#FF6600",
        strengths=["High accuracy", "Fast training", "Feature importance"],
        best_for=["Anomaly detection", "Classification", "Feature analysis"]
    ),
    "autoencoder-anomaly": AIModel(
        id="autoencoder-anomaly",
        name="Autoencoder Anomaly",
        provider=AIProvider.LOCAL,
        version="Custom",
        specialties=["unsupervised", "anomaly_detection", "zero_day"],
        speed_rating=9.0,
        accuracy_rating=7.5,
        cost_rating=1.0,
        context_window=0,
        description="Deep learning autoencoder for unsupervised anomaly detection",
        icon="🔄",
        color="#9932CC",
        strengths=["Zero-day detection", "Unsupervised", "Pattern learning"],
        best_for=["Unknown threats", "Zero-day detection", "Baseline deviation"]
    ),
    "transformer-nids": AIModel(
        id="transformer-nids",
        name="Transformer NIDS",
        provider=AIProvider.LOCAL,
        version="Custom",
        specialties=["sequence_analysis", "temporal_patterns", "advanced"],
        speed_rating=7.0,
        accuracy_rating=9.0,
        cost_rating=2.0,
        context_window=0,
        description="Transformer-based model for sequential network traffic analysis",
        icon="🤖",
        color="#FF1493",
        strengths=["Sequence understanding", "Temporal patterns", "Context aware"],
        best_for=["Multi-stage attacks", "Session analysis", "Behavioral detection"]
    ),
}


# Attack type to optimal model mapping with reasoning
ATTACK_MODEL_MAPPING: Dict[str, Dict] = {
    "DDoS": {
        "primary": "claude-3-haiku",
        "secondary": "gemini-2.0-flash",
        "local": "random-forest-nids",
        "reason": "DDoS attacks require ultra-fast detection due to high volume. Claude 3 Haiku provides the fastest inference time while maintaining accuracy. Local Random Forest model handles initial real-time classification.",
        "defense_strategy": "Immediate rate limiting, traffic analysis, and source blocking. Fast models essential for real-time response."
    },
    "Port Scan": {
        "primary": "gpt-4-turbo",
        "secondary": "mistral-large-2",
        "local": "xgboost-nids",
        "reason": "Port scanning requires pattern recognition across multiple connection attempts. GPT-4 Turbo excels at identifying scan patterns while XGBoost provides fast initial detection.",
        "defense_strategy": "Pattern-based detection with temporal analysis. Need to distinguish between legitimate scanning and malicious reconnaissance."
    },
    "Brute Force": {
        "primary": "claude-3.5-sonnet",
        "secondary": "gpt-4-turbo",
        "local": "random-forest-nids",
        "reason": "Brute force attacks need analysis of authentication patterns and credential stuffing attempts. Claude 3.5 Sonnet provides balanced speed and accuracy for authentication analysis.",
        "defense_strategy": "Credential analysis, rate limiting, and account lockout recommendations. Model suggests optimal thresholds."
    },
    "SQL Injection": {
        "primary": "gpt-4o",
        "secondary": "codestral",
        "local": "transformer-nids",
        "reason": "SQL injection requires deep code analysis and understanding of query structures. GPT-4o's advanced reasoning and Codestral's code expertise provide comprehensive detection.",
        "defense_strategy": "Query parsing, payload analysis, and parameterized query recommendations. Models identify obfuscation techniques."
    },
    "XSS": {
        "primary": "claude-3.5-sonnet",
        "secondary": "gemini-2.0-flash",
        "local": "xgboost-nids",
        "reason": "XSS detection requires understanding of JavaScript, HTML context, and encoding schemes. Claude 3.5 Sonnet excels at web security analysis.",
        "defense_strategy": "Script analysis, context-aware encoding detection, and CSP recommendations."
    },
    "Command Injection": {
        "primary": "gpt-4o",
        "secondary": "codestral",
        "local": "transformer-nids",
        "reason": "Command injection requires understanding of shell syntax, escaping, and OS-specific commands. GPT-4o's reasoning capability identifies complex injection patterns.",
        "defense_strategy": "Command parsing, shell escape analysis, and input sanitization recommendations."
    },
    "Path Traversal": {
        "primary": "claude-3.5-sonnet",
        "secondary": "mistral-large-2",
        "local": "random-forest-nids",
        "reason": "Path traversal needs file system understanding and encoding detection. Claude's security focus handles directory traversal patterns effectively.",
        "defense_strategy": "Path normalization, encoding detection, and chroot recommendations."
    },
    "Malware Communication": {
        "primary": "claude-4-opus",
        "secondary": "gpt-4o",
        "local": "autoencoder-anomaly",
        "reason": "C2 communication detection requires deep analysis of encrypted traffic patterns and behavioral anomalies. Claude 4 Opus provides the most thorough security analysis.",
        "defense_strategy": "Behavioral analysis, beacon detection, and DNS/HTTP pattern analysis. Deep reasoning for APT attribution."
    },
    "Data Exfiltration": {
        "primary": "o1-preview",
        "secondary": "claude-4-opus",
        "local": "autoencoder-anomaly",
        "reason": "Data exfiltration is often subtle and requires deep reasoning about normal vs anomalous data flows. o1-preview's advanced reasoning identifies sophisticated exfil techniques.",
        "defense_strategy": "DLP analysis, volume anomaly detection, and encoding/compression analysis. Strategic thinking for insider threats."
    },
    "Cryptomining": {
        "primary": "gemini-2.0-flash",
        "secondary": "llama-3.3-70b",
        "local": "xgboost-nids",
        "reason": "Cryptomining detection needs resource usage analysis and mining pool communication identification. Fast detection important to minimize resource theft.",
        "defense_strategy": "CPU/GPU usage analysis, mining pool blocklist, and process monitoring recommendations."
    },
    "Phishing": {
        "primary": "llama-3.2-vision",
        "secondary": "gemini-2.5-pro",
        "local": "transformer-nids",
        "reason": "Phishing detection benefits from visual analysis of fraudulent pages. Llama 3.2 Vision can analyze screenshots while Gemini handles content analysis.",
        "defense_strategy": "Visual similarity detection, URL analysis, and brand impersonation identification."
    },
    "Zero-day": {
        "primary": "o1-preview",
        "secondary": "claude-4-opus",
        "local": "autoencoder-anomaly",
        "reason": "Zero-day exploits require the deepest reasoning capabilities to identify unknown attack patterns. o1-preview's chain-of-thought reasoning is essential.",
        "defense_strategy": "Anomaly-based detection, behavioral analysis, and heuristic pattern matching. No signatures available."
    },
    "APT": {
        "primary": "o1-preview",
        "secondary": "gemini-2.5-pro",
        "local": "transformer-nids",
        "reason": "Advanced Persistent Threats require long-term pattern analysis and strategic thinking. o1-preview handles complex multi-stage attack correlation.",
        "defense_strategy": "Kill chain analysis, lateral movement detection, and persistence mechanism identification."
    },
    "Lateral Movement": {
        "primary": "gemini-2.5-pro",
        "secondary": "claude-4-opus",
        "local": "transformer-nids",
        "reason": "Lateral movement detection needs massive context to correlate events across time. Gemini's 2M token context window enables full session analysis.",
        "defense_strategy": "Authentication correlation, privilege escalation detection, and network segmentation analysis."
    },
    "Unknown": {
        "primary": "autoencoder-anomaly",
        "secondary": "gpt-4o",
        "local": "autoencoder-anomaly",
        "reason": "Unknown threats are best detected by unsupervised models that identify deviations from normal behavior without predefined signatures.",
        "defense_strategy": "Baseline comparison, statistical anomaly detection, and human analyst escalation."
    }
}


class AIModelSelector:
    """Intelligent AI model selector for defense operations."""
    
    def __init__(self):
        self.models = AI_MODELS
        self.attack_mapping = ATTACK_MODEL_MAPPING
        self.model_usage_stats = {}
    
    def select_model(
        self, 
        attack_type: str, 
        severity: str = "medium",
        speed_priority: bool = False,
        cost_priority: bool = False,
        privacy_required: bool = False
    ) -> Tuple[AIModel, str, str]:
        """
        Select the optimal AI model for the given attack type.
        
        Returns:
            Tuple of (AIModel, reason, defense_strategy)
        """
        # Get attack mapping or default to Unknown
        mapping = self.attack_mapping.get(attack_type, self.attack_mapping["Unknown"])
        
        # Determine which model to use based on priorities
        if privacy_required:
            model_id = mapping.get("local", "random-forest-nids")
        elif speed_priority and severity not in ["critical", "high"]:
            # Use fastest available
            model_id = self._get_fastest_model(mapping)
        elif cost_priority:
            model_id = mapping.get("local", "gpt-3.5-turbo")
        elif severity == "critical":
            model_id = mapping.get("primary", "gpt-4o")
        elif severity == "high":
            model_id = mapping.get("primary", "claude-3.5-sonnet")
        else:
            model_id = mapping.get("secondary", mapping.get("primary", "gpt-4-turbo"))
        
        model = self.models.get(model_id, self.models["gpt-4o"])
        reason = mapping.get("reason", "Default model selection based on general capabilities.")
        strategy = mapping.get("defense_strategy", "Standard defense protocols applied.")
        
        # Update usage stats
        self.model_usage_stats[model_id] = self.model_usage_stats.get(model_id, 0) + 1
        
        return model, reason, strategy
    
    def _get_fastest_model(self, mapping: Dict) -> str:
        """Get the fastest model from the mapping options."""
        options = [mapping.get("local"), mapping.get("secondary"), mapping.get("primary")]
        options = [o for o in options if o and o in self.models]
        
        if not options:
            return "claude-3-haiku"
        
        return max(options, key=lambda x: self.models[x].speed_rating)
    
    def get_model_info(self, model_id: str) -> Optional[AIModel]:
        """Get information about a specific model."""
        return self.models.get(model_id)
    
    def get_all_models(self) -> Dict[str, AIModel]:
        """Get all available models."""
        return self.models
    
    def get_models_by_provider(self, provider: AIProvider) -> List[AIModel]:
        """Get all models from a specific provider."""
        return [m for m in self.models.values() if m.provider == provider]
    
    def get_recommended_models(self, attack_type: str) -> Dict[str, AIModel]:
        """Get all recommended models for an attack type."""
        mapping = self.attack_mapping.get(attack_type, self.attack_mapping["Unknown"])
        result = {}
        
        for key in ["primary", "secondary", "local"]:
            model_id = mapping.get(key)
            if model_id and model_id in self.models:
                result[key] = self.models[model_id]
        
        return result
    
    def get_defense_analysis(self, attack_type: str, severity: str = "medium") -> Dict:
        """Get comprehensive defense analysis for an attack type."""
        model, reason, strategy = self.select_model(attack_type, severity)
        recommended = self.get_recommended_models(attack_type)
        
        return {
            "selected_model": {
                "id": model.id,
                "name": model.name,
                "provider": model.provider.value,
                "version": model.version,
                "icon": model.icon,
                "color": model.color,
                "strengths": model.strengths,
                "speed_rating": model.speed_rating,
                "accuracy_rating": model.accuracy_rating,
            },
            "reason": reason,
            "defense_strategy": strategy,
            "alternative_models": [
                {
                    "role": role,
                    "id": m.id,
                    "name": m.name,
                    "icon": m.icon
                }
                for role, m in recommended.items()
            ],
            "attack_type": attack_type,
            "severity": severity,
        }


# Global instance
ai_selector = AIModelSelector()


def get_model_for_alert(attack_type: str, severity: str = "medium") -> Dict:
    """Convenience function to get model info for an alert."""
    return ai_selector.get_defense_analysis(attack_type, severity)


def get_all_ai_models() -> List[Dict]:
    """Get all AI models as dictionaries."""
    return [
        {
            "id": m.id,
            "name": m.name,
            "provider": m.provider.value,
            "version": m.version,
            "icon": m.icon,
            "color": m.color,
            "description": m.description,
            "specialties": m.specialties,
            "speed_rating": m.speed_rating,
            "accuracy_rating": m.accuracy_rating,
            "cost_rating": m.cost_rating,
            "context_window": m.context_window,
            "strengths": m.strengths,
            "best_for": m.best_for,
        }
        for m in AI_MODELS.values()
    ]
