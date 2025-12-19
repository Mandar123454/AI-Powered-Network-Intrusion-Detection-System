"""
SHAP Explainability Module for AI-NIDS
Provides interpretable explanations for model predictions
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for intrusion detection models.
    Provides both global and local explanations.
    """
    
    def __init__(
        self,
        model: object,
        feature_names: List[str],
        model_type: str = 'tree',
        background_samples: int = 100
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model instance
            feature_names: List of feature names
            model_type: Type of model ('tree', 'deep', 'kernel')
            background_samples: Number of background samples for SHAP
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.background_samples = background_samples
        
        self.explainer: Optional[shap.Explainer] = None
        self.background_data: Optional[np.ndarray] = None
        self.global_shap_values: Optional[np.ndarray] = None
        
        self.metadata: Dict = {
            'created_at': datetime.now().isoformat(),
            'model_type': model_type,
            'n_features': len(feature_names)
        }
        
        logger.info(f"Initialized SHAPExplainer for {model_type} model")
    
    def fit(self, X_background: np.ndarray) -> None:
        """
        Fit the SHAP explainer with background data.
        
        Args:
            X_background: Background data for SHAP calculations
        """
        logger.info("Fitting SHAP explainer...")
        
        # Sample background data if too large
        if len(X_background) > self.background_samples:
            indices = np.random.choice(len(X_background), self.background_samples, replace=False)
            self.background_data = X_background[indices]
        else:
            self.background_data = X_background
        
        # Create appropriate explainer
        if self.model_type == 'tree':
            # For tree-based models (XGBoost, RandomForest, etc.)
            try:
                self.explainer = shap.TreeExplainer(self.model.model if hasattr(self.model, 'model') else self.model)
            except:
                # Fallback to kernel explainer
                logger.warning("TreeExplainer failed, falling back to KernelExplainer")
                self.model_type = 'kernel'
                self._create_kernel_explainer()
        
        elif self.model_type == 'deep':
            # For deep learning models
            try:
                self.explainer = shap.DeepExplainer(
                    self.model.model if hasattr(self.model, 'model') else self.model,
                    self.background_data
                )
            except:
                logger.warning("DeepExplainer failed, falling back to KernelExplainer")
                self.model_type = 'kernel'
                self._create_kernel_explainer()
        
        elif self.model_type == 'kernel':
            self._create_kernel_explainer()
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info("SHAP explainer fitted successfully")
    
    def _create_kernel_explainer(self) -> None:
        """Create kernel SHAP explainer."""
        # Get prediction function
        if hasattr(self.model, 'predict_proba'):
            predict_fn = lambda x: self.model.predict_proba(x)[:, 1] if self.model.predict_proba(x).ndim > 1 else self.model.predict_proba(x)
        else:
            predict_fn = self.model.predict
        
        self.explainer = shap.KernelExplainer(predict_fn, self.background_data)
    
    def explain(self, X: np.ndarray) -> shap.Explanation:
        """
        Calculate SHAP values for input data.
        
        Args:
            X: Input features to explain
            
        Returns:
            SHAP Explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case - use positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        return shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[1],
            data=X,
            feature_names=self.feature_names
        )
    
    def explain_single(self, x: np.ndarray) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            x: Single input sample (1D or 2D array)
            
        Returns:
            Dictionary with explanation details
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        explanation = self.explain(x)
        shap_values = explanation.values[0] if explanation.values.ndim > 1 else explanation.values
        
        # Get top contributing features
        feature_importance = list(zip(self.feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_positive = [(f, v) for f, v in feature_importance if v > 0][:5]
        top_negative = [(f, v) for f, v in feature_importance if v < 0][:5]
        
        return {
            'shap_values': dict(zip(self.feature_names, shap_values.tolist())),
            'base_value': float(explanation.base_values) if np.isscalar(explanation.base_values) else float(explanation.base_values[0]),
            'prediction_contribution': float(np.sum(shap_values)),
            'top_positive_contributors': [(f, float(v)) for f, v in top_positive],
            'top_negative_contributors': [(f, float(v)) for f, v in top_negative],
            'feature_values': dict(zip(self.feature_names, x[0].tolist()))
        }
    
    def global_importance(self, X: np.ndarray, return_df: bool = True) -> Union[np.ndarray, 'pd.DataFrame']:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            X: Data to calculate importance over
            return_df: Whether to return as DataFrame
            
        Returns:
            Feature importance array or DataFrame
        """
        explanation = self.explain(X)
        self.global_shap_values = explanation.values
        
        # Mean absolute SHAP values
        mean_shap = np.abs(explanation.values).mean(axis=0)
        
        if return_df:
            import pandas as pd
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            return df
        
        return mean_shap
    
    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 20
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to plot
            save_path: Path to save the plot
            max_display: Maximum features to display
        """
        explanation = self.explain(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            explanation.values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved summary plot to {save_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        x: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            x: Single input sample
            save_path: Path to save the plot
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        explanation = self.explain(x)
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=explanation.values[0],
                base_values=explanation.base_values if np.isscalar(explanation.base_values) else explanation.base_values[0],
                data=x[0],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved waterfall plot to {save_path}")
        
        plt.close()
    
    def plot_force(
        self,
        x: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create SHAP force plot (returns HTML for embedding).
        
        Args:
            x: Single input sample
            save_path: Path to save HTML file
            
        Returns:
            HTML string for the force plot
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        explanation = self.explain(x)
        
        # Create force plot
        force_plot = shap.force_plot(
            explanation.base_values if np.isscalar(explanation.base_values) else explanation.base_values[0],
            explanation.values[0],
            x[0],
            feature_names=self.feature_names
        )
        
        html = shap.getjs() + force_plot.html()
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            logger.info(f"Saved force plot to {save_path}")
        
        return html
    
    def plot_dependence(
        self,
        feature: str,
        X: np.ndarray,
        save_path: Optional[str] = None,
        interaction_feature: Optional[str] = None
    ) -> None:
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            feature: Feature name to plot
            X: Data for the plot
            save_path: Path to save the plot
            interaction_feature: Feature for interaction coloring
        """
        explanation = self.explain(X)
        
        feature_idx = self.feature_names.index(feature) if feature in self.feature_names else 0
        interaction_idx = None
        if interaction_feature and interaction_feature in self.feature_names:
            interaction_idx = self.feature_names.index(interaction_feature)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            explanation.values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved dependence plot to {save_path}")
        
        plt.close()
    
    def generate_explanation_text(self, x: np.ndarray, prediction: str) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            x: Single input sample
            prediction: Model prediction (e.g., 'Attack', 'Normal')
            
        Returns:
            Human-readable explanation
        """
        explanation = self.explain_single(x)
        
        text = f"### Prediction: {prediction}\n\n"
        text += f"**Base probability:** {explanation['base_value']:.3f}\n"
        text += f"**Final contribution:** {explanation['prediction_contribution']:.3f}\n\n"
        
        text += "**Top factors increasing risk:**\n"
        for feature, value in explanation['top_positive_contributors']:
            feature_val = explanation['feature_values'].get(feature, 'N/A')
            text += f"- {feature} = {feature_val:.2f} (contribution: +{value:.3f})\n"
        
        text += "\n**Top factors decreasing risk:**\n"
        for feature, value in explanation['top_negative_contributors']:
            feature_val = explanation['feature_values'].get(feature, 'N/A')
            text += f"- {feature} = {feature_val:.2f} (contribution: {value:.3f})\n"
        
        return text
    
    def save_explanation_report(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        output_dir: str
    ) -> None:
        """
        Generate and save comprehensive explanation report.
        
        Args:
            X: Input data
            y_pred: Model predictions
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Global importance
        importance_df = self.global_importance(X)
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        # Summary plot
        self.plot_summary(X, os.path.join(output_dir, 'summary_plot.png'))
        
        # Individual explanations for attacks
        attack_indices = np.where(y_pred == 1)[0][:10]  # First 10 attacks
        
        explanations = []
        for i, idx in enumerate(attack_indices):
            exp = self.explain_single(X[idx])
            exp['sample_index'] = int(idx)
            explanations.append(exp)
            
            # Individual waterfall plot
            self.plot_waterfall(X[idx], os.path.join(output_dir, f'waterfall_attack_{i}.png'))
        
        # Save explanations
        with open(os.path.join(output_dir, 'attack_explanations.json'), 'w') as f:
            json.dump(explanations, f, indent=2)
        
        logger.info(f"Saved explanation report to {output_dir}")


def create_explainer(
    model: object,
    feature_names: List[str],
    X_background: np.ndarray,
    model_type: str = 'auto'
) -> SHAPExplainer:
    """
    Factory function to create and fit a SHAPExplainer.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_background: Background data for SHAP
        model_type: Type of model ('auto', 'tree', 'deep', 'kernel')
        
    Returns:
        Fitted SHAPExplainer
    """
    # Auto-detect model type
    if model_type == 'auto':
        model_obj = model.model if hasattr(model, 'model') else model
        model_class = model_obj.__class__.__name__.lower()
        
        if any(t in model_class for t in ['xgb', 'forest', 'tree', 'gradient']):
            model_type = 'tree'
        elif any(t in model_class for t in ['lstm', 'neural', 'torch', 'tensor']):
            model_type = 'deep'
        else:
            model_type = 'kernel'
    
    explainer = SHAPExplainer(model, feature_names, model_type)
    explainer.fit(X_background)
    
    return explainer
