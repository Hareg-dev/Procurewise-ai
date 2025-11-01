"""
SHAP explainability service for generating model explanations and visualizations.
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import base64
import io
from typing import Dict, List, Any, Tuple
from src.model_training import load_model
from src.preprocessing import preprocess_single_input
import config

class SHAPExplainer:
    """Service for generating SHAP explanations and visualizations."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.explainer = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and initialize SHAP explainer."""
        try:
            self.model, self.feature_names = load_model()
            self.explainer = shap.TreeExplainer(self.model)
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            raise
    
    def get_global_feature_importance(self, sample_size: int = 100) -> Dict[str, Any]:
        """Generate global feature importance using SHAP."""
        try:
            # Load sample data for global explanation
            from src.preprocessing import preprocess_data
            data = preprocess_data()
            X_sample = data['X_train'][:sample_size]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]  # Use medium risk class
            
            # Calculate mean absolute SHAP values for global importance
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(mean_shap):
                    feature_importance[feature] = float(mean_shap[i])
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'global_importance': dict(sorted_features),
                'top_features': dict(sorted_features[:10]),
                'sample_size': sample_size
            }
            
        except Exception as e:
            print(f"Error generating global SHAP explanation: {e}")
            return {'error': str(e)}
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction."""
        try:
            # Preprocess input
            X_processed = preprocess_single_input(input_data)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Handle multi-class output
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]  # Use medium risk class
            
            # Create feature impact dictionary
            feature_impact = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_values[0]):
                    feature_impact[feature] = float(shap_values[0][i])
            
            # Sort by absolute impact
            sorted_impact = sorted(feature_impact.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Categorize impacts
            positive_impacts = [(f, v) for f, v in sorted_impact if v > 0][:5]
            negative_impacts = [(f, v) for f, v in sorted_impact if v < 0][:5]
            
            return {
                'feature_impact': feature_impact,
                'top_positive': dict(positive_impacts),
                'top_negative': dict(negative_impacts),
                'most_influential': sorted_impact[0] if sorted_impact else None
            }
            
        except Exception as e:
            print(f"Error generating SHAP explanation: {e}")
            return {'error': str(e)}
    
    def generate_summary_plot(self, sample_size: int = 50) -> str:
        """Generate SHAP summary plot and return as base64 string."""
        try:
            # Load sample data
            from src.preprocessing import preprocess_data
            data = preprocess_data()
            X_sample = data['X_train'][:sample_size]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            print(f"Error generating summary plot: {e}")
            return None
    
    def generate_force_plot(self, input_data: Dict[str, Any]) -> str:
        """Generate SHAP force plot for single prediction."""
        try:
            # Preprocess input
            X_processed = preprocess_single_input(input_data)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Handle multi-class output
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]
            
            # Create force plot
            plt.figure(figsize=(12, 6))
            
            # Manual force plot creation (simplified)
            feature_values = X_processed[0]
            shap_vals = shap_values[0]
            
            # Sort by SHAP value magnitude
            indices = np.argsort(np.abs(shap_vals))[-10:]  # Top 10 features
            
            colors = ['red' if val < 0 else 'blue' for val in shap_vals[indices]]
            
            plt.barh(range(len(indices)), shap_vals[indices], color=colors, alpha=0.7)
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Feature Impact on Risk Prediction')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            print(f"Error generating force plot: {e}")
            return None
    
    def get_explanation_summary(self, shap_explanation: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary of SHAP explanation."""
        try:
            if 'error' in shap_explanation:
                return {'error': shap_explanation['error']}
            
            top_positive = shap_explanation.get('top_positive', {})
            top_negative = shap_explanation.get('top_negative', {})
            most_influential = shap_explanation.get('most_influential')
            
            summary = {
                'key_insight': '',
                'risk_drivers': '',
                'risk_mitigators': '',
                'recommendation': ''
            }
            
            if most_influential:
                feature, impact = most_influential
                direction = "increases" if impact < 0 else "decreases"
                summary['key_insight'] = f"The most influential factor is {feature.replace('_', ' ').title()}, which {direction} risk by {abs(impact):.3f} points."
            
            if top_negative:
                drivers = [f.replace('_', ' ').title() for f in list(top_negative.keys())[:3]]
                summary['risk_drivers'] = f"Main risk drivers: {', '.join(drivers)}"
            
            if top_positive:
                mitigators = [f.replace('_', ' ').title() for f in list(top_positive.keys())[:3]]
                summary['risk_mitigators'] = f"Risk mitigating factors: {', '.join(mitigators)}"
            
            # Basic recommendation based on top factors
            if top_negative:
                top_risk_factor = list(top_negative.keys())[0]
                if 'payment' in top_risk_factor.lower():
                    summary['recommendation'] = "Focus on improving payment terms and monitoring cash flow."
                elif 'delivery' in top_risk_factor.lower():
                    summary['recommendation'] = "Enhance delivery performance monitoring and set clear SLAs."
                elif 'dispute' in top_risk_factor.lower():
                    summary['recommendation'] = "Implement better communication and conflict resolution processes."
                else:
                    summary['recommendation'] = "Address the key risk factors through targeted mitigation strategies."
            
            return summary
            
        except Exception as e:
            print(f"Error generating explanation summary: {e}")
            return {'error': str(e)}