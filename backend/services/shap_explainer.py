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
        self.initialized = False
        self._load_model()
    
    def _load_model(self):
        """Load trained model and initialize SHAP explainer."""
        try:
            self.model, self.feature_names = load_model()
            
            # Fix XGBoost base_score issue for SHAP
            if hasattr(self.model, 'base_score'):
                try:
                    if isinstance(self.model.base_score, str):
                        # Parse string array format like '[4.325E-1,1.775E-1,3.9E-1]'
                        base_score_str = self.model.base_score.strip('[]')
                        # Split by comma and convert to float, handling scientific notation
                        base_scores = []
                        for x in base_score_str.split(','):
                            try:
                                base_scores.append(float(x.strip()))
                            except ValueError:
                                # Handle scientific notation parsing issues
                                base_scores.append(0.5)  # Default fallback
                        
                        # Use the first value or mean if multiple
                        self.model.base_score = base_scores[0] if len(base_scores) == 1 else np.mean(base_scores)
                        print(f"Fixed base_score: {self.model.base_score}")
                except Exception as base_score_error:
                    print(f"Could not fix base_score, using default: {base_score_error}")
                    self.model.base_score = 0.5  # Safe default
            
            self.explainer = shap.TreeExplainer(self.model)
            self.initialized = True
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print(f"Warning: SHAP explainer initialization failed: {e}")
            self.initialized = False
    
    def _check_initialized(self):
        """Check if explainer is properly initialized."""
        if not self.initialized:
            # Try to reinitialize
            self._load_model()
            if not self.initialized:
                raise Exception("SHAP explainer not properly initialized")
    
    def get_global_feature_importance(self, sample_size: int = 100) -> Dict[str, Any]:
        """Generate global feature importance using SHAP."""
        try:
            self._check_initialized()
            
            # Load sample data for global explanation
            from src.preprocessing import preprocess_data
            data = preprocess_data()
            X_sample = data['X_train'][:sample_size]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Multi-class output
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
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
            self._check_initialized()
            
            # Preprocess input
            X_processed = preprocess_single_input(input_data)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Multi-class output
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Get first sample
            
            # Create feature impact dictionary
            feature_impact = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_values):
                    feature_impact[feature] = float(shap_values[i])
            
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
            # Return rule-based fallback explanation
            return self._generate_fallback_explanation(input_data)
    
    def _generate_fallback_explanation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rule-based explanation when SHAP is unavailable."""
        # Rule-based feature importance based on domain knowledge
        feature_impacts = {
            'credit_score': -0.1 if input_data.get('credit_score', 700) > 650 else 0.15,
            'delivery_score': 0.08 if input_data.get('delivery_score', 80) > 75 else -0.12,
            'financial_stability_index': 0.05 if input_data.get('financial_stability_index', 0.5) > 0.6 else -0.1,
            'past_disputes': 0.08 * min(input_data.get('past_disputes', 0), 5),
            'avg_payment_delay_days': 0.006 * min(input_data.get('avg_payment_delay_days', 0), 30),
            'dispute_rate': 0.2 * input_data.get('dispute_rate', 0),
            'late_ratio': 0.15 * input_data.get('late_ratio', 0),
            'contract_value': -0.05 if input_data.get('contract_value', 0) > 1000000 else 0.02,
            'relationship_years': -0.03 * min(input_data.get('relationship_years', 0), 10)
        }
        
        # Sort by absolute impact
        sorted_impact = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Categorize impacts
        positive_impacts = [(f, v) for f, v in sorted_impact if v > 0][:5]
        negative_impacts = [(f, v) for f, v in sorted_impact if v < 0][:5]
        
        return {
            'feature_impact': feature_impacts,
            'top_positive': dict(positive_impacts),
            'top_negative': dict(negative_impacts),
            'most_influential': sorted_impact[0] if sorted_impact else None,
            'note': 'Rule-based explanation (SHAP unavailable)'
        }
    
    def generate_summary_plot(self, sample_size: int = 50) -> str:
        """Generate SHAP summary plot and return as base64 string."""
        try:
            self._check_initialized()
            
            # Load sample data
            from src.preprocessing import preprocess_data
            data = preprocess_data()
            X_sample = data['X_train'][:sample_size]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Multi-class output
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]
            
            # Create SHAP-style summary plot
            plt.figure(figsize=(12, 8))
            
            # Calculate feature importance and create scatter plot
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(feature_importance)[-15:]  # Top 15 features
            
            # Create scatter plot for each feature
            y_pos = np.arange(len(top_indices))
            
            for i, idx in enumerate(top_indices):
                # Get SHAP values for this feature
                feature_shap = shap_values[:, idx]
                # Get feature values (normalized for color)
                feature_vals = X_sample.iloc[:, idx] if hasattr(X_sample, 'iloc') else X_sample[:, idx]
                
                # Create scatter plot
                scatter = plt.scatter(feature_shap, [i] * len(feature_shap), 
                                    c=feature_vals, cmap='coolwarm', alpha=0.6, s=20)
            
            # Customize plot
            plt.yticks(y_pos, [self.feature_names[i].replace('_', ' ').title() for i in top_indices])
            plt.xlabel('SHAP Value (Impact on Model Output)')
            plt.title('SHAP Summary Plot - Feature Impact Distribution')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Feature Value', rotation=270, labelpad=20)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            print(f"Error generating summary plot: {e}")
            # Generate a proper fallback visualization
            return self._generate_fallback_summary_plot()
    
    def generate_force_plot(self, input_data: Dict[str, Any]) -> str:
        """Generate SHAP force plot for single prediction."""
        try:
            self._check_initialized()
            
            # Preprocess input
            X_processed = preprocess_single_input(input_data)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Multi-class output
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Get first sample
            
            # Create enhanced force plot
            plt.figure(figsize=(14, 8))
            
            # Get SHAP values and sort by magnitude
            shap_vals = shap_values
            indices = np.argsort(np.abs(shap_vals))[-12:]  # Top 12 features
            
            # Create horizontal bar chart with enhanced styling
            y_pos = np.arange(len(indices))
            values = shap_vals[indices]
            
            # Color coding: red for risk-increasing, blue for risk-decreasing
            colors = ['#ff4444' if val > 0 else '#4444ff' for val in values]
            
            bars = plt.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            feature_names = [self.feature_names[i].replace('_', ' ').title() for i in indices]
            plt.yticks(y_pos, feature_names)
            plt.xlabel('SHAP Value (Impact on Risk Prediction)', fontsize=12)
            plt.title('SHAP Force Plot - Individual Feature Contributions', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                width = bar.get_width()
                plt.text(width + (0.005 if width >= 0 else -0.005), 
                        bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', 
                        ha='left' if width >= 0 else 'right', 
                        va='center', fontweight='bold', fontsize=9)
            
            # Add legend with better positioning
            legend_text = 'Red: Increases Risk (Positive SHAP)\nBlue: Decreases Risk (Negative SHAP)'
            plt.text(0.02, 0.98, legend_text, 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # Add grid for better readability
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            print(f"Error generating force plot: {e}")
            # Generate a proper fallback visualization
            return self._generate_fallback_force_plot(input_data)
    
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
    
    def _generate_fallback_summary_plot(self) -> str:
        """Generate fallback summary plot when SHAP is unavailable."""
        try:
            # Create a meaningful fallback plot using domain knowledge
            plt.figure(figsize=(12, 8))
            
            # Domain knowledge-based feature importance
            features = [
                'Credit Score', 'Financial Stability', 'Delivery Score', 'Dispute Rate',
                'Late Payment Ratio', 'Payment Delay Days', 'Past Disputes', 'Delivery Quality',
                'Contract Value', 'Relationship Years', 'Clause Risk Score', 'Annual Spend'
            ]
            
            importance_values = [0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            
            bars = plt.barh(y_pos, importance_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.yticks(y_pos, features)
            plt.xlabel('Feature Importance (Domain Knowledge)', fontsize=12)
            plt.title('Procurement Risk Feature Importance\n(Fallback Analysis)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, importance_values):
                width = bar.get_width()
                plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{value:.2f}', ha='left', va='center', fontweight='bold')
            
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            print(f"Error generating fallback summary plot: {e}")
            return None
    
    def _generate_fallback_force_plot(self, input_data: Dict[str, Any]) -> str:
        """Generate fallback force plot when SHAP is unavailable."""
        try:
            # Generate rule-based impacts
            feature_impacts = {
                'credit_score': 0.15 if input_data.get('credit_score', 700) < 650 else -0.1,
                'delivery_score': -0.12 if input_data.get('delivery_score', 80) < 70 else 0.08,
                'financial_stability_index': -0.1 if input_data.get('financial_stability_index', 0.5) < 0.4 else 0.05,
                'past_disputes': 0.08 * min(input_data.get('past_disputes', 0), 5),
                'avg_payment_delay_days': 0.006 * min(input_data.get('avg_payment_delay_days', 0), 30),
                'dispute_rate': 0.2 * input_data.get('dispute_rate', 0),
                'late_ratio': 0.15 * input_data.get('late_ratio', 0),
                'contract_value': -0.05 if input_data.get('contract_value', 0) > 1000000 else 0.02,
                'relationship_years': -0.03 * min(input_data.get('relationship_years', 0), 10),
                'avg_delivery_quality': 0.05 if input_data.get('avg_delivery_quality', 80) > 85 else -0.08
            }
            
            # Sort by absolute impact
            sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            
            plt.figure(figsize=(14, 8))
            
            features = [f.replace('_', ' ').title() for f, _ in sorted_impacts]
            values = [v for _, v in sorted_impacts]
            
            # Color coding
            colors = ['#ff4444' if val > 0 else '#4444ff' for val in values]
            
            y_pos = np.arange(len(features))
            bars = plt.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.yticks(y_pos, features)
            plt.xlabel('Estimated Impact on Risk Prediction', fontsize=12)
            plt.title('Feature Impact Analysis\n(Rule-based Fallback)', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                width = bar.get_width()
                plt.text(width + (0.005 if width >= 0 else -0.005), 
                        bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', 
                        ha='left' if width >= 0 else 'right', 
                        va='center', fontweight='bold', fontsize=9)
            
            # Add legend
            legend_text = 'Red: Increases Risk\nBlue: Decreases Risk\n(Rule-based Analysis)'
            plt.text(0.02, 0.98, legend_text, 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
            
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            print(f"Error generating fallback force plot: {e}")
            return None