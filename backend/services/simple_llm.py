"""
Simple LLM service with fallback responses for procurement insights.
"""
import random
from typing import Dict, Any

class SimpleLLMAgent:
    """Simple LLM agent with predefined responses for procurement scenarios."""
    
    def __init__(self):
        self.responses = {
            'high_risk': [
                "This supplier shows high risk factors. Recommend: 1) Enhanced due diligence 2) Shorter contract terms 3) Performance bonds 4) Regular monitoring",
                "High risk detected. Consider: 1) Alternative suppliers 2) Risk mitigation clauses 3) Frequent reviews 4) Penalty terms for non-compliance",
                "Significant risk identified. Actions: 1) Detailed financial review 2) Reference checks 3) Pilot contracts 4) Diversification strategy"
            ],
            'medium_risk': [
                "Medium risk supplier. Recommend: 1) Standard monitoring 2) Performance metrics 3) Regular check-ins 4) Contract safeguards",
                "Moderate risk level. Consider: 1) Performance incentives 2) Quarterly reviews 3) Clear SLAs 4) Backup supplier identification",
                "Balanced risk profile. Actions: 1) Regular performance tracking 2) Relationship management 3) Contract optimization 4) Risk monitoring"
            ],
            'low_risk': [
                "Low risk supplier. Recommend: 1) Long-term partnership 2) Volume discounts 3) Strategic collaboration 4) Minimal oversight",
                "Excellent supplier profile. Consider: 1) Extended contracts 2) Preferred partner status 3) Joint initiatives 4) Reduced monitoring",
                "Strong supplier relationship. Actions: 1) Partnership development 2) Innovation collaboration 3) Loyalty programs 4) Strategic planning"
            ],
            'payment_delay': [
                "Payment delays detected. Recommend: 1) Shorter payment terms 2) Early payment discounts 3) Automated reminders 4) Credit monitoring",
                "Cash flow concerns identified. Consider: 1) Payment milestones 2) Performance guarantees 3) Financial health checks 4) Alternative terms"
            ],
            'delivery_issues': [
                "Delivery performance concerns. Recommend: 1) Clear SLAs 2) Penalty clauses 3) Backup suppliers 4) Performance tracking",
                "Supply chain risks detected. Consider: 1) Diversification 2) Buffer inventory 3) Alternative logistics 4) Performance bonds"
            ],
            'general': [
                "Based on the risk assessment, I recommend implementing appropriate risk mitigation strategies tailored to the specific risk factors identified.",
                "The analysis suggests focusing on the key risk drivers while maintaining supplier relationships through balanced contract terms.",
                "Consider implementing a tiered approach to supplier management based on the risk level and strategic importance."
            ]
        }
    
    def generate_recommendation(self, prediction_result: Dict[str, Any], 
                              shap_explanation: Dict[str, Any],
                              input_data: Dict[str, Any]) -> str:
        """Generate contextual recommendation based on prediction results."""
        
        risk_level = prediction_result.get('prediction', 'unknown').lower()
        confidence = prediction_result.get('confidence', 0)
        
        # Get base recommendation
        if risk_level in self.responses:
            base_response = random.choice(self.responses[risk_level])
        else:
            base_response = random.choice(self.responses['general'])
        
        # Add context based on SHAP values
        context = self._analyze_shap_factors(shap_explanation)
        
        # Combine response
        full_response = f"{base_response}\n\n"
        
        if context:
            full_response += f"Key factors to address: {context}\n\n"
        
        full_response += f"Confidence Level: {confidence:.1%} - "
        
        if confidence > 0.8:
            full_response += "High confidence in this assessment."
        elif confidence > 0.6:
            full_response += "Moderate confidence - monitor closely."
        else:
            full_response += "Lower confidence - gather more data."
        
        return full_response
    
    def _analyze_shap_factors(self, shap_explanation: Dict[str, Any]) -> str:
        """Analyze SHAP factors and provide specific guidance."""
        
        if not shap_explanation or 'top_negative' not in shap_explanation:
            return ""
        
        top_negative = shap_explanation.get('top_negative', {})
        factors = []
        
        for feature, impact in list(top_negative.items())[:3]:
            if 'payment' in feature.lower() or 'delay' in feature.lower():
                factors.append("payment terms and cash flow management")
            elif 'delivery' in feature.lower() or 'performance' in feature.lower():
                factors.append("delivery performance and reliability")
            elif 'dispute' in feature.lower():
                factors.append("dispute resolution and relationship management")
            elif 'financial' in feature.lower() or 'credit' in feature.lower():
                factors.append("financial stability and creditworthiness")
            elif 'contract' in feature.lower():
                factors.append("contract terms and risk allocation")
        
        return ", ".join(set(factors)) if factors else "overall risk profile"
    
    def chat_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """Generate chat response based on user message."""
        
        message_lower = user_message.lower()
        
        # Risk-related queries
        if any(word in message_lower for word in ['risk', 'high', 'dangerous']):
            return random.choice([
                "High risk suppliers require enhanced due diligence, shorter contract terms, and regular monitoring. Consider performance bonds and penalty clauses.",
                "Risk management involves diversification, clear SLAs, financial monitoring, and backup supplier strategies.",
                "Key risk factors include payment delays, delivery issues, financial instability, and poor performance history."
            ])
        
        # Payment-related queries
        elif any(word in message_lower for word in ['payment', 'cash', 'financial']):
            return random.choice([
                "Payment risk can be mitigated through shorter terms, early payment discounts, credit monitoring, and performance guarantees.",
                "Financial stability indicators include credit scores, cash flow, debt ratios, and payment history patterns.",
                "Consider milestone payments, automated reminders, and financial health assessments for payment risk management."
            ])
        
        # Contract-related queries
        elif any(word in message_lower for word in ['contract', 'terms', 'agreement']):
            return random.choice([
                "Contract terms should include clear SLAs, penalty clauses, performance metrics, and risk allocation provisions.",
                "Key contract elements: delivery schedules, quality standards, payment terms, dispute resolution, and termination clauses.",
                "Risk-based contracting involves tiered terms, performance incentives, and appropriate safeguards based on supplier risk level."
            ])
        
        # Performance queries
        elif any(word in message_lower for word in ['performance', 'delivery', 'quality']):
            return random.choice([
                "Performance management requires clear metrics, regular monitoring, feedback mechanisms, and improvement plans.",
                "Delivery performance can be improved through better planning, communication, backup suppliers, and performance incentives.",
                "Quality assurance involves specifications, testing, audits, and continuous improvement processes."
            ])
        
        # General procurement queries
        elif any(word in message_lower for word in ['procurement', 'supplier', 'vendor']):
            return random.choice([
                "Effective procurement involves supplier evaluation, risk assessment, contract management, and relationship building.",
                "Supplier management best practices include diversification, performance monitoring, relationship development, and strategic partnerships.",
                "Procurement risk management requires assessment, mitigation strategies, monitoring, and continuous improvement."
            ])
        
        # Default response
        else:
            return random.choice([
                "I can help you with procurement risk assessment, supplier management, contract terms, and risk mitigation strategies. What specific area would you like to discuss?",
                "I'm here to assist with procurement decisions, risk analysis, and supplier relationship management. How can I help you today?",
                "Feel free to ask about risk factors, supplier evaluation, contract optimization, or performance management strategies."
            ])
    
    def explain_prediction(self, prediction_result: Dict[str, Any],
                          shap_explanation: Dict[str, Any],
                          input_data: Dict[str, Any]) -> str:
        """Explain prediction results in natural language."""
        
        risk_level = prediction_result.get('prediction', 'unknown')
        confidence = prediction_result.get('confidence', 0)
        
        explanation = f"The AI model predicts this supplier has {risk_level.upper()} risk with {confidence:.1%} confidence.\n\n"
        
        # Analyze key factors
        if shap_explanation and 'top_negative' in shap_explanation:
            top_negative = shap_explanation['top_negative']
            top_positive = shap_explanation.get('top_positive', {})
            
            if top_negative:
                explanation += "Main risk drivers:\n"
                for i, (feature, impact) in enumerate(list(top_negative.items())[:3], 1):
                    feature_name = feature.replace('_', ' ').title()
                    explanation += f"{i}. {feature_name} (impact: {impact:.3f})\n"
                explanation += "\n"
            
            if top_positive:
                explanation += "Risk mitigating factors:\n"
                for i, (feature, impact) in enumerate(list(top_positive.items())[:3], 1):
                    feature_name = feature.replace('_', ' ').title()
                    explanation += f"{i}. {feature_name} (impact: +{impact:.3f})\n"
                explanation += "\n"
        
        # Add recommendation
        explanation += "Recommended actions:\n"
        recommendation = self.generate_recommendation(prediction_result, shap_explanation, input_data)
        explanation += recommendation.split('\n\n')[0]  # First paragraph only
        
        return explanation