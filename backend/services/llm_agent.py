"""
LLM agent service using Mistral 7B via Ollama for procurement risk insights.
"""
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import config

class ProcurementLLMAgent:
    """LLM agent for procurement risk analysis and recommendations."""
    
    def __init__(self):
        self.ollama_url = config.OLLAMA_BASE_URL
        self.model_name = "phi3:mini"
        self.conversation_history = {}
    
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Make API call to Ollama."""
        try:
            payload = {
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def generate_prediction_explanation(self, prediction_result: Dict[str, Any], 
                                      shap_explanation: Dict[str, Any],
                                      input_data: Dict[str, Any]) -> str:
        """Generate natural language explanation of prediction and SHAP results."""
        
        system_prompt = "You are a procurement risk expert. Give brief, actionable advice."
        
        # Prepare context
        risk_level = prediction_result.get('prediction', 'unknown')
        confidence = prediction_result.get('confidence', 0)
        
        # SHAP insights
        top_positive = shap_explanation.get('top_positive', {})
        top_negative = shap_explanation.get('top_negative', {})
        
        prompt = f"""Risk Level: {risk_level.upper()} ({confidence:.0%} confidence)
Industry: {input_data.get('industry', 'Unknown')}
Top Risk Factors: {', '.join(list(top_negative.keys())[:2])}

Provide 3 specific recommendations to reduce procurement risk:"""
        
        return self._call_ollama(prompt, system_prompt)
    
    def generate_recommendations(self, prediction_result: Dict[str, Any],
                               shap_explanation: Dict[str, Any],
                               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured recommendations based on prediction and SHAP analysis."""
        
        system_prompt = "You are a procurement consultant. Give brief, specific recommendations."
        
        risk_level = prediction_result.get('prediction', 'unknown')
        top_negative = shap_explanation.get('top_negative', {})
        
        prompt = f"""{risk_level.upper()} risk supplier in {input_data.get('industry', 'Unknown')} industry.
Main issues: {', '.join(list(top_negative.keys())[:2])}

Give 3 actionable recommendations:
1.
2.
3."""
        
        response = self._call_ollama(prompt, system_prompt)
        
        # Parse recommendations into structured format
        recommendations = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Clean up the line
                clean_line = line.lstrip('0123456789.- ').strip()
                if clean_line:
                    recommendations.append(clean_line)
        
        return {
            'recommendations': recommendations,
            'full_response': response,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def chat_with_context(self, user_message: str, session_id: str,
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Handle conversational chat with context awareness."""
        
        system_prompt = "You are a procurement advisor. Give helpful, brief responses about supplier risk management."
        
        # Build context-aware prompt
        context_info = ""
        if context:
            if 'prediction' in context:
                pred = context['prediction']
                context_info += f"\nCurrent Prediction Context: {pred.get('prediction', 'N/A')} risk (confidence: {pred.get('confidence', 0):.1%})"
            
            if 'shap_explanation' in context:
                shap = context['shap_explanation']
                top_factors = list(shap.get('top_negative', {}).keys())[:2]
                if top_factors:
                    context_info += f"\nKey Risk Factors: {', '.join(top_factors)}"
        
        # Get conversation history
        history = self.conversation_history.get(session_id, [])
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_history = history[-4:]  # Last 4 exchanges
            for exchange in recent_history:
                conversation_context += f"\nUser: {exchange['user']}\nAssistant: {exchange['assistant']}"
        
        full_prompt = f"""
{context_info}

Previous conversation:
{conversation_context}

Current user question: {user_message}

Provide a helpful, contextual response about procurement risk management.
"""
        
        response = self._call_ollama(full_prompt, system_prompt)
        
        # Store conversation
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            'user': user_message,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
        
        return response
    
    def analyze_what_if_scenario(self, original_data: Dict[str, Any],
                                modified_data: Dict[str, Any],
                                original_prediction: Dict[str, Any]) -> str:
        """Analyze what-if scenarios by comparing original and modified inputs."""
        
        system_prompt = """You are a procurement analyst helping with scenario planning. Compare original vs modified supplier data and explain the potential impact on risk assessment."""
        
        # Find key differences
        differences = []
        for key, new_value in modified_data.items():
            original_value = original_data.get(key)
            if original_value != new_value:
                differences.append(f"{key}: {original_value} → {new_value}")
        
        prompt = f"""
Analyze this what-if scenario for procurement risk:

ORIGINAL PREDICTION: {original_prediction.get('prediction', 'unknown')} risk

PROPOSED CHANGES:
{chr(10).join(differences)}

Explain:
1. How these changes might affect the risk assessment
2. Which changes would have the most impact
3. Whether this scenario would likely improve or worsen the risk profile
4. Additional considerations for implementing these changes

Keep response focused and under 200 words.
"""
        
        return self._call_ollama(prompt, system_prompt)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]