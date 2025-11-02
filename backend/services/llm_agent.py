"""
LLM agent service using Phi3:mini via Ollama for procurement risk insights.
Refactored to use LangChain for calling Phi3:Mini model.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import config

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage

class ProcurementLLMAgent:
    """LLM agent for procurement risk analysis and recommendations."""
    
    def __init__(self):
        self.ollama_url = config.OLLAMA_BASE_URL
        self.model_name = config.OLLAMA_MODEL
        self.conversation_history = {}

        # Initialize LangChain Ollama LLM optimized for TinyLlama
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_url,
            temperature=0.8,
            num_ctx=2048,  # Appropriate context for TinyLlama
            num_predict=512,  # Reasonable prediction length for TinyLlama
            top_p=0.95,
            repeat_penalty=1.05,
            timeout=config.OLLAMA_TIMEOUT
        )
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make API call to Ollama via LangChain."""
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Use invoke with TinyLlama-optimized parameters
            result = self.llm.invoke(
                messages,
                config={
                    "configurable": {
                        "num_predict": 512,  # Reasonable length for TinyLlama
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "repeat_penalty": 1.05
                    }
                }
            )
            
            # LangChain returns a message object with .content
            response = result.content if hasattr(result, 'content') else str(result)
            
            # Ensure we have a complete response
            if not response or len(response.strip()) < 10:
                raise Exception("Received incomplete or empty response from model")
            
            return response.strip()
        
        except Exception as e:
            err_msg = str(e).lower()
            if 'memory' in err_msg or 'gpu' in err_msg:
                raise Exception("Insufficient memory to load TinyLlama model. Please free memory or restart Ollama.")
            elif 'connection' in err_msg:
                raise Exception("Cannot connect to Ollama. Ensure Ollama is running and tinyllama:latest model is available.")
            elif 'timeout' in err_msg:
                return "The model is taking too long to respond. Please try again shortly."
            else:
                raise Exception(f"Ollama LangChain call failed: {e}")

    def generate_prediction_explanation(self, prediction_result: Dict[str, Any], 
                                        shap_explanation: Dict[str, Any],
                                        input_data: Dict[str, Any]) -> str:
        """Generate natural language explanation using Phi3 model via LangChain."""
        
        risk_level = prediction_result.get('prediction', 'unknown')
        confidence = prediction_result.get('confidence', 0)
        
        system_prompt = """You are an expert procurement risk analyst. Explain the risk assessment results in clear, actionable terms. Focus on the key factors driving the risk level and provide specific recommendations."""
        
        # Build prompt
        top_positive = shap_explanation.get('top_positive', {})
        top_negative = shap_explanation.get('top_negative', {})
        
        prompt = f"""SUPPLIER RISK ASSESSMENT EXPLANATION:

RISK LEVEL: {risk_level.upper()} ({confidence:.1%} confidence)

SUPPLIER PROFILE:
- Industry: {input_data.get('industry', 'Unknown')}
- Region: {input_data.get('region', 'Unknown')}
- Annual Revenue: ${input_data.get('annual_revenue', 0):,}
- Contract Value: ${input_data.get('contract_value', 0):,}
- Relationship: {input_data.get('relationship_years', 0)} years

KEY RISK DRIVERS (factors increasing risk):
{chr(10).join([f'- {k.replace("_", " ").title()}: Impact {v:.3f}' for k, v in list(top_negative.items())[:4]])}

RISK MITIGATORS (factors decreasing risk):
{chr(10).join([f'- {k.replace("_", " ").title()}: Impact {v:.3f}' for k, v in list(top_positive.items())[:4]])}

PERFORMANCE INDICATORS:
- Credit Score: {input_data.get('credit_score', 'N/A')}
- Delivery Score: {input_data.get('delivery_score', 'N/A')}/100
- Payment Delays: {input_data.get('avg_payment_delay_days', 0)} days average
- Past Disputes: {input_data.get('past_disputes', 0)}

Provide a comprehensive explanation of this risk assessment, focusing on:
1. Why this supplier received a {risk_level} risk rating
2. The most critical factors to monitor
3. Specific actions to take based on this assessment
4. Early warning signs to watch for

Keep the explanation practical and actionable for procurement professionals."""
        
        try:
            response = self._call_ollama(prompt, system_prompt)
            return response
        except Exception as e:
            raise Exception(f"TinyLlama explanation generation failed: {str(e)}")

    def generate_recommendations(self, prediction_result: Dict[str, Any],
                               shap_explanation: Dict[str, Any],
                               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured recommendations using Phi3 model via LangChain."""
        
        risk_level = prediction_result.get('prediction', 'unknown')
        confidence = prediction_result.get('confidence', 0)
        
        system_prompt = """You are an expert procurement risk consultant. Analyze the supplier data and provide specific, actionable recommendations. Focus on practical steps to mitigate identified risks."""
        
        # Extract key risk factors
        top_negative = shap_explanation.get('top_negative', {})
        top_positive = shap_explanation.get('top_positive', {})
        
        prompt = f"""SUPPLIER RISK ANALYSIS:

Risk Level: {risk_level.upper()} ({confidence:.1%} confidence)
Industry: {input_data.get('industry', 'Unknown')}
Region: {input_data.get('region', 'Unknown')}
Contract Value: ${input_data.get('contract_value', 0):,}
Relationship Duration: {input_data.get('relationship_years', 0)} years

KEY RISK FACTORS (increasing risk):
{chr(10).join([f'- {k}: {v:.3f}' for k, v in list(top_negative.items())[:3]])}

MITIGATING FACTORS (decreasing risk):
{chr(10).join([f'- {k}: {v:.3f}' for k, v in list(top_positive.items())[:3]])}

FINANCIAL METRICS:
- Annual Revenue: ${input_data.get('annual_revenue', 0):,}
- Credit Score: {input_data.get('credit_score', 0)}
- Payment Delays: {input_data.get('avg_payment_delay_days', 0)} days avg
- Dispute Rate: {input_data.get('dispute_rate', 0):.1%}

PERFORMANCE METRICS:
- Delivery Score: {input_data.get('delivery_score', 0)}/100
- Quality Score: {input_data.get('avg_delivery_quality', 0)}/100
- Past Disputes: {input_data.get('past_disputes', 0)}

Provide 4 specific, actionable recommendations to address this supplier's risk profile. Focus on the most impactful risk factors identified above.

1.
2.
3.
4."""
        
        try:
            response = self._call_ollama(prompt, system_prompt)
            
            # Parse recommendations into structured format
            recommendations = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Clean up the line
                    clean_line = line.lstrip('0123456789.- ').strip()
                    if clean_line and len(clean_line) > 10:  # Filter out very short responses
                        recommendations.append(clean_line)
            
            return {
                'recommendations': recommendations,
                'full_response': response,
                'risk_level': risk_level,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'model': self.model_name
            }
            
        except Exception as e:
            raise Exception(f"TinyLlama recommendation generation failed: {str(e)}")

    def chat_with_context(self, user_message: str, session_id: str,
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Handle conversational chat using Phi3 model via LangChain."""
        
        system_prompt = """You are an expert procurement risk management consultant. Provide helpful, specific advice about supplier risk assessment, contract management, and procurement best practices. Use the provided context about recent predictions to give relevant, actionable guidance."""
        
        # Build comprehensive context for Phi3
        context_info = ""
        if context:
            if 'prediction' in context:
                pred = context['prediction']
                context_info += f"\nRECENT PREDICTION CONTEXT:"
                context_info += f"\n- Risk Level: {pred.get('prediction', 'N/A')} ({pred.get('confidence', 0):.1%} confidence)"
                
                if 'input_data' in context:
                    input_data = context['input_data']
                    context_info += f"\n- Industry: {input_data.get('industry', 'Unknown')}"
                    context_info += f"\n- Contract Value: ${input_data.get('contract_value', 0):,}"
                    context_info += f"\n- Credit Score: {input_data.get('credit_score', 'N/A')}"
                    context_info += f"\n- Delivery Score: {input_data.get('delivery_score', 'N/A')}/100"
                    context_info += f"\n- Past Disputes: {input_data.get('past_disputes', 'N/A')}"
            
            if 'shap_explanation' in context:
                shap = context['shap_explanation']
                top_factors = list(shap.get('top_negative', {}).keys())[:3]
                if top_factors:
                    context_info += f"\n- Key Risk Factors: {', '.join(top_factors)}"
        
        # Get conversation history
        history = self.conversation_history.get(session_id, [])
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_history = history[-3:]  # Last 3 exchanges for better context
            for exchange in recent_history:
                conversation_context += f"\nUser: {exchange['user']}"
                conversation_context += f"\nAssistant: {exchange['assistant']}"
        
        # Build comprehensive prompt
        full_prompt = f"""{context_info}

CONVERSATION HISTORY:
{conversation_context}

CURRENT USER QUESTION: {user_message}

Provide a helpful, specific response about procurement risk management. If the user is asking about the recent prediction, use the context provided above. Give actionable advice and practical recommendations."""
        
        try:
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
            
        except Exception as e:
            # Provide fallback response for timeout/memory issues
            if 'timeout' in str(e).lower() or 'memory' in str(e).lower():
                fallback_response = self._generate_fallback_chat_response(user_message, context)
                
                # Store conversation with fallback
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                
                self.conversation_history[session_id].append({
                    'user': user_message,
                    'assistant': fallback_response,
                    'timestamp': datetime.now().isoformat(),
                    'fallback': True
                })
                
                return fallback_response
            raise Exception(f"TinyLlama chat failed: {str(e)}")

    def analyze_what_if_scenario(self, original_data: Dict[str, Any],
                                modified_data: Dict[str, Any],
                                original_prediction: Dict[str, Any]) -> str:
        """Analyze what-if scenarios using Phi3 model via LangChain."""
        
        system_prompt = """You are an expert procurement risk analyst specializing in scenario planning. Analyze how proposed changes to supplier parameters would impact risk assessment. Provide specific, actionable insights."""
        
        # Find key differences
        differences = []
        for key, new_value in modified_data.items():
            original_value = original_data.get(key)
            if original_value != new_value:
                differences.append(f"{key.replace('_', ' ').title()}: {original_value} → {new_value}")
        
        prompt = f"""WHAT-IF SCENARIO ANALYSIS:

ORIGINAL SUPPLIER ASSESSMENT:
- Risk Level: {original_prediction.get('prediction', 'unknown').upper()}
- Confidence: {original_prediction.get('confidence', 0):.1%}
- Industry: {original_data.get('industry', 'Unknown')}
- Current Contract Value: ${original_data.get('contract_value', 0):,}

PROPOSED CHANGES:
{chr(10).join(differences)}

ORIGINAL KEY METRICS:
- Credit Score: {original_data.get('credit_score', 'N/A')}
- Delivery Score: {original_data.get('delivery_score', 'N/A')}/100
- Payment Delays: {original_data.get('avg_payment_delay_days', 0)} days
- Past Disputes: {original_data.get('past_disputes', 0)}
- Financial Stability: {original_data.get('financial_stability_index', 0):.2f}

Analyze this scenario and provide:

1. **Risk Impact Assessment**: How would these changes likely affect the overall risk level?
2. **Critical Changes**: Which specific changes would have the most significant impact on risk?
3. **Risk Direction**: Would this scenario improve or worsen the risk profile?
4. **Implementation Considerations**: What should be monitored if these changes occur?
5. **Recommended Actions**: Specific steps to take based on this scenario.

Provide a comprehensive but concise analysis focused on practical procurement risk management."""
        
        try:
            response = self._call_ollama(prompt, system_prompt)
            return response
        except Exception as e:
            raise Exception(f"TinyLlama what-if analysis failed: {str(e)}")

    def _generate_fallback_chat_response(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate fallback response when Phi3 is unavailable."""
        message_lower = user_message.lower()
        
        # Context-aware responses
        if context and 'prediction' in context:
            pred = context['prediction']
            risk_level = pred.get('prediction', 'unknown')
            confidence = pred.get('confidence', 0)
            
            if any(word in message_lower for word in ['risk', 'prediction', 'assessment']):
                return f"Based on your recent prediction, the supplier has a {risk_level} risk level with {confidence:.1%} confidence. The key factors to monitor include credit score, delivery performance, and payment history. Would you like specific recommendations for managing this risk level?"
        
        # General procurement advice
        if any(word in message_lower for word in ['recommend', 'advice', 'suggest']):
            return "For effective procurement risk management, I recommend: 1) Regular supplier performance reviews, 2) Diversified supplier base, 3) Clear contract terms with penalties, 4) Financial health monitoring. What specific area would you like to focus on?"
        
        if any(word in message_lower for word in ['hello', 'hi', 'help']):
            return "Hello! I'm your procurement risk assistant. I can help you understand supplier risks, interpret predictions, and provide recommendations. What would you like to know about procurement risk management?"
        
        if any(word in message_lower for word in ['credit', 'financial']):
            return "Credit and financial stability are crucial risk factors. Monitor supplier credit scores, payment history, and financial statements regularly. Consider requiring financial guarantees for high-risk suppliers."
        
        if any(word in message_lower for word in ['delivery', 'performance']):
            return "Delivery performance directly impacts operational risk. Track on-time delivery rates, quality scores, and establish clear SLAs with penalties for poor performance. Consider backup suppliers for critical items."
        
        # Default response
        return "I'm here to help with procurement risk management. Due to high system load, I'm providing quick responses. Ask me about supplier risk factors, contract management, or risk mitigation strategies."
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]