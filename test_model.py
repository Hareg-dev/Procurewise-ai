#!/usr/bin/env python3
"""
Test script to verify the trained model works correctly.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.model_training import predict_risk


def test_single_prediction():
    """Test single prediction functionality."""
    
    # Sample input data
    test_input = {
        'annual_revenue': 1000000,
        'annual_spend': 500000,
        'avg_payment_delay_days': 15,
        'contract_value': 250000,
        'contract_duration_months': 12,
        'past_disputes': 2,
        'delivery_score': 85,
        'financial_stability_index': 0.75,
        'relationship_years': 5,
        'credit_score': 720,
        'txn_count': 10,
        'avg_txn_amount': 25000,
        'avg_delay': 12,
        'late_ratio': 0.2,
        'dispute_rate': 0.1,
        'avg_delivery_quality': 88,
        'clause_risk_score': 25,
        'region': 'North America',
        'industry': 'IT Services',
        'contract_criticality': 'Medium'
    }
    
    print("Testing single prediction...")
    print("Input data:", test_input)
    print()
    
    try:
        result = predict_risk(test_input)
        
        print("Prediction Results:")
        print(f"Risk Level: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("Probabilities:")
        for risk_level, prob in result['probabilities'].items():
            print(f"  {risk_level}: {prob:.4f}")
        
        print("\n✅ Single prediction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Single prediction test failed: {str(e)}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("PROCUREMENT RISK PREDICTION - MODEL TESTING")
    print("=" * 60)
    
    success = test_single_prediction()
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED! 🎉")
        print("The model is ready for integration.")
    else:
        print("TESTS FAILED! ❌")
        print("Please check the model training and preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()