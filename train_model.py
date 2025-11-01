#!/usr/bin/env python3
"""
Main training script for the Procurement Risk Prediction Platform.
Run this script to train the XGBoost model with the provided data.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.model_training import train_and_evaluate_model
import config


def main():
    """Main training function."""
    print("=" * 60)
    print("PROCUREMENT RISK PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    print(f"Data source: {config.MERGED_TRAINING_DATA}")
    print(f"Model will be saved to: {config.MODEL_PATH}")
    print()
    
    try:
        # Train and evaluate model
        result = train_and_evaluate_model()
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        metrics = result['metrics']
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
        print(f"Final F1 Score: {metrics['f1_score']:.4f}")
        
        print("\nClassification Report:")
        for class_name, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict):
                print(f"  {class_name}:")
                print(f"    Precision: {class_metrics.get('precision', 0):.4f}")
                print(f"    Recall: {class_metrics.get('recall', 0):.4f}")
                print(f"    F1-Score: {class_metrics.get('f1-score', 0):.4f}")
        
        print(f"\nModel artifacts saved successfully!")
        print(f"✓ Model: {config.MODEL_PATH}")
        print(f"✓ Scaler: {config.SCALER_PATH}")
        print(f"✓ Feature names: {config.MODELS_DIR}/feature_names.json")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()