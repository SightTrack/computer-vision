#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

def analyze_training_results():
    """Analyze training results and suggest improvements"""
    
    print("🔍 TRAINING ANALYSIS")
    print("=" * 50)
    
    # Load saved model data
    try:
        checkpoint = torch.load('best_model.pth', map_location='cpu')
        print("✅ Loaded training checkpoint")
        
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses'] 
        train_accs = checkpoint['train_accuracies']
        val_accs = checkpoint['val_accuracies']
        best_val_acc = checkpoint['best_val_acc']
        
        print(f"📊 Training Summary:")
        print(f"  • Total epochs trained: {len(train_losses)}")
        print(f"  • Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  • Final training accuracy: {train_accs[-1]:.2f}%")
        print(f"  • Final validation accuracy: {val_accs[-1]:.2f}%")
        print(f"  • Final training loss: {train_losses[-1]:.4f}")
        print(f"  • Final validation loss: {val_losses[-1]:.4f}")
        
    except FileNotFoundError:
        print("❌ No training checkpoint found")
        return
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Analyze training patterns
    print(f"\n🔎 TRAINING PATTERN ANALYSIS:")
    print("=" * 50)
    
    # 1. Check for overfitting
    train_val_gap = train_accs[-1] - val_accs[-1]
    loss_gap = val_losses[-1] - train_losses[-1]
    
    if train_val_gap > 15:
        print("⚠️  SEVERE OVERFITTING DETECTED")
        print(f"   Training-validation accuracy gap: {train_val_gap:.2f}%")
        print("   🔧 Suggested fixes:")
        print("   • Increase dropout (try 0.5-0.7)")
        print("   • Add more data augmentation")
        print("   • Reduce model complexity")
        print("   • Increase weight decay")
        print("   • Enable label smoothing")
    elif train_val_gap > 7:
        print("⚠️  MODERATE OVERFITTING DETECTED") 
        print(f"   Training-validation accuracy gap: {train_val_gap:.2f}%")
        print("   🔧 Suggested fixes:")
        print("   • Increase dropout slightly")
        print("   • Add data augmentation")
        print("   • Increase weight decay")
    else:
        print("✅ No significant overfitting")
    
    # 2. Check for underfitting
    if best_val_acc < 30:
        print("⚠️  SEVERE UNDERFITTING DETECTED")
        print("   🔧 Suggested fixes:")
        print("   • Increase model capacity")
        print("   • Lower learning rate")
        print("   • Train for more epochs")
        print("   • Check data quality")
    elif best_val_acc < 50:
        print("⚠️  POSSIBLE UNDERFITTING")
        print("   🔧 Suggested fixes:")
        print("   • Consider larger model")
        print("   • Adjust learning rate")
        print("   • More training epochs")
    
    # 3. Check learning rate issues
    if len(train_losses) > 10:
        early_loss = np.mean(train_losses[:5])
        late_loss = np.mean(train_losses[-5:])
        loss_reduction = (early_loss - late_loss) / early_loss
        
        if loss_reduction < 0.1:
            print("⚠️  LEARNING RATE TOO LOW")
            print("   Loss barely decreased over training")
            print("   🔧 Suggested fixes:")
            print("   • Increase learning rate")
            print("   • Use different scheduler")
        elif loss_reduction > 0.8 and len(train_losses) < 20:
            print("⚠️  LEARNING RATE TOO HIGH")
            print("   Loss decreased too quickly")
            print("   🔧 Suggested fixes:")
            print("   • Decrease learning rate")
            print("   • Use gradual warmup")
    
    # 4. Check for convergence
    if len(val_accs) > 10:
        recent_improvement = max(val_accs[-10:]) - val_accs[-10]
        if recent_improvement < 1.0:
            print("⚠️  CONVERGENCE PLATEAU")
            print("   Little improvement in recent epochs")
            print("   🔧 Suggested fixes:")
            print("   • Lower learning rate")
            print("   • Different optimizer")
            print("   • Fine-tune pretrained model")
    
    # 5. Data quality issues
    if val_accs[0] > 10:  # Random guess would be ~1/num_classes
        print("⚠️  POSSIBLE DATA LEAKAGE")
        print("   High initial accuracy suggests data issues")
        print("   🔧 Check for:")
        print("   • Duplicate images in train/val")
        print("   • Data preprocessing errors")
    
    return {
        'overfitting': train_val_gap > 7,
        'underfitting': best_val_acc < 50,
        'converged': recent_improvement < 1.0 if len(val_accs) > 10 else False,
        'best_val_acc': best_val_acc,
        'suggestions': generate_suggestions(train_val_gap, best_val_acc, loss_reduction if 'loss_reduction' in locals() else 0)
    }

def generate_suggestions(train_val_gap, best_val_acc, loss_reduction):
    """Generate specific configuration suggestions"""
    suggestions = {}
    
    # Base config improvements
    current_config = {}
    try:
        with open('config.json', 'r') as f:
            current_config = json.load(f)
    except:
        pass
    
    suggested_config = current_config.copy()
    
    # Adjust based on detected issues
    if train_val_gap > 15:  # Severe overfitting
        suggested_config['dropout'] = 0.6
        suggested_config['weight_decay'] = 0.001
        suggested_config['label_smoothing'] = 0.2
        suggested_config['batch_size'] = max(16, current_config.get('batch_size', 32) // 2)
        
    elif train_val_gap > 7:  # Moderate overfitting  
        suggested_config['dropout'] = 0.5
        suggested_config['weight_decay'] = 0.0005
        suggested_config['label_smoothing'] = 0.15
        
    if best_val_acc < 30:  # Severe underfitting
        suggested_config['learning_rate'] = 0.0005
        suggested_config['backbone'] = 'efficientnet_v2_m'  # Larger model
        suggested_config['num_epochs'] = 150
        
    elif best_val_acc < 50:  # Moderate underfitting
        suggested_config['learning_rate'] = 0.0002
        suggested_config['num_epochs'] = 120
    
    if loss_reduction < 0.1:  # Learning rate too low
        suggested_config['learning_rate'] = current_config.get('learning_rate', 0.0001) * 2
        
    elif loss_reduction > 0.8:  # Learning rate too high
        suggested_config['learning_rate'] = current_config.get('learning_rate', 0.0001) * 0.5
    
    return suggested_config

def create_improved_config():
    """Create an improved configuration based on analysis"""
    
    analysis = analyze_training_results()
    if analysis and 'suggestions' in analysis:
        
        print(f"\n💡 SUGGESTED IMPROVEMENTS:")
        print("=" * 50)
        
        suggested_config = analysis['suggestions']
        
        # Save improved config
        with open('improved_config_v2.json', 'w') as f:
            json.dump(suggested_config, f, indent=2)
        
        print("✅ Created improved_config_v2.json with suggested improvements:")
        for key, value in suggested_config.items():
            print(f"  • {key}: {value}")
        
        return suggested_config
    
    return None

if __name__ == "__main__":
    analysis = analyze_training_results()
    improved_config = create_improved_config()
    
    print(f"\n🚀 NEXT STEPS:")
    print("=" * 50)
    print("1. Review the analysis above")
    print("2. Use improved_config_v2.json for next training run")
    print("3. Monitor training curves more carefully")
    print("4. Consider implementing mixup/cutmix if overfitting persists") 