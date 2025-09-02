#!/usr/bin/env python3
"""
Baseline ParticleTransformer Learning Rate Sweep
===============================================

This script performs a learning rate sweep on the baseline ParticleTransformer
to find the optimal learning rate for hyperparameter transfer to μP models.

Usage:
    python run_baseline_lr_sweep.py

The script will:
1. Test multiple learning rates on the baseline model
2. Identify the optimal learning rate
3. Save results for comparison with μP models
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import argparse

# Add networks to path
sys.path.append('networks')

# Import the baseline model
from example_ParticleTransformer_hyperparam_transfer import get_model, get_loss

def create_dummy_data(batch_size=32, seq_len=100, input_dim=7):
    """Create dummy data for testing"""
    # Create dummy input data
    points = torch.randn(batch_size, seq_len, input_dim)
    features = torch.randn(batch_size, seq_len, input_dim) 
    lorentz_vectors = torch.randn(batch_size, seq_len, 4)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Create dummy labels (5 classes for JetClass)
    labels = torch.randint(0, 5, (batch_size,))
    
    return (points, features, lorentz_vectors, mask), labels

def train_epoch(model, dataloader, optimizer, criterion, device, max_batches=10):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Move data to device
        data = [d.to(device) for d in data]
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(*data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Break after max_batches for quick testing
        if batch_idx >= max_batches - 1:
            break
    
    return total_loss / num_batches

def run_lr_sweep(learning_rates, batch_size=512, epochs=5, max_batches=10, results_file=None):
    """Run learning rate sweep on baseline model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 BASELINE LEARNING RATE SWEEP")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("")
    
    # Model configuration
    config = {
        'input_dim': 7,
        'num_classes': 5,
        'embed_dims': [128, 128, 128, 128, 128, 128, 128, 128],
        'ffn_ratios': [4, 4, 4, 4, 4, 4, 4, 4],
        'num_heads': [8, 8, 8, 8, 8, 8, 8, 8],
        'num_layers': 8,
        'dropout': 0.1,
        'activation': 'gelu'
    }
    
    print("📊 SWEEP CONFIGURATION:")
    print(f"• Learning rates: {learning_rates}")
    print(f"• Batch size: {batch_size}")
    print(f"• Epochs: {epochs}")
    print(f"• Max batches per epoch: {max_batches}")
    print("")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/baseline_lr_sweep_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    if results_file is None:
        results_file = os.path.join(results_dir, "baseline_lr_sweep_results.json")
    
    all_results = []
    
    # Run learning rate sweep
    for i, lr in enumerate(learning_rates):
        print(f"🔬 Testing learning rate: {lr} ({i+1}/{len(learning_rates)})")
        print("-" * 50)
        
        try:
            # Create model
            model = get_model(config)
            model = model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Create dummy dataset
            dummy_data = []
            dummy_labels = []
            
            for _ in range(50):  # 50 batches
                data, labels = create_dummy_data(batch_size=batch_size)
                dummy_data.append(data)
                dummy_labels.append(labels)
            
            # Create dataloader
            dataset = list(zip(dummy_data, dummy_labels))
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Training setup
            criterion = get_loss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            
            # Training loop
            start_time = time.time()
            best_loss = float('inf')
            training_losses = []
            
            for epoch in range(epochs):
                epoch_start = time.time()
                avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, max_batches)
                epoch_time = time.time() - epoch_start
                
                training_losses.append(avg_loss)
                
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, Time = {epoch_time:.2f}s")
                
                # Track best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
            
            total_time = time.time() - start_time
            
            # Check for NaN or infinite values
            has_nan = not all(torch.isfinite(torch.tensor(training_losses)))
            
            # Results
            result = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': epochs,
                'max_batches': max_batches,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'final_loss': training_losses[-1],
                'best_loss': best_loss,
                'training_losses': training_losses,
                'total_time': total_time,
                'avg_epoch_time': total_time / epochs,
                'has_nan': has_nan,
                'timestamp': datetime.now().isoformat(),
                'model_name': 'baseline_hyperparam_transfer'
            }
            
            all_results.append(result)
            
            if has_nan:
                print(f"  ❌ FAILED: NaN or infinite values detected")
            else:
                print(f"  ✅ SUCCESS: Final loss = {result['final_loss']:.6f}")
            
            print(f"  📊 Best loss: {result['best_loss']:.6f}")
            print(f"  ⏱️  Total time: {result['total_time']:.2f}s")
            print("")
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            print("")
            
            # Add failed result
            result = {
                'learning_rate': lr,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'model_name': 'baseline_hyperparam_transfer'
            }
            all_results.append(result)
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Analyze results
    analyze_results(all_results)
    
    return all_results

def analyze_results(results):
    """Analyze and display results"""
    print("")
    print("📊 BASELINE LEARNING RATE SWEEP ANALYSIS")
    print("=" * 60)
    print()
    
    # Filter successful results
    successful_results = [r for r in results if 'final_loss' in r and not r.get('has_nan', False)]
    
    if not successful_results:
        print("❌ No successful training runs found!")
        return
    
    # Sort by learning rate
    successful_results.sort(key=lambda x: x['learning_rate'])
    
    print("📋 RESULTS SUMMARY:")
    print("-" * 90)
    print(f"{'LR':<10} {'Final Loss':<12} {'Best Loss':<12} {'Time (s)':<10} {'Status':<15}")
    print("-" * 90)
    
    best_lr = None
    best_loss = float('inf')
    
    for r in successful_results:
        lr = r['learning_rate']
        final_loss = r['final_loss']
        best_epoch_loss = r['best_loss']
        total_time = r['total_time']
        
        # Determine status
        if final_loss < 1.0 and final_loss > 0.001:
            status = "✅ Good"
        elif final_loss >= 1.0:
            status = "⚠️  High Loss"
        elif final_loss <= 0.001:
            status = "❌ Too Low"
        else:
            status = "❓ Unknown"
        
        print(f"{lr:<10.4f} {final_loss:<12.6f} {best_epoch_loss:<12.6f} {total_time:<10.1f} {status:<15}")
        
        # Track best learning rate
        if final_loss < best_loss and final_loss > 0.001:
            best_loss = final_loss
            best_lr = lr
    
    print("-" * 90)
    print()
    
    if best_lr:
        print(f"🎯 RECOMMENDED LEARNING RATE: {best_lr}")
        print(f"   • Achieved loss: {best_loss:.6f}")
        print(f"   • This LR can be transferred to μP models!")
        print()
        print("💡 NEXT STEPS:")
        print("   1. Use this learning rate for μP model training")
        print("   2. Run μP learning rate sweep to verify transfer")
        print("   3. Compare baseline vs μP performance")
    else:
        print("⚠️  No stable learning rate found - may need to adjust range")
    
    print()
    print("🔄 To run μP comparison:")
    print(f"   python run_mup_only_lr_sweep.py --lr {best_lr if best_lr else 0.001}")

def main():
    parser = argparse.ArgumentParser(description='Baseline ParticleTransformer Learning Rate Sweep')
    parser.add_argument('--learning-rates', nargs='+', type=float, 
                       default=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                       help='Learning rates to test')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs per learning rate')
    parser.add_argument('--max-batches', type=int, default=10,
                       help='Maximum batches per epoch (for quick testing)')
    parser.add_argument('--results-file', type=str, default=None,
                       help='Output results file path')
    
    args = parser.parse_args()
    
    # Run the sweep
    results = run_lr_sweep(
        learning_rates=args.learning_rates,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_batches=args.max_batches,
        results_file=args.results_file
    )
    
    return results

if __name__ == "__main__":
    main()
