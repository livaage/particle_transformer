#!/bin/bash
#SBATCH --job-name=baseline_lr_sweep
#SBATCH --account=pi-lgrandi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_lr_sweep_%j.out
#SBATCH --error=logs/baseline_lr_sweep_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules and activate environment
source ~/.bashrc
micromamba activate particle_transformer

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 BASELINE PARTICLE TRANSFORMER LEARNING RATE SWEEP"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo ""

# Learning rates to sweep
LEARNING_RATES=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.1)

# Baseline model configuration
BATCH_SIZE=512
EPOCHS=5
MODEL_NAME="baseline_hyperparam_transfer"

echo "📊 SWEEP CONFIGURATION:"
echo "• Learning rates: ${LEARNING_RATES[*]}"
echo "• Batch size: $BATCH_SIZE"
echo "• Epochs: $EPOCHS"
echo "• Model: $MODEL_NAME"
echo ""

# Create results directory
RESULTS_DIR="results/baseline_lr_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Initialize results file
RESULTS_FILE="$RESULTS_DIR/baseline_lr_sweep_results.json"
echo "[]" > "$RESULTS_FILE"

echo "📁 Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run training for a given learning rate
run_training() {
    local lr=$1
    local job_id=$2
    
    echo "🔬 Testing learning rate: $lr"
    echo "----------------------------------------"
    
    # Create temporary training script
    cat > "temp_baseline_train_${job_id}.py" << EOF
#!/usr/bin/env python3
"""
Temporary training script for baseline model with learning rate: $lr
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

def train_epoch(model, dataloader, optimizer, criterion, device):
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
        
        # Break after a few batches for quick testing
        if batch_idx >= 10:
            break
    
    return total_loss / num_batches

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create model
    print("Creating baseline model...")
    model = get_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dummy dataset
    print("Creating dummy dataset...")
    dummy_data = []
    dummy_labels = []
    
    for _ in range(100):  # 100 batches
        data, labels = create_dummy_data(batch_size=$BATCH_SIZE)
        dummy_data.append(data)
        dummy_labels.append(labels)
    
    # Create dataloader
    dataset = list(zip(dummy_data, dummy_labels))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Training setup
    criterion = get_loss()
    optimizer = optim.AdamW(model.parameters(), lr=$lr, weight_decay=0.01)
    
    print(f"Starting training with learning rate: $lr")
    print(f"Batch size: $BATCH_SIZE")
    print(f"Epochs: $EPOCHS")
    print("")
    
    # Training loop
    start_time = time.time()
    best_loss = float('inf')
    training_losses = []
    
    for epoch in range($EPOCHS):
        epoch_start = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        epoch_time = time.time() - epoch_start
        
        training_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss = {avg_loss:.6f}, Time = {epoch_time:.2f}s")
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    total_time = time.time() - start_time
    
    # Results
    results = {
        'learning_rate': $lr,
        'batch_size': $BATCH_SIZE,
        'epochs': $EPOCHS,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'final_loss': training_losses[-1],
        'best_loss': best_loss,
        'training_losses': training_losses,
        'total_time': total_time,
        'avg_epoch_time': total_time / $EPOCHS,
        'timestamp': datetime.now().isoformat(),
        'model_name': '$MODEL_NAME'
    }
    
    print("")
    print("📊 TRAINING RESULTS:")
    print(f"• Final loss: {results['final_loss']:.6f}")
    print(f"• Best loss: {results['best_loss']:.6f}")
    print(f"• Total time: {results['total_time']:.2f}s")
    print(f"• Avg epoch time: {results['avg_epoch_time']:.2f}s")
    
    # Save results
    results_file = '$RESULTS_FILE'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(results)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")
    
    # Check for NaN or infinite values
    if not all(torch.isfinite(torch.tensor(training_losses))):
        print("⚠️  WARNING: NaN or infinite values detected in training!")
        return False
    else:
        print("✅ Training completed successfully (no NaN values)")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

    # Run the training
    python "temp_baseline_train_${job_id}.py"
    local exit_code=$?
    
    # Clean up temporary file
    rm -f "temp_baseline_train_${job_id}.py"
    
    return $exit_code
}

# Run learning rate sweep
echo "🎯 STARTING LEARNING RATE SWEEP"
echo "================================"

successful_runs=0
failed_runs=0

for i in "${!LEARNING_RATES[@]}"; do
    lr="${LEARNING_RATES[$i]}"
    job_id="${SLURM_JOB_ID}_${i}"
    
    echo ""
    echo "📈 Progress: $((i+1))/${#LEARNING_RATES[@]} learning rates"
    
    if run_training "$lr" "$job_id"; then
        echo "✅ Learning rate $lr: SUCCESS"
        ((successful_runs++))
    else
        echo "❌ Learning rate $lr: FAILED"
        ((failed_runs++))
    fi
    
    echo ""
done

# Summary
echo "🏁 LEARNING RATE SWEEP COMPLETE"
echo "================================"
echo "• Successful runs: $successful_runs"
echo "• Failed runs: $failed_runs"
echo "• Total learning rates tested: ${#LEARNING_RATES[@]}"
echo ""

# Analyze results
if [ -f "$RESULTS_FILE" ]; then
    echo "📊 ANALYZING RESULTS..."
    echo "======================="
    
    # Create analysis script
    cat > "temp_analyze_results.py" << 'EOF'
import json
import sys

def analyze_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No results found!")
        return
    
    print(f"📈 BASELINE LEARNING RATE SWEEP ANALYSIS")
    print("=" * 50)
    print()
    
    # Sort by learning rate
    results.sort(key=lambda x: x['learning_rate'])
    
    print("📋 RESULTS SUMMARY:")
    print("-" * 80)
    print(f"{'LR':<10} {'Final Loss':<12} {'Best Loss':<12} {'Time (s)':<10} {'Status':<10}")
    print("-" * 80)
    
    best_lr = None
    best_loss = float('inf')
    
    for r in results:
        lr = r['learning_rate']
        final_loss = r['final_loss']
        best_epoch_loss = r['best_loss']
        total_time = r['total_time']
        
        # Determine status
        if final_loss < 1.0 and final_loss > 0.001:
            status = "✅ Good"
        elif final_loss >= 1.0:
            status = "⚠️  High"
        elif final_loss <= 0.001:
            status = "❌ NaN/Inf"
        else:
            status = "❓ Unknown"
        
        print(f"{lr:<10.4f} {final_loss:<12.6f} {best_epoch_loss:<12.6f} {total_time:<10.1f} {status:<10}")
        
        # Track best learning rate
        if final_loss < best_loss and final_loss > 0.001:
            best_loss = final_loss
            best_lr = lr
    
    print("-" * 80)
    print()
    
    if best_lr:
        print(f"🎯 RECOMMENDED LEARNING RATE: {best_lr}")
        print(f"   • Achieved loss: {best_loss:.6f}")
        print(f"   • This LR can be transferred to μP models!")
    else:
        print("⚠️  No stable learning rate found - may need to adjust range")
    
    print()
    print("💡 NEXT STEPS:")
    print("   1. Use the recommended learning rate for μP model training")
    print("   2. Run μP learning rate sweep to verify transfer")
    print("   3. Compare baseline vs μP performance")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file>")
        sys.exit(1)
    
    analyze_results(sys.argv[1])
EOF
    
    python temp_analyze_results.py "$RESULTS_FILE"
    rm -f temp_analyze_results.py
    
    echo ""
    echo "📁 Full results saved to: $RESULTS_FILE"
else
    echo "❌ No results file found!"
fi

echo ""
echo "🏁 Job completed at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
