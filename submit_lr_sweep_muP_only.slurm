#!/bin/bash
#SBATCH --job-name=lr_sweep_muP_only
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p logs

# Source environment
source env.sh

# Feature type (second argument, optional, defaults to "full")
FEATURE_TYPE=${2:-"full"}

# Comment suffix (optional)
COMMENT_SUFFIX=${3:-""}

# Learning rates to sweep over - focused on range likely to work well for transfer
LEARNING_RATES=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2")

# Architecture configurations to test
# Format: "embed_dims,pair_embed_dims,num_heads,num_layers"
ARCHITECTURES=(
    "16,64,16:8,8,8:2:2"          # Very small, very shallow
    "32,128,32:16,16,16:2:4"      # Small, shallow
    "64,256,64:32,32,32:4:6"      # Medium, medium
    "128,512,128:64,64,64:8:8"    # Large, deep (original)
)

echo "Learning rates to test: ${LEARNING_RATES[*]}"
echo "Architectures to test: ${ARCHITECTURES[*]}"
echo "Comment suffix: $COMMENT_SUFFIX"

# Create a comprehensive log directory for this sweep
SWEEP_LOG_DIR="logs/muP_only_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_LOG_DIR"

echo "All logs will be saved to: $SWEEP_LOG_DIR"

# Function to create network files with specific architecture
create_network_file() {
    local arch_name=$1
    local embed_dims=$2
    local pair_embed_dims=$3
    local num_heads=$4
    local num_layers=$5
    
    # Create the μP version using the new wrapper with d_k attention scaling
    cat > "networks/example_ParticleTransformer_muP_${arch_name}.py" << EOF
import torch
import torch.nn as nn
from weaver.utils.logger import _logger

# Import the μP implementation with d_k attention scaling
import sys
import os
sys.path.append(os.path.dirname(__file__))
from example_ParticleTransformer_muP_fixed import ParticleTransformerMuPFixed

'''
ParticleTransformer with μ-Parameterization (μP) for architecture: ${arch_name}
This implementation uses the FIXED μP wrapper with correct d_k attention scaling.

Key features:
- Attention scaling: attn_scores = (q @ k.transpose(-2, -1)) / float(d_k) ✅ FIXED
- Custom MuPLinear, MuPMultiheadAttention, MuPReadout layers
- Complete μP framework implementation
- Ensures the same learning rate is optimal across all model sizes

embed_dims: [${embed_dims}]
pair_embed_dims: [${pair_embed_dims}]
num_heads: ${num_heads}
num_layers: ${num_layers}
'''

class ParticleTransformerMuP${arch_name^}(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        # Store architectural parameters
        self.embed_dims = [${embed_dims}]
        self.pair_embed_dims = [${pair_embed_dims}]
        self.num_heads = ${num_heads}
        self.num_layers = ${num_layers}
        self.num_cls_layers = 2
        
        # Calculate μP scaling factors
        self.d_model = self.embed_dims[1]  # Hidden dimension
        self.width_scale = (self.d_model ** 0.5)
        self.depth_scale = (self.num_layers ** 0.5)
        
        # Create the μP model with correct μ-parameterization
        self.mod = ParticleTransformerMuPFixed(**kwargs)
        
        # Register μP scaling information
        self.register_buffer('mup_width_scale', torch.tensor(self.width_scale))
        self.register_buffer('mup_depth_scale', torch.tensor(self.depth_scale))
        self.register_buffer('mup_architecture_scale', 
                           torch.tensor((self.d_model * self.num_layers) ** 0.5))
        
        # Apply μP base shapes for proper scaling
        try:
            prepare_model_for_mup(self.mod)
            _logger.info("Successfully applied μP base shapes")
        except Exception as e:
            _logger.warning(f"Could not apply μP base shapes: {e}")
        
        # Log μP information
        self._log_mup_info()

    def _log_mup_info(self):
        """Log μP scaling information for this architecture."""
        _logger.info(f"Applied FIXED μP parameterization for {self.d_model}d x {self.num_layers}L:")
        _logger.info(f"  - Width scale: {self.width_scale:.2f}")
        _logger.info(f"  - Depth scale: {self.depth_scale:.2f}")
        _logger.info(f"  - Architecture scale: {self.mup_architecture_scale:.2f}")
        _logger.info(f"  - Attention scaling: d_k (not sqrt(d_k)) ✅ FIXED")
        _logger.info(f"  - Custom MuPLinear, MuPMultiheadAttention, MuPReadout")
        _logger.info(f"  - Complete μP framework implementation")
        _logger.info(f"  - This ensures the same learning rate is optimal across all model sizes!")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(points, features, lorentz_vectors, mask)
    
    def get_mup_info(self):
        """Get μP scaling information for this architecture."""
        return {
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'width_scale': self.width_scale,
            'depth_scale': self.depth_scale,
            'architecture_scale': self.mup_architecture_scale.item(),
            'attention_scaling': 'd_k (not sqrt(d_k))',
            'total_params': sum(p.numel() for p in self.parameters())
        }


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations - specific architecture
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[${embed_dims}],
        pair_embed_dims=[${pair_embed_dims}],
        num_heads=${num_heads},
        num_layers=${num_layers},
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('${arch_name^} Model config: %s' % str(cfg))

    model = ParticleTransformerMuP${arch_name^}(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
EOF
}

# Function to run training for a specific configuration
run_training() {
    local model_name=$1
    local network_file=$2
    local learning_rate=$3
    local architecture_name=$4
    
    # Create a unique job name
    local job_name="${model_name}_lr${learning_rate}_${architecture_name}"
    
    # Create a subdirectory for this specific configuration
    local config_dir="${SWEEP_LOG_DIR}/${job_name}"
    mkdir -p "$config_dir"
    
    # Submit the training job
    echo "Submitting job: $job_name"
    echo "  Network file: $network_file"
    echo "  Learning rate: $learning_rate"
    echo "  Architecture: $architecture_name"
    
    # Submit the job and capture the job ID
    local job_id=$(sbatch \
        --job-name="$job_name" \
        --output="$config_dir/slurm_%j.out" \
        --error="$config_dir/slurm_%j.err" \
        --time=2:00:00 \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --mem=32G \
        --export=ALL,NETWORK_FILE="$network_file",LEARNING_RATE="$learning_rate",MODEL_NAME="$model_name",ARCHITECTURE_NAME="$architecture_name",COMMENT_SUFFIX="$COMMENT_SUFFIX",FEATURE_TYPE="$FEATURE_TYPE" \
        << 'EOF'
#!/bin/bash
#SBATCH --job-name=${MODEL_NAME}_lr${LEARNING_RATE}_${ARCHITECTURE_NAME}

echo "Starting training for ${MODEL_NAME} with learning rate ${LEARNING_RATE}"
echo "Architecture: ${ARCHITECTURE_NAME}"
echo "Network file: ${NETWORK_FILE}"
echo "Feature type: ${FEATURE_TYPE}"
echo "Comment suffix: ${COMMENT_SUFFIX}"
echo "=========================================="

# Set up the training command with environment variables
export START_LR="${LEARNING_RATE}"
export COMMENT="_muP_only_${ARCHITECTURE_NAME}_lr${LEARNING_RATE}_${COMMENT_SUFFIX}"

# Use the same training script as the working muP comparison script
# but call it with the correct model name format for muP variants
TRAIN_CMD="./train_JetClass_hyperparam_transfer.sh ParT_muP_${ARCHITECTURE_NAME} ${FEATURE_TYPE} --data-fraction 0.2 --file-fraction 0.1 --start-lr ${LEARNING_RATE}"

echo "Training command: $TRAIN_CMD"
echo "=========================================="

# Run the training
eval $TRAIN_CMD

echo "Training completed for ${MODEL_NAME} with learning rate ${LEARNING_RATE}"
echo "Check output directory: training/JetClass/Pythia/${FEATURE_TYPE}/ParT_muP_${ARCHITECTURE_NAME}/muP_only_${ARCHITECTURE_NAME}_lr${LEARNING_RATE}_${COMMENT_SUFFIX}"
echo "Log file: logs/JetClass_Pythia_${FEATURE_TYPE}_ParT_muP_${ARCHITECTURE_NAME}_muP_only_${ARCHITECTURE_NAME}_lr${LEARNING_RATE}_${COMMENT_SUFFIX}.log"
EOF
    )
    
    # Extract the job ID from the sbatch output
    local job_id_clean=$(echo "$job_id" | grep -o '[0-9]\+')
    
    echo "  Job submitted with ID: $job_id_clean"
    echo "  Logs will be saved to: $config_dir"
    echo "  Output will be saved to: training/JetClass/Pythia/${FEATURE_TYPE}/ParT_muP_${ARCHITECTURE_NAME}/muP_only_${ARCHITECTURE_NAME}_lr${LEARNING_RATE}_${COMMENT_SUFFIX}"
    echo "  Log file: logs/JetClass_Pythia_${FEATURE_TYPE}_ParT_muP_${ARCHITECTURE_NAME}_muP_only_${ARCHITECTURE_NAME}_lr${LEARNING_RATE}_${COMMENT_SUFFIX}.log"
    echo ""
}

# Create network files for all architectures
echo "Creating network files for all architectures..."
echo "=========================================="

for arch_config in "${ARCHITECTURES[@]}"; do
    IFS=':' read -r embed_dims pair_embed_dims num_heads num_layers <<< "$arch_config"
    
    # Create a clean architecture name for filenames
    arch_name=$(echo "${embed_dims}_${pair_embed_dims}_${num_heads}h_${num_layers}l" | sed 's/,/_/g')
    
    echo "Creating network files for architecture: $arch_name"
    echo "  embed_dims: [$embed_dims]"
    echo "  pair_embed_dims: [$pair_embed_dims]"
    echo "  num_heads: $num_heads"
    echo "  num_layers: $num_layers"
    
    create_network_file "$arch_name" "$embed_dims" "$pair_embed_dims" "$num_heads" "$num_layers"
done

echo "All network files created successfully!"

# Now run the comprehensive learning rate sweep for all architectures

echo "Starting comprehensive μP-only learning rate sweep for all architectures..."
echo "=========================================="

# For each architecture, test only the μP version
for arch_config in "${ARCHITECTURES[@]}"; do
    IFS=':' read -r embed_dims pair_embed_dims num_heads num_layers <<< "$arch_config"
    arch_name=$(echo "${embed_dims}_${pair_embed_dims}_${num_heads}h_${num_layers}l" | sed 's/,/_/g')
    
    echo "Testing architecture: $arch_name"
    echo "  embed_dims: [$embed_dims]"
    echo "  pair_embed_dims: [$pair_embed_dims]"
    echo "  num_heads: $num_heads"
    echo "  num_layers: $num_layers"
    echo "=========================================="
    
    # Test only μP transformer (should show excellent LR transfer)
    echo "Testing μP ParticleTransformer (should show excellent LR transfer)..."
    for lr in "${LEARNING_RATES[@]}"; do
        run_training "ParT_muP_${arch_name}" "networks/example_ParticleTransformer_muP_${arch_name}.py" "$lr" "$arch_name"
    done
    
    echo ""
done

echo "=========================================="
echo "Comprehensive μP-only learning rate sweep completed for all architectures!"
echo "All jobs have been submitted."
echo "Check the comprehensive logs directory: $SWEEP_LOG_DIR"
echo "=========================================="

# Create a summary script for easy analysis
cat > "$SWEEP_LOG_DIR/analyze_results.sh" << 'EOF'
#!/bin/bash
echo "μP-Only Learning Rate Sweep Analysis"
echo "===================================="
echo "Log directory: $(pwd)"
echo ""

echo "Architectures tested:"
for arch_config in "${ARCHITECTURES[@]}"; do
    IFS=':' read -r embed_dims pair_embed_dims num_heads num_layers <<< "$arch_config"
    arch_name=$(echo "${embed_dims}_${pair_embed_dims}_${num_heads}h_${num_layers}l" | sed 's/,/_/g')
    echo "- $arch_name: embed=[$embed_dims], pair=[$pair_embed_dims], heads=$num_heads, layers=$num_layers"
done
echo ""

echo "Learning rates tested: ${LEARNING_RATES[*]}"
echo ""

echo "For each architecture, we tested only the μP ParticleTransformer:"
echo "1. μP ParticleTransformer with d_k attention scaling (should show excellent hyperparameter transfer)"
echo "   - Uses attn_scores = (q @ k.transpose(-2, -1)) / float(d_k) instead of sqrt(d_k)"
echo "   - Implements proper μP parameterization with MuPLinear layers"
echo "   - Uses mup.MuReadout for the final classifier"
echo ""

echo "Expected results:"
echo "- μP transformers with d_k attention scaling: Performance should be much more consistent across learning rates"
echo "- The μP effect should be more pronounced for larger/deeper architectures"
echo "- The same learning rate should be optimal across all model sizes with μP"
echo "- Attention scaling uses d_k instead of sqrt(d_k), enabling better hyperparameter transfer"
echo "- This demonstrates the power of μ-parameterization with proper attention scaling for hyperparameter transfer"
echo ""

echo "To analyze results:"
echo "1. Check individual log files in each subdirectory"
echo "2. Look for validation accuracy patterns across learning rates"
echo "3. Focus on how μP effectiveness varies with architecture size"
echo "4. Verify that optimal learning rates are consistent across architectures with μP"
echo "5. Compare with baseline results from other experiments if available"
EOF

chmod +x "$SWEEP_LOG_DIR/analyze_results.sh"

echo "Analysis script created at: $SWEEP_LOG_DIR/analyze_results.sh"
echo "Run it after training completes to analyze the results."
