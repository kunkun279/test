#!/bin/bash

# HumanMAC Human3.6M MPJPE Evaluation Script
# This script provides convenient commands for running different evaluation tasks

set -e  # Exit on any error

# Default parameters
MODEL_PATH="checkpoints/ckpt_ema_500.pt"
OUTPUT_DIR="./eval_results"
NUM_SAMPLES=50
DEVICE="cuda"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

# Function to create directory if it doesn't exist
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_info "Created directory: $1"
    fi
}

# Function to run tests
run_tests() {
    print_info "Running HumanMAC H36M evaluation tests..."
    
    if python eval_h36m_simple.py --all_tests; then
        print_success "All tests passed!"
        return 0
    else
        print_error "Tests failed!"
        return 1
    fi
}

# Function to run quick evaluation
run_quick_eval() {
    print_info "Running quick baseline evaluation..."
    
    if python eval_h36m_simple.py --quick_eval; then
        print_success "Quick evaluation completed!"
        return 0
    else
        print_error "Quick evaluation failed!"
        return 1
    fi
}

# Function to run full evaluation
run_full_eval() {
    local model_path="$1"
    local output_dir="$2"
    local num_samples="$3"
    local device="$4"
    local actions="$5"
    
    print_info "Running full HumanMAC evaluation..."
    print_info "Model: $model_path"
    print_info "Output: $output_dir"
    print_info "Samples: $num_samples"
    print_info "Device: $device"
    
    # Check if model exists
    if ! check_file "$model_path"; then
        return 1
    fi
    
    # Create output directory
    ensure_dir "$output_dir"
    
    # Build command
    local cmd="python eval_h36m_mpjpe_fixed.py --model_path $model_path --output_dir $output_dir --num_samples $num_samples --device $device"
    
    if [ -n "$actions" ]; then
        cmd="$cmd --actions $actions"
        print_info "Actions: $actions"
    else
        print_info "Actions: All"
    fi
    
    # Run evaluation
    if eval "$cmd"; then
        print_success "Full evaluation completed!"
        print_info "Results saved to: $output_dir/h36m_mpjpe_results.csv"
        return 0
    else
        print_error "Full evaluation failed!"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  test                    Run basic tests"
    echo "  quick                   Run quick baseline evaluation"
    echo "  eval                    Run full model evaluation"
    echo "  walking                 Evaluate Walking action only"
    echo "  locomotion              Evaluate locomotion actions (Walking, WalkDog, WalkTogether)"
    echo "  sitting                 Evaluate sitting actions (Sitting, SittingDown)"
    echo "  all                     Run all evaluations (test + quick + full)"
    echo ""
    echo "Options for 'eval' command:"
    echo "  --model_path PATH       Path to model checkpoint (default: $MODEL_PATH)"
    echo "  --output_dir PATH       Output directory (default: $OUTPUT_DIR)"
    echo "  --num_samples N         Number of samples (default: $NUM_SAMPLES)"
    echo "  --device DEVICE         Device to use (default: $DEVICE)"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 quick"
    echo "  $0 eval"
    echo "  $0 eval --model_path my_model.pt --num_samples 100"
    echo "  $0 walking"
    echo "  $0 locomotion --output_dir locomotion_results"
}

# Parse command line arguments
COMMAND="$1"
shift || true

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
case "$COMMAND" in
    test)
        run_tests
        ;;
    quick)
        run_quick_eval
        ;;
    eval)
        run_full_eval "$MODEL_PATH" "$OUTPUT_DIR" "$NUM_SAMPLES" "$DEVICE" ""
        ;;
    walking)
        run_full_eval "$MODEL_PATH" "${OUTPUT_DIR}_walking" "$NUM_SAMPLES" "$DEVICE" "Walking"
        ;;
    locomotion)
        run_full_eval "$MODEL_PATH" "${OUTPUT_DIR}_locomotion" "$NUM_SAMPLES" "$DEVICE" "Walking WalkDog WalkTogether"
        ;;
    sitting)
        run_full_eval "$MODEL_PATH" "${OUTPUT_DIR}_sitting" "$NUM_SAMPLES" "$DEVICE" "Sitting SittingDown"
        ;;
    all)
        print_info "Running complete evaluation suite..."
        
        if run_tests && run_quick_eval && run_full_eval "$MODEL_PATH" "$OUTPUT_DIR" "$NUM_SAMPLES" "$DEVICE" ""; then
            print_success "Complete evaluation suite finished successfully!"
        else
            print_error "Some evaluations failed!"
            exit 1
        fi
        ;;
    "")
        print_error "No command specified!"
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
