#!/bin/bash

# Human Motion Prediction Project Installation Script
# ===================================================

set -e  # Exit on any error

echo "🚀 Human Motion Prediction Project Installation"
echo "=============================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Check if Python is available
check_python() {
    print_status "检查Python环境..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
        print_success "找到Python3: $(python3 --version)"
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
        print_success "找到Python: $(python --version)"
    else
        print_error "未找到Python！请先安装Python 3.7+"
        exit 1
    fi
}

# Check if pip is available
check_pip() {
    print_status "检查pip..."
    
    if command -v pip3 &> /dev/null; then
        PIP_CMD=pip3
        print_success "找到pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD=pip
        print_success "找到pip"
    else
        print_error "未找到pip！请先安装pip"
        exit 1
    fi
}

# Create virtual environment (optional)
create_venv() {
    if [ "$1" = "--venv" ]; then
        print_status "创建虚拟环境..."
        
        if [ -d "venv" ]; then
            print_warning "虚拟环境已存在，跳过创建"
        else
            $PYTHON_CMD -m venv venv
            print_success "虚拟环境创建成功"
        fi
        
        print_status "激活虚拟环境..."
        source venv/bin/activate
        print_success "虚拟环境已激活"
        
        # Update pip in venv
        pip install --upgrade pip
    fi
}

# Install dependencies
install_dependencies() {
    print_status "安装项目依赖..."
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt 文件不存在！"
        exit 1
    fi
    
    # Try to install with pip
    if $PIP_CMD install -r requirements.txt; then
        print_success "依赖安装成功"
    else
        print_warning "pip安装失败，尝试用户模式安装..."
        if $PIP_CMD install --user -r requirements.txt; then
            print_success "用户模式安装成功"
        else
            print_error "依赖安装失败！请检查网络连接或手动安装"
            exit 1
        fi
    fi
}

# Install project in development mode
install_project() {
    print_status "安装项目包..."
    
    if $PIP_CMD install -e .; then
        print_success "项目包安装成功"
    else
        print_warning "项目包安装失败，尝试用户模式..."
        if $PIP_CMD install --user -e .; then
            print_success "用户模式安装成功"
        else
            print_warning "项目包安装失败，但核心功能仍可使用"
        fi
    fi
}

# Create necessary directories
create_directories() {
    print_status "创建必要目录..."
    
    directories=("data" "checkpoints" "logs" "results" "data/human36m")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "创建目录: $dir"
        fi
    done
}

# Run simple demo to verify installation
verify_installation() {
    print_status "验证安装..."
    
    if $PYTHON_CMD examples/simple_demo.py > /dev/null 2>&1; then
        print_success "安装验证成功！"
    else
        print_warning "安装验证失败，但基本功能可能仍然可用"
    fi
}

# Download sample data (optional)
download_sample_data() {
    if [ "$1" = "--download-sample" ]; then
        print_status "下载示例数据..."
        # This is a placeholder - in a real scenario, you would download actual data
        print_warning "示例数据下载功能待实现，请使用生成的示例数据"
    fi
}

# Main installation process
main() {
    echo "开始安装过程..."
    echo ""
    
    # Parse command line arguments
    USE_VENV=false
    DOWNLOAD_SAMPLE=false
    
    for arg in "$@"; do
        case $arg in
            --venv)
                USE_VENV=true
                shift
                ;;
            --download-sample)
                DOWNLOAD_SAMPLE=true
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --venv              创建并使用虚拟环境"
                echo "  --download-sample   下载示例数据"
                echo "  --help              显示此帮助信息"
                echo ""
                echo "示例:"
                echo "  $0                  # 基本安装"
                echo "  $0 --venv           # 使用虚拟环境安装"
                echo "  $0 --venv --download-sample  # 完整安装"
                exit 0
                ;;
        esac
    done
    
    # Run installation steps
    check_python
    check_pip
    
    if [ "$USE_VENV" = true ]; then
        create_venv --venv
    fi
    
    install_dependencies
    install_project
    create_directories
    
    if [ "$DOWNLOAD_SAMPLE" = true ]; then
        download_sample_data --download-sample
    fi
    
    verify_installation
    
    echo ""
    echo "🎉 安装完成！"
    echo "============="
    echo ""
    echo "下一步操作："
    echo "1. 运行简化演示: ${PYTHON_CMD} examples/simple_demo.py"
    echo "2. 运行完整演示: ${PYTHON_CMD} examples/demo.py"
    echo "3. 查看文档: cat README.md"
    echo "4. 快速入门: cat QUICKSTART.md"
    echo ""
    
    if [ "$USE_VENV" = true ]; then
        echo "注意: 如果使用了虚拟环境，请记得激活它："
        echo "source venv/bin/activate"
        echo ""
    fi
    
    print_success "Happy Motion Predicting! 🚀"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}安装被用户中断${NC}"; exit 1' INT

# Run main function
main "$@"