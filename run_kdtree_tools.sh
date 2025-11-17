#!/bin/bash
# KD-tree 参数调整和对比实验脚本

# 激活环境
eval "$(conda shell.bash hook)"
conda activate rogs

echo "=========================================="
echo "KD-tree 参数调整工具集"
echo "=========================================="
echo ""
echo "请选择要运行的脚本:"
echo "1. 查看所有参数配置说明"
echo "2. 运行网格对比实验（KD-tree vs 固定网格）"
echo "3. 交互式参数调整工具（实时预览）"
echo "4. 运行完整 demo（使用当前最佳参数）"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "========== 参数配置说明 =========="
        python configs/kdtree_params.py
        ;;
    2)
        echo ""
        echo "========== 运行网格对比实验 =========="
        python demo_grid_comparison.py
        ;;
    3)
        echo ""
        echo "========== 启动交互式调整工具 =========="
        python demo_interactive_tuning.py
        ;;
    4)
        echo ""
        echo "========== 运行完整 Demo =========="
        python demo_multiview_bev_kdtree.py
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "完成！"
