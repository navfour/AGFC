"""
将生命科学主题共现矩阵从Excel批量转换为pickle格式

输入: data/citation_lifescience/lifescience_coo/*.xlsx
输出: data/citation_lifescience/lifescience_coo_pkl/*.pkl

格式与GraphWaveNet兼容: [category_ids, category_id_to_ind, adj_matrix]

使用方法:
    python convert_cooccurrence_to_pkl.py
"""

import numpy as np
import pandas as pd
import pickle
import os


def load_cooccurrence_matrix(excel_path):
    """
    从Excel加载共现矩阵

    Args:
        excel_path: Excel文件路径

    Returns:
        adj_matrix: 邻接矩阵 (numpy array)
        categories: 类别列表
    """
    print(f"读取共现矩阵: {excel_path}")

    # 读取Excel，第一列作为索引
    df = pd.read_excel(excel_path, index_col=0)

    # 提取类别名称
    categories = list(df.index)

    # 转换为numpy数组
    adj_matrix = df.values.astype(np.float32)

    print(f"✓ 矩阵形状: {adj_matrix.shape}")
    print(f"✓ 类别数量: {len(categories)}")
    print(f"✓ 类别示例: {categories[:5]}")
    print(f"✓ 数值范围: [{adj_matrix.min():.2f}, {adj_matrix.max():.2f}]")
    print(f"✓ 平均值: {adj_matrix.mean():.2f}")
    print(f"✓ 非零比例: {(adj_matrix > 0).sum() / adj_matrix.size * 100:.1f}%")

    return adj_matrix, categories


def normalize_adjacency(adj_matrix, method='minmax'):
    """
    归一化邻接矩阵

    Args:
        adj_matrix: 原始邻接矩阵
        method: 归一化方法
            - 'minmax': 归一化到 [0, 1]
            - 'max': 除以最大值
            - 'none': 不归一化

    Returns:
        归一化后的邻接矩阵
    """
    if method == 'minmax':
        min_val = adj_matrix.min()
        max_val = adj_matrix.max()
        if max_val - min_val > 0:
            adj_normalized = (adj_matrix - min_val) / (max_val - min_val)
        else:
            adj_normalized = adj_matrix
        print(f"✓ MinMax归一化: [{min_val:.2f}, {max_val:.2f}] -> [0, 1]")

    elif method == 'max':
        max_val = adj_matrix.max()
        if max_val > 0:
            adj_normalized = adj_matrix / max_val
        else:
            adj_normalized = adj_matrix
        print(f"✓ 最大值归一化: 除以 {max_val:.2f}")

    elif method == 'none':
        adj_normalized = adj_matrix
        print("✓ 保持原始数值")

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return adj_normalized


def save_adjacency_pkl(adj_matrix, categories, output_path):
    """
    保存邻接矩阵为pickle格式（兼容GraphWaveNet）

    格式: [category_ids, category_id_to_ind, adj_matrix]

    Args:
        adj_matrix: 邻接矩阵
        categories: 类别列表
        output_path: 输出文件路径
    """
    # 创建类别到索引的映射
    category_ids = categories
    category_id_to_ind = {cat: i for i, cat in enumerate(categories)}

    # 保存为pickle（与GraphWaveNet格式兼容）
    data = [category_ids, category_id_to_ind, adj_matrix]

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✓ 邻接矩阵已保存: {output_path}")
    print(f"  格式: [category_ids, category_id_to_ind, adj_matrix]")
    print(f"  矩阵形状: {adj_matrix.shape}")
    print(f"  类别数量: {len(categories)}")


def visualize_adjacency(adj_matrix, categories, output_path):
    """
    可视化邻接矩阵

    Args:
        adj_matrix: 邻接矩阵
        categories: 类别列表
        output_path: 输出图片路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 12))

        # 简化标签
        short_labels = [str(c) for c in categories]

        # 绘制热力图
        sns.heatmap(adj_matrix,
                    xticklabels=short_labels,
                    yticklabels=short_labels,
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Co-occurrence Count'},
                    linewidths=0.5,
                    linecolor='gray',
                    square=True,
                    ax=ax)

        ax.set_title('Topic Co-occurrence Matrix (edges = shared papers)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Topic B', fontsize=12)
        ax.set_ylabel('Topic A', fontsize=12)

        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ 可视化已保存: {output_path}")
        plt.close()

    except ImportError:
        print("⚠️  跳过可视化（matplotlib或seaborn未安装）")


def verify_pkl(pkl_path):
    """
    验证生成的pkl文件

    Args:
        pkl_path: pkl文件路径
    """
    print(f"\n验证pkl文件: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    category_ids, category_id_to_ind, adj_matrix = data

    print(f"✓ 类别数量: {len(category_ids)}")
    print(f"✓ 类别示例: {category_ids[:3]}")
    print(f"✓ 索引映射数量: {len(category_id_to_ind)}")
    print(f"✓ 邻接矩阵形状: {adj_matrix.shape}")
    print(f"✓ 邻接矩阵类型: {adj_matrix.dtype}")
    print(f"✓ 数值范围: [{adj_matrix.min():.4f}, {adj_matrix.max():.4f}]")

    # 检查一致性
    assert len(category_ids) == len(category_id_to_ind), "类别数量不一致"
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "邻接矩阵不是方阵"
    assert len(category_ids) == adj_matrix.shape[0], "类别数量与矩阵维度不匹配"

    print("✓ 格式验证通过")


def process_excel_file(excel_path, output_dir, normalization_method='max', generate_viz=False):
    """
    处理单个Excel文件并生成对应的pkl（和可选可视化）

    Args:
        excel_path: 输入Excel路径
        output_dir: pkl输出目录
        normalization_method: 归一化策略
        generate_viz: 是否生成热力图
    """
    print("\n" + "=" * 80)
    print(f"处理文件: {excel_path}")
    print("=" * 80)

    adj_matrix, categories = load_cooccurrence_matrix(excel_path)
    adj_normalized = normalize_adjacency(adj_matrix, method=normalization_method)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.pkl")

    save_adjacency_pkl(adj_normalized, categories, output_file)

    if generate_viz:
        viz_path = os.path.join(output_dir, f"{base_name}.png")
        print("\n生成可视化")
        print("-" * 80)
        visualize_adjacency(adj_normalized, categories, viz_path)

    verify_pkl(output_file)


def main():
    print("=" * 80)
    print("生命科学共现矩阵批量转换为Pickle格式")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_dir = os.path.join(script_dir, 'oa_1710_coo')
    output_dir = os.path.join(script_dir, 'oa_1710_coo_pkl')

    if not os.path.isdir(excel_dir):
        print(f"❌ 错误: 找不到输入目录 {excel_dir}")
        return

    excel_files = sorted(
        os.path.join(excel_dir, f)
        for f in os.listdir(excel_dir)
        if f.lower().endswith('.xlsx')
    )

    if not excel_files:
        print(f"❌ 错误: {excel_dir} 中没有找到Excel文件")
        return

    normalization_method = 'max'
    generate_viz = False  # 批量生成时默认关闭，可按需打开

    print(f"\n共发现 {len(excel_files)} 个Excel文件，将输出到: {output_dir}")
    for idx, excel_path in enumerate(excel_files, 1):
        print(f"\n>>> [{idx}/{len(excel_files)}] 开始处理 {os.path.basename(excel_path)}")
        process_excel_file(excel_path, output_dir, normalization_method, generate_viz)

    print("\n" + "=" * 80)
    print("✓ 所有文件转换完成！")
    print("=" * 80)
    print(f"\n生成的pkl已保存至: {output_dir}")
    print("\n使用示例:")
    print(f"  python train_patent_improved.py --adjdata {os.path.join(output_dir, '<文件名>.pkl')}")


if __name__ == "__main__":
    main()
