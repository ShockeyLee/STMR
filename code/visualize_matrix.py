import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

def is_color_dark(color):
    """
    检查颜色是否为深色，用于决定其上的文本应为白色还是黑色。
    """
    rgb = to_rgba(color)[:3]
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return luminance < 0.5

def visualize_stmr_matrix(matrix: np.ndarray, id_to_label: dict):
    """
    为 STMR 语义矩阵生成可视化图像。
    v3: 修正了无人机位置'P'在值为0的单元格上不显示的逻辑错误。
    """
    # --- 颜色配置 ---
    colors = [
        '#E0E0E0',  # 0: unexplored (浅灰色)
        '#FFC0CB',  # 1: (粉色)
        '#FFDAB9',  # 2: (浅橙色/桃色)
        '#2F4F4F',  # 3: (暗灰绿色)
        '#ADD8E6',  # 4: (浅蓝色)
        '#DA70D6',  # 5: (兰花紫)
        '#90EE90',  # 6: (浅绿色)
        '#BDB76B',  # 7: (暗卡其色)
    ]
    
    max_id = matrix.max()
    while len(colors) <= max_id:
        colors.append(np.random.rand(3,))

    custom_cmap = ListedColormap(colors)
    # --- 颜色配置结束 ---

    height, width = matrix.shape
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(matrix, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)

    drone_pos_y, drone_pos_x = (height - 1) // 2, (width - 1) // 2

    # ==================== 修正后的核心逻辑 ====================
    # 遍历所有单元格进行绘制
    for i in range(height):
        for j in range(width):
            # 优先级 1: 检查并绘制无人机位置'P'
            if i == drone_pos_y and j == drone_pos_x:
                ax.text(j, i, 'P', 
                        ha='center', va='center', 
                        color='red', fontsize=16, fontweight='bold')
            # 优先级 2: 如果不是无人机位置，再绘制地标ID数字
            else:
                cell_id = matrix[i, j]
                # 仅当ID不为0时才绘制数字
                if cell_id != 0:
                    bg_color = colors[cell_id]
                    text_color = 'white' if is_color_dark(bg_color) else 'black'
                    ax.text(j, i, str(cell_id), 
                            ha='center', va='center', 
                            color=text_color, fontsize=12)
    # =========================================================

    # --- 绘图定制 ---
    ax.set_title("Semantic Matrix Visualization", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # --- 图例 ---
    legend_text = "P = (Drone's Position)"
    ax.text(1.02, 0.98, legend_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
    plt.savefig("semantic_matrix_visualization.png", dpi=300)

if __name__ == '__main__':
    print("Running visualization with sample data...")
    # 示例：创建一个中心点为0的矩阵来测试'P'是否能正常显示
    sample_matrix = np.zeros((20, 20), dtype=int)
    sample_matrix[2:5, 15:18] = 2
    sample_matrix[4:9, 5:8] = 3
    # 确保中心点 (9,9) 的值为 0
    assert sample_matrix[(20-1)//2, (20-1)//2] == 0 
    
    sample_id_to_label = {
        0: 'unexplored', 1: 'road', 2: 'building', 3: 'parking lot'
    }
    visualize_stmr_matrix(sample_matrix, sample_id_to_label)