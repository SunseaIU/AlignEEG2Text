"""
Fig.3 投影架构消融图（论文用图）
- 数据来源：summary.csv（40 个真实训练 run，10 名被试 × 4 变体）
- 柱序（左→右）：Single Token → Independent Heads → Transformer → Joint Linear (Ours)
- 颜色：与旧版图逐像素一致（Single 灰蓝 / Indep 浅蓝 / Trans 珊瑚 / Ours 绿）
- 无标题、无图例、无显著性标记；X 轴仅变体名（n=10 写入图注）
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = r'd:\01-Code\EEGProject\ParaEEGToText\experiments\results_exp2a_formal\summary.csv'
OUT_DIR  = r'd:\01-Code\EEGProject\ParaEEGToText\experiments\results_exp2a_formal'

df = pd.read_csv(CSV_PATH)

# 柱子顺序（Ours 放最右）
order = ['Single Token', 'Independent Heads', 'Transformer', 'Joint Linear (Ours)']

# 每个变体对应的固定颜色（从原版 fig3 提取的精确 RGB）
color_map = {
    'Single Token':        (144/255, 175/255, 197/255),   # greyblue
    'Independent Heads':   (127/255, 179/255, 213/255),   # lightblue
    'Transformer':         (241/255, 148/255, 138/255),   # coral
    'Joint Linear (Ours)': ( 82/255, 190/255, 128/255),   # green
}

means, stds = [], []
for name in order:
    sub = df[df['variant_display'] == name]
    means.append(sub['bleu1'].mean())
    stds.append(sub['bleu1'].std(ddof=1))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 1.2,
})

fig, ax = plt.subplots(figsize=(7.0, 4.5))
x = np.arange(len(order))
colors = [color_map[n] for n in order]

ax.bar(x, means, yerr=stds, width=0.6, color=colors,
       edgecolor='black', linewidth=1.0,
       error_kw={'elinewidth': 1.2, 'capsize': 6, 'capthick': 1.2})

ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=10)            # 去掉 (n=10)
ax.set_ylabel('BLEU-1', fontsize=12)
ax.set_ylim(0, max(means) * 1.28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_tick_params(width=1.0)
ax.xaxis.set_tick_params(width=1.0)

# 柱顶数值（三位小数）
for xi, m, s in zip(x, means, stds):
    ax.text(xi, m + s + 0.0035, f'{m:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}\\fig3_new.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}\\fig3_new.pdf',            bbox_inches='tight')
print('saved fig3_new.png / fig3_new.pdf')
for n, m, s in zip(order, means, stds):
    print(f'  {n:22s}  BLEU-1 = {m:.4f} +/- {s:.4f}')
