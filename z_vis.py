import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 基础绘图风格（更接近论文风格）
plt.style.use("classic")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"


def parse_markdown_tables(md_file: str):
	"""从 markdown 表格中解析实验结果，返回 {exp_name: DataFrame}。"""

	with open(md_file, "r", encoding="utf-8") as f:
		content = f.read()

	experiments = {}

	# 匹配：标题行 + 表头 + 分隔行 + 多行表格内容
	pattern = r"###\s+\d+、([^\n]+)\s+\|[^\n]+\n\|[-:\s|]+\n((?:\|[^\n]+\n)+)"
	matches = re.findall(pattern, content)

	for exp_name, table_content in matches:
		exp_name = exp_name.strip()

		rows = [line.strip() for line in table_content.strip().split("\n") if line.strip()]
		data = []

		for row in rows:
			cells = [c.strip() for c in row.split("|") if c.strip()]
			if len(cells) < 6:
				continue
			try:
				metric_type = cells[0]
				metric_subtype = cells[1]
				iou = cells[2]
				area = cells[3]
				maxdets = cells[4]
				value = float(cells[5])
			except (ValueError, IndexError):
				continue

			data.append(
				{
					"metric": metric_type,
					"type": metric_subtype,
					"iou": iou,
					"area": area,
					"maxDets": maxdets,
					"value": value,
				}
			)

		if data:
			experiments[exp_name] = pd.DataFrame(data)

	return experiments


def plot_ap_iou_line(experiments, save_path="ap_iou_line.png"):
	"""画 6 个 AP 点的折线图：

	IoU=0.50:0.95 / 0.50 / 0.75, area 为 all / small / medium / large，
	x 轴为 (IoU, area) 组合，y 轴为 AP，三条线对应三个模型。
	"""

	fig, ax = plt.subplots(figsize=(7, 5))

	# 要画的 6 个配置（顺序固定）
	ap_configs = [
		("IoU=0.50:0.95", "all", "0.50:0.95\nall"),
		("IoU=0.50", "all", "0.50\nall"),
		("IoU=0.75", "all", "0.75\nall"),
		("IoU=0.50:0.95", "small", "0.50:0.95\nsmall"),
		("IoU=0.50:0.95", "medium", "0.50:0.95\nmedium"),
		("IoU=0.50:0.95", "large", "0.50:0.95\nlarge"),
	]

	x = np.arange(len(ap_configs))
	tick_labels = [cfg[2] for cfg in ap_configs]

	exp_names = list(experiments.keys())
	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
	markers = ["o", "s", "^", "D", "v"]

	for idx, (exp_name, df) in enumerate(experiments.items()):
		ap_df = df[(df["metric"] == "Average Precision")]
		y_vals = []
		for iou_key, area_key, _ in ap_configs:
			sel = ap_df[(ap_df["iou"] == iou_key) & (ap_df["area"] == area_key)]
			if len(sel) > 0:
				y_vals.append(float(sel["value"].values[0]))
			else:
				y_vals.append(0.0)

		color = colors[idx % len(colors)]
		marker = markers[idx % len(markers)]

		ax.plot(
			x,
			y_vals,
			marker=marker,
			markersize=6,
			linewidth=2.0,
			color=color,
			label=exp_name,
		)

	# 论文风格细节
	ax.set_xlim(-0.2, len(ap_configs) - 0.8)
	ax.set_ylim(0.0, 1.0)
	ax.set_xticks(x)
	ax.set_xticklabels(tick_labels, fontsize=10)
	ax.set_ylabel("AP", fontsize=12)
	ax.set_xlabel("IoU / area", fontsize=12)
	ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

	# 只画水平网格，类似很多论文图
	ax.yaxis.grid(True, linestyle="--", alpha=0.4)
	ax.xaxis.grid(False)

	for spine in ["top", "right"]:
		ax.spines[spine].set_visible(True)

	ax.legend(frameon=False, fontsize=10)

	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	print(f"Saved: {save_path}")
	plt.close()


def main():
	md_file = "实验数据.md"

	if not Path(md_file).exists():
		print(f"Error: file not found: {md_file}")
		return

	experiments = parse_markdown_tables(md_file)
	if not experiments:
		print("Error: no experiments parsed from markdown")
		return

	print("Parsed experiments:")
	for name in experiments.keys():
		print("  -", name)

	out_dir = Path("visualization_results")
	out_dir.mkdir(exist_ok=True)

	plot_ap_iou_line(experiments, out_dir / "ap_iou_line.png")

	print(f"All figures saved in: {out_dir}")


if __name__ == "__main__":
	main()

