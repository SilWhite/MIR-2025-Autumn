import matplotlib.pyplot as plt
import os


RESULT_DIR = "result/"


def plot_all_recall_at_k(result_dir=RESULT_DIR):
    recall_at_k = {}
    for dir in os.listdir(result_dir):
        if not os.path.exists(os.path.join(result_dir, dir, "recall_at_k.csv")):
            continue
        with open(os.path.join(result_dir, dir, "recall_at_k.csv"), "r") as f:
            lines = f.readlines()
            recalls = [float(line.strip().split(",")[1]) for line in lines]
            recall_at_k[dir] = recalls

    recall_at_k = dict(sorted(recall_at_k.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    lines_by_model = {}
    for model_name, recalls in recall_at_k.items():
        ks = list(range(1, len(recalls) + 1))
        (line,) = plt.plot(ks, recalls, marker="o", label=model_name, markersize=3)
        lines_by_model[model_name] = line

    # 在 K=1,10,100 处画竖向虚线，并标注各模型的数值
    k_marks = [1, 10, 20, 50, 100]
    offset_map = {  # 适度偏移以避免和点重叠
        1: (-28, 0),
        10: (0, -10),
        20: (0, -10),
        50: (0, -10),
        100: (0, -10),
    }
    y_max = plt.ylim()[1]
    for k in k_marks:
        plt.axvline(x=k, linestyle="--", color="gray", linewidth=1, alpha=0.6)
        # 在图顶部标记 @K
        plt.text(k, y_max, f"@{k}", ha="center", va="top", fontsize=9, color="gray")
        # 为每个模型在该 K 处标注
        for model_name, recalls in recall_at_k.items():
            y = recalls[k - 1]
            color = lines_by_model[model_name].get_color()
            plt.scatter([k], [y], s=28, color=color, zorder=3)

            dx, dy = offset_map.get(k, (6, 6))
            plt.annotate(
                f"{y:.3f}",
                xy=(k, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    plt.xlabel("K")
    plt.ylabel("Hit@K")
    plt.title("Hit@K for Different Models")
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "hit_at_k_comparison.png"))
    plt.close()
    print(f"Saved result to {os.path.join(RESULT_DIR, 'hit_at_k_comparison.png')}")


if __name__ == "__main__":
    plot_all_recall_at_k()
