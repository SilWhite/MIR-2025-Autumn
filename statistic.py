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
    for model_name, recalls in recall_at_k.items():
        ks = list(range(1, len(recalls) + 1))
        plt.plot(ks, recalls, marker="o", label=model_name, markersize=3)

    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall@K for Different Models")
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "recall_at_k_comparison.png"))
    plt.close()
    print(f"Saved result to {os.path.join(RESULT_DIR, 'recall_at_k_comparison.png')}")


if __name__ == "__main__":
    plot_all_recall_at_k()
