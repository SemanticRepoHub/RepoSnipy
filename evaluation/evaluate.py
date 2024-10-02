import matplotlib.pyplot as plt
import torch
import pandas as pd
from itertools import combinations
from math import comb
from pathlib import Path
from docarray.index import InMemoryExactNNIndex
from sklearn.metrics import auc, roc_curve
from torch.nn.functional import cosine_similarity
from tqdm.auto import tqdm
from common.repo_doc import RepoDoc

INDEX_PATH = Path(__file__).parent.parent.joinpath("data/index.bin")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def has_same_topic(topics1, topics2):
    """
    The function to find two topic lists whether they have the same topic
    :param topics1: the first topic list
    :param topics2: the second topic list
    :return: result
    """
    # Find shared topics other than "python" and "python3"
    intersection = set(topics1) & set(topics2) - {"python", "python3"}
    return len(intersection) > 0


def evaluate(df, level):
    filtered_df = df[df[level].notna()][["name", "topics", level]]

    # Find similarity and shared topics between all pairs of repos
    tqdm.write("Evaluating {}...".format(level))
    rows_list = []
    for row1, row2 in tqdm(
            combinations(filtered_df.itertuples(), 2), total=comb(len(filtered_df), 2)
    ):
        rows_list.append(
            {
                "repo1": row1.name,
                "repo2": row2.name,
                "has_same_topic": has_same_topic(row1.topics, row2.topics),
                level: max(
                    0.0,  # zero out negative similarities
                    cosine_similarity(
                        torch.tensor(row1._asdict()[level], device=device),
                        torch.tensor(row2._asdict()[level], device=device),
                        dim=0
                    ).cpu().detach().numpy().item(),
                ),
            }
        )

    similarity_df = pd.DataFrame(rows_list)
    similarity_df.to_csv("{}_eval_res.csv".format(level), index=False)

    # Plot ROC curve
    y_true, y_score = similarity_df["has_same_topic"], similarity_df[level]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line representing random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve of {}".format(level))
    plt.legend(loc="lower right")
    plt.savefig('ROC_evaluation_{}.png'.format(level))
    plt.show()


if __name__ == "__main__":
    levels = ["code_embedding", "doc_embedding", "readme_embedding", "requirement_embedding", "repository_embedding"]
    index = InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH)
    df = index._docs.to_dataframe()
    for level in levels:
        evaluate(df, level)
