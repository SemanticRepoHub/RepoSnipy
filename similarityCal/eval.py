import warnings

warnings.filterwarnings('ignore')
import torch
from utils import build_cluster_repo_embedding, build_dataset, evaluate, calculate_similarity
from dataset import PairDataset
from torch.utils.data import DataLoader
from prettytable import PrettyTable

if __name__ == "__main__":
    MODE = 'code'  # 'topic'
    RATIO = 0
    BATCH_SIZE = 1024
    NUM_WORKER = 0

    print("Evaluate: ", MODE)

    data = build_cluster_repo_embedding(mode=MODE)
    _, data_index = build_dataset(data, ratio=RATIO)
    data_set = PairDataset(data, data_index)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKER)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = torch.load(MODE + ".pt")
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    model.eval()
    criterion.eval()
    print("Data Set:", len(data_set))
    print("Evaluating on", device)

    acc, loss, precision, recall, f1, auroc, eff, tot = evaluate(model, data_loader, device, criterion)

#########################################################################################
    # only for show how to calculate similarity and show a part of results
    # calculate similarity
    table = PrettyTable()
    table.field_names = ["NUM", "REPO1", "REPO2", "ID1", "ID2", "LABEL", "SIMILARITY", "LOSS"]
    repo_num = 15  ## modify this to control the number of shown samples
    total_loss = 0
    for i, (i1, i2, label) in enumerate(data_index[:repo_num]):
        repo1, repo2 = data[i1], data[i2]
        output = calculate_similarity(model, torch.tensor([repo1['embedding']], device=device),
                                      torch.tensor([repo2['embedding']], device=device))
        sim = output[0][0].item()
        sim_loss = abs(label[0] - sim)
        table.add_row(
            (i + 1, repo1['name'], repo2['name'], repo1['id'], repo2['id'], label, f"{sim:.5f}", f"{sim_loss:.5f}"))
        total_loss += sim_loss
    print(table.header)
    print(table)
    print("The total similarity loss is", total_loss, "\nThe avg similarity loss is", total_loss / repo_num)
