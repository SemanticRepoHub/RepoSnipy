import warnings
warnings.filterwarnings('ignore')
import torch
from utils import build_cluster_repo_embedding, build_dataset, train_epoch, evaluate, calculate_similarity
from dataset import PairDataset
from common.pair_classifier import PairClassifier
from torch.utils.data import DataLoader
from prettytable import PrettyTable


if __name__ == "__main__":
    MODE = 'topic'
    RATIO = 0.7
    EPOCH = 0
    BATCH_SIZE = 1024
    LR = 0.001
    WEIGHT_DECAY = 0.01
    NUM_WORKER = 0

    print("Mode:", MODE)

    data = build_cluster_repo_embedding(mode=MODE)
    train_index, test_index = build_dataset(data, ratio=RATIO)
    train_set = PairDataset(data, train_index)
    test_set = PairDataset(data, test_index)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKER)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = PairClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)

    print("Train Set:", len(train_set))
    print("Test Set:", len(test_set))
    print("Training on", device)

    # train the model
    best_acc = 0
    for epoch in range(1, EPOCH + 1):
        train_epoch(epoch, model, train_loader, device, criterion, optimizer)

        # test
        acc, loss, precision, recall, f1, auroc = evaluate(model, test_loader, device, criterion)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.cpu(), MODE + ".pt")
            model.to(device)
            print("Save model")

    # calculate similarity
    table = PrettyTable()
    table.field_names = ["NUM", "REPO1", "REPO2", "ID1", "ID2", "LABEL", "SIMILARITY", "LOSS"]
    repo_num = 15
    total_loss = 0
    model = torch.load(MODE + ".pt")
    model.to(device)
    model.eval()
    for i, (i1, i2, label) in enumerate(test_index[:repo_num]):
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
