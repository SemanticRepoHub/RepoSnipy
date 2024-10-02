import json
import os
from pathlib import Path

import torch
from docarray.index import InMemoryExactNNIndex
from common.repo_doc import RepoDoc
import random
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm

INDEX_PATH = Path(__file__).parent.joinpath("..\\data\\")
TOPIC_CLUSTER_PATH = Path(__file__).parent.joinpath("..\\data\\repo_topic_clusters.json")
CODE_CLUSTER_PATH = Path(__file__).parent.joinpath("..\\data\\repo_code_clusters.json")


def read_repo_cluster(filename):
    # return repo name - cluster id key value pair
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def find_files_in_directory(directory):
    # loop all index files
    files = []
    for file in os.listdir(directory):
        if file[:5] == "index" and file[5] != ".":
            files.append(os.path.join(directory, file))
    return files


def read_repo_embedding():
    # return repo name - embedding k-v pair
    map = {}
    for filename in find_files_in_directory(INDEX_PATH):
        data = InMemoryExactNNIndex[RepoDoc](index_file_path=Path(__file__).parent.joinpath(filename))
        docs_tmp = data._docs
        for doc in docs_tmp:
            map[doc.name] = doc.repository_embedding
    return map


def build_cluster_repo_embedding(mode: str):
    """
    build the dataset according to code cluster
    where mode is "code" or "topic"
    """
    embedding = read_repo_embedding()
    if mode == "code":
        cluster_id = read_repo_cluster(CODE_CLUSTER_PATH)
    elif mode == "topic":
        cluster_id = read_repo_cluster(TOPIC_CLUSTER_PATH)
    else:
        raise ValueError("parameter 'mode' must be 'code' or 'topic'")
    data = []
    for name in embedding:
        data.append({'name': name, 'embedding': embedding[name], 'id': cluster_id[name]})
    return data


def build_dataset(data, ratio=0.7):
    """
    return the train set and test set which are like (index1, index2) : (same, not same)
    """
    positive_repo = []
    negative_repo = []
    n = len(data)
    # build the binary dataset
    for i in range(n):
        for j in range(i, n):
            if data[i]['id'] == data[j]['id']:
                positive_repo.append((i, j, (1.0, 0.0)))
                positive_repo.append((j, i, (1.0, 0.0)))
            else:
                negative_repo.append((i, j, (0.0, 1.0)))
                negative_repo.append((j, i, (0.0, 1.0)))
    # make balance
    positive_length = len(positive_repo)
    negative_repo = random.choices(negative_repo, k=positive_length)
    # split the dataset
    random.shuffle(positive_repo)
    random.shuffle(negative_repo)
    split_index = int(positive_length * ratio)
    train_set = positive_repo[:split_index] + negative_repo[:split_index]
    random.shuffle(train_set)
    test_set = positive_repo[split_index:] + negative_repo[split_index:]
    random.shuffle(test_set)
    print("Positive data:", len(positive_repo))
    print("Negative data:", len(negative_repo))
    return train_set, test_set


def train_epoch(epoch, model, loader, device, criterion, optimizer):
    model.train()
    accuracy = Accuracy(task='binary')
    precision = Precision(task='binary')
    recall = Recall(task='binary')
    f1 = F1Score(task='binary')
    auroc = AUROC(task='binary')
    accuracy.to(device)
    precision.to(device)
    recall.to(device)
    f1.to(device)
    auroc.to(device)
    total_loss = 0
    count = 0
    for repo1, repo2, label in tqdm(loader):
        count += len(label)
        optimizer.zero_grad()
        repo1 = repo1.to(device)
        repo2 = repo2.to(device)
        label = label.to(device)
        pred = model(repo1, repo2)

        loss = criterion(pred, label)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        accuracy(pred, label)
        precision(pred, label)
        recall(pred, label)
        f1(pred, label)
        auroc(pred, label)
    print("Epoch", epoch, "Train loss:", total_loss / count, "Acc", accuracy.compute().item(), "Precision:",
          precision.compute().item(), "Recall:", recall.compute().item(), "F1:", f1.compute().item(),
          "AUROC:", auroc.compute().item())


def evaluate(model, loader, device, criterion):
    model.eval()
    with torch.no_grad():
        test_accuracy = Accuracy(task='binary')
        test_precision = Precision(task='binary')
        test_recall = Recall(task='binary')
        test_f1 = F1Score(task='binary')
        test_auroc = AUROC(task='binary')
        test_accuracy.to(device)
        test_precision.to(device)
        test_recall.to(device)
        test_f1.to(device)
        test_auroc.to(device)
        total_loss = 0
        count = 0
        for repo1, repo2, label in tqdm(loader):
            count += len(label)
            repo1 = repo1.to(device)
            repo2 = repo2.to(device)
            label = label.to(device)
            pred = model(repo1, repo2)
            loss = criterion(pred, label)
            total_loss += loss.item()

            test_accuracy(pred, label)
            test_precision(pred, label)
            test_recall(pred, label)
            test_f1(pred, label)
            test_auroc(pred, label)
        print("Test loss:", total_loss / count, "Acc", test_accuracy.compute().item(), "Precision:",
              test_precision.compute().item(), "Recall:", test_recall.compute().item(), "F1:", test_f1.compute().item(),
              "AUROC:", test_auroc.compute().item())

    return test_accuracy.compute().item(), total_loss / count, test_precision.compute().item(), test_recall.compute().item(), \
           test_f1.compute().item(), test_auroc.compute().item()


def calculate_similarity(model, repo_emb1, repo_emb2):
    return torch.nn.functional.softmax(model(repo_emb1, repo_emb2) + model(repo_emb2, repo_emb1), dim=1)
