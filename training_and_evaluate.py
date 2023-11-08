import torch
import torchvision
from torchvision import datasets, transforms, ops, models
import copy
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from typing import Tuple, Dict
from torch.utils.data import default_collate
from torchvision.transforms import v2

NUM_CLASSES = 10
mixup = v2.MixUp(num_classes=NUM_CLASSES)


def collate_fn(batch):
    return mixup(*default_collate(batch))


def train_and_eval(model: torch.nn.Module, trainset: Dataset, testset: Dataset, batch_sizes: Tuple[int],
                   NAME_OF_MODEL: str = "test_model", folder_to_save_in: str = "", NUM_OF_EPOCHS: int = 100,
                   early_stopping_patience: int = 15, NUM_OF_WORKERS: int = 2, mixup=False) -> Dict[
    int, Dict[str, Dict[str, float]]]:
    """
    Train and evaluate a PyTorch model on given datasets.
    Note: The model's layer should be frozen before calling this function.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        trainset (Dataset): Training dataset.
        testset (Dataset): Testing dataset.
        batch_sizes (Tuple[int]): Batch sizes for training.
        NAME_OF_MODEL (str): Name of the model (default: "test_model").
        folder_to_save_in (str): Folder to save model weights and training information (default: "").
        NUM_OF_EPOCHS (int): Number of training epochs (default: 100).
        early_stopping_patience (int): Patience for early stopping (default: 15).
        NUM_OF_WORKERS (int): Number of workers for data loading (default: 2).

    Returns:
        Dict[int, Dict[str, Dict[str, float]]]: A dictionary containing training and testing loss and accuracy for
        different batch sizes.

    """
    batch_tr_info = {}
    batch_te_info = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for BATCH_SIZE in batch_sizes:

        current_model = copy.deepcopy(model)

        # Move your model to the GPU if available
        current_model.to(device)

        optimizer = torch.optim.Adam(current_model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        if mixup:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                                      num_workers=NUM_OF_WORKERS, collate_fn=collate_fn)
            testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                                                     num_workers=NUM_OF_WORKERS, collate_fn=collate_fn)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                                      num_workers=NUM_OF_WORKERS)
            testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                                                     num_workers=NUM_OF_WORKERS)

        # epoch loss and accuracy
        tr_loss, tr_acc = [], []
        te_loss, te_acc = [], []
        best_test_loss = float('inf')

        early_stopping_counter = 0

        for t in (range(NUM_OF_EPOCHS)):
            current_model.train()
            batch_loss, batch_accuracy = [], []
            print(f"Epoch {t + 1} Training...")
            for X, y in tqdm(trainloader):
                X = X.to(device)
                y = y.to(device)
                predicted = current_model(X)
                loss = loss_fn(predicted, y)
                batch_accuracy.append(float(torch.argmax(predicted, dim=1).eq(y).sum().item() / len(y)))
                batch_loss.append(float(loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batches = len(batch_loss)
            tr_loss.append(sum(batch_loss) / batches)
            tr_acc.append(sum(batch_accuracy) / batches)

            current_model.eval()
            with torch.no_grad():
                batch_loss, batch_accuracy = [], []
                print(f"Epoch {t + 1} Testing...")
                for X, y in tqdm(testloader):
                    X = X.to(device)
                    y = y.to(device)
                    predicted = current_model(X)
                    loss = loss_fn(predicted, y)
                    batch_accuracy.append(float(torch.argmax(predicted, dim=1).eq(y).sum().item() / len(y)))
                    batch_loss.append(float(loss.item()))

                batches = len(batch_loss)
                te_loss.append(sum(batch_loss) / batches)
                te_acc.append(sum(batch_accuracy) / batches)

            model_weights_dir = f"model_weights/{NAME_OF_MODEL}/{folder_to_save_in}"

            if not os.path.exists(model_weights_dir):
                os.makedirs(model_weights_dir)

            # Save the current_model's weights after each epoch
            torch.save(current_model.state_dict(), f"{model_weights_dir}/{BATCH_SIZE}_model_weights.pth")

            # Check if the current test loss is the best so far
            if te_loss[-1] < best_test_loss:
                best_test_loss = te_loss[-1]
                early_stopping_counter = 0
                # Save the current_model's weights with the best test loss
                torch.save(current_model.state_dict(), f"{model_weights_dir}/{BATCH_SIZE}_best_model_weights.pth")
            else:
                early_stopping_counter += 1

            print(
                f"Epoch {t + 1}: Train_accuracy: {(100 * tr_acc[-1]):>0.2f}%, Train_loss: {tr_loss[-1]:>8f}, Test_accuracy: {(100 * te_acc[-1]):>0.2f}%, Test_loss: {te_loss[-1]:>8f}")
            batch_tr_info[BATCH_SIZE] = {"loss": tr_loss, "acc": tr_acc}
            batch_te_info[BATCH_SIZE] = {"loss": te_loss, "acc": te_acc}

            training_info_dir = f"training_information/{NAME_OF_MODEL}/{folder_to_save_in}"
            if not os.path.exists(training_info_dir):
                os.makedirs(training_info_dir)

            # Save the dictionary to a file
            with open(f"{training_info_dir}/batch_tr_info.pkl", 'wb') as file:
                pickle.dump(batch_tr_info, file)

            # Save the dictionary to a file
            with open(f"{training_info_dir}/batch_te_info.pkl", 'wb') as file:
                pickle.dump(batch_te_info, file)

            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered. No improvement for {early_stopping_patience} epochs.")
                break
    return batch_tr_info, batch_te_info
