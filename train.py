import pickle
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from data_utils import DocumentDataset, read_document, read_topic_codes

batch_size = 32
num_workers = 10
max_seq_len = None  # if None, uses default BERT, which is 512
num_epochs = 15
early_stop_patience = 3  # epochs
unfreeze_base_after = 2  # epochs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(data_dir, model_dir, topic_codes):
    # Read the 75 most common topic codes from a separate file.
    # These topic codes will be used to filter the documents and topics.
    # This file was created when exploring the data in a Jupyter Notebook.
    with open("codes_75.tsv") as fh:
        code_filter = set(line.split()[0] for line in fh)
    code2idx, idx2code, code2desc = read_topic_codes(
        topic_codes, code_filter=code_filter
    )
    # Save the three dictionaries used for topic code conversion
    # as a pickle file in model_dir, for easier use in prediction.
    Path(model_dir).mkdir(exist_ok=True)
    with open(Path(model_dir) / "topic_codes.pkl", "wb") as fh:
        pickle.dump((code2idx, idx2code, code2desc), fh)
    num_classes = len(code2desc)  # == len(code_filter)

    # Read all XML-files in available
    data_files = sorted(Path(data_dir).rglob("*.xml"))
    print(f"[INFO] {len(data_files)} documents available")
    print("[INFO] Reading data")
    # Filters out all documents without any codes in `code_filter`
    data = list(
        filter(
            None, [read_document(file, code_filter=code_filter) for file in data_files]
        )
    )
    print(f"[INFO] Using data from {len(data)} documents")

    print("[INFO] Initializing model")
    # BERT is downloaded from HuggingFace
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=num_classes
    )
    # Freeze base model for few epochs at start
    for param in model.base_model.parameters():
        param.requires_grad = False
    # Set a smaller learning rate for base model, as it is already pre-tuned
    optimizer = AdamW(
        [
            {"params": model.base_model.parameters(), "lr": 5e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    )
    # Binary Cross Entropy Loss (takes sigmoids as input)
    loss_fn = torch.nn.BCELoss()
    # Tokenizer is also downloaded from HuggingFace
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Split training data: train 0.75, val 0.125, test 0.125
    train_data, test_data = train_test_split(
        data, train_size=0.75, random_state=8, shuffle=True
    )
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, random_state=8, shuffle=True
    )
    # DocumentDataset is a custom class inheriting from PyTorch Dataet
    train_dataset = DocumentDataset(train_data, tokenizer, code2idx, max_seq_len)
    val_dataset = DocumentDataset(val_data, tokenizer, code2idx, max_seq_len)
    test_dataset = DocumentDataset(test_data, tokenizer, code2idx, max_seq_len)
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("[INFO] Training...")
    train_model(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        model_dir,
    )

    # Test the final model version obtained from training (saved to model_dir)
    best_model = BertForSequenceClassification.from_pretrained(model_dir)
    test_model(best_model, test_loader)
    print(f"[INFO] All done! Best model saved to {model_dir}")


def train_model(model, optimizer, loss_fn, train_loader, val_loader, model_dir):
    model.to(device)
    max_val_f1 = 0.0
    min_val_loss = 0.0
    no_improvement = 0
    try:
        for epoch in range(1, num_epochs + 1):
            print(f"\n----- Epoch {epoch} -----")

            # Training phase
            model.train()
            train_loss = 0.0
            train_hamming = 0.0
            train_f1 = 0.0
            train_emr = 0.0
            num_samples = 0
            num_batches = 0
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                input_ids, attention_mask, targets = batch
                out = model(
                    input_ids.to(device), attention_mask=attention_mask.to(device)
                )
                sigmoids = torch.sigmoid(out.logits)
                loss = loss_fn(sigmoids, targets.to(device))
                loss.backward()
                optimizer.step()
                # Make multi-hot tensors from sigmoids.
                # Topic is set to 1 if sigmoid is above theshold and 0 otherwise
                sigmoids[sigmoids >= 0.5] = 1
                sigmoids[sigmoids < 0.5] = 0
                predictions = sigmoids.detach().cpu()
                train_loss += loss.item() * len(targets)
                train_hamming += hamming_loss(targets, predictions)
                train_f1 += f1_score(
                    targets, predictions, average="macro", zero_division=0
                )
                train_emr += accuracy_score(targets, predictions)
                num_samples += len(targets)
                num_batches += 1
            train_loss /= num_samples
            train_hamming /= num_batches
            train_f1 /= num_batches
            train_emr /= num_batches
            print(
                f"[TRAIN] loss: {train_loss}\n"
                f"        hamming loss: {train_hamming}\n"
                f"        F1-score: {train_f1}\n"
                f"        exact match ratio: {train_emr * 100} %"
            )

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_hamming = 0.0
            val_f1 = 0.0
            val_emr = 0.0
            num_samples = 0
            num_batches = 0
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    input_ids, attention_mask, targets = batch
                    out = model(
                        input_ids.to(device), attention_mask=attention_mask.to(device)
                    )
                    sigmoids = torch.sigmoid(out.logits)
                    loss = loss_fn(sigmoids, targets.to(device))
                    sigmoids[sigmoids >= 0.5] = 1
                    sigmoids[sigmoids < 0.5] = 0
                    predictions = sigmoids.detach().cpu()
                    val_loss += loss.item() * len(targets)
                    val_hamming += hamming_loss(targets, predictions)
                    val_f1 += f1_score(
                        targets, predictions, average="macro", zero_division=0
                    )
                    val_emr += accuracy_score(targets, predictions)
                    num_samples += len(targets)
                    num_batches += 1
            val_loss /= num_samples
            val_hamming /= num_batches
            val_f1 /= num_batches
            val_emr /= num_batches
            print(
                f"[VAL] loss: {val_loss}\n"
                f"      hamming loss: {val_hamming}\n"
                f"      F1-score: {val_f1}\n"
                f"      exact match ratio: {val_emr * 100} %"
            )

            # Checkpoint
            if val_f1 > max_val_f1:
                print("[INFO] Increased F1-score, saving model state")
                max_val_f1 = val_f1
                model.save_pretrained(model_dir)
            if val_loss < min_val_loss or epoch == 1:
                no_improvement = 0
                min_val_loss = val_loss
            else:
                no_improvement += 1
                print(f"[INFO] No reduction in loss for {no_improvement} epochs")
            if no_improvement >= early_stop_patience:
                print("[INFO] Stopping early")
                break
            if epoch == unfreeze_base_after:
                # Make base model (BERT) trainable
                print("[INFO] Unfreezing base model")
                for param in model.base_model.parameters():
                    param.requires_grad = True
            if no_improvement >= 1 and epoch > unfreeze_base_after:
                # Learning has plateaued, divide head learning rate by 10,
                # and base model learning rate by 5
                print("[INFO] Decreasing classifier learning rate")
                optimizer.param_groups[1]["lr"] *= 0.1  # head
                optimizer.param_groups[0]["lr"] *= 0.5  # base
    except KeyboardInterrupt:
        print("[INFO] Stopping early")


def test_model(model, data_loader, f1_average="macro", threshold=0.5):
    # Testing phase
    model.to(device)
    model.eval()
    test_hamming = 0.0
    test_f1 = 0.0
    test_emr = 0.0
    num_samples = 0
    num_batches = 0
    print("\n----- Model Evaluation -----")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask, targets = batch
            out = model(input_ids.to(device), attention_mask=attention_mask.to(device))
            sigmoids = torch.sigmoid(out.logits)
            sigmoids[sigmoids >= threshold] = 1
            sigmoids[sigmoids < threshold] = 0
            predictions = sigmoids.detach().cpu()
            test_hamming += hamming_loss(targets, predictions)
            test_f1 += f1_score(
                targets, predictions, average=f1_average, zero_division=0
            )
            test_emr += accuracy_score(targets, predictions)
            num_samples += len(targets)
            num_batches += 1
    test_hamming /= num_batches
    test_f1 /= num_batches
    test_emr /= num_batches
    print(
        f"[TEST] hamming loss: {test_hamming}\n"
        f"       F1-score ({f1_average}): {test_f1}\n"
        f"       exact match ratio: {test_emr * 100} %"
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit("Required arguments: DATA_DIR MODEL_DIR TOPIC_CODES")
    train(sys.argv[1], sys.argv[2], sys.argv[3])
