import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from data_utils import DocumentDataset, read_document

threshold = 0.5  # Prediction threshold for sigmoids

batch_size = 64
num_workers = 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict(data_dir, model_dir, out_file):
    # Read topic code conversion dictionaries from a pickle in model_dir
    with open(Path(model_dir) / "topic_codes.pkl", "rb") as fh:
        code2idx, idx2code, code2desc = pickle.load(fh)
    data_files = sorted(Path(data_dir).rglob("*.xml"))
    print(f"[INFO] Reading data for {len(data_files)} documents")
    data = [read_document(file, include_codes=False) for file in data_files]
    print("[INFO] Loading model")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataset = DocumentDataset(data, tokenizer, code2idx, predict=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("[INFO] Predicting...")
    model.to(device)
    model.eval()
    out_file_contents = ""
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask, files = batch
            out = model(input_ids.to(device), attention_mask=attention_mask.to(device))
            sigmoids = torch.sigmoid(out.logits)
            sigmoids[sigmoids >= threshold] = 1
            sigmoids[sigmoids < threshold] = 0
            predictions = sigmoids.detach().cpu()
            # Get topic codes for each vector in batch
            topics = [hot2codes(vec, idx2code) for vec in predictions]
            # Join file names with corresponding topic code predictions
            out_file_contents += (
                "\n".join(
                    [
                        f"{Path(file_name).name}\t{','.join(codes)}"
                        for file_name, codes in zip(files, topics)
                    ]
                )
                + "\n"
            )
    # Write outputs to file
    with open(out_file, "w") as fh:
        fh.write(out_file_contents)
    print(f"[INFO] Done! Results saved to {out_file}")


def hot2codes(multi_hot, idx2code):
    # Converts multi-hot vectors to lists of topic codes
    indeces = multi_hot.squeeze(0).nonzero(as_tuple=True)[0].tolist()
    return [idx2code[i] for i in indeces]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit("Required arguments: DATA_DIR MODEL_DIR OUT_FILE")
    predict(sys.argv[1], sys.argv[2], sys.argv[3])
