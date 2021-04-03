import xml.etree.ElementTree as ET

import torch


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, code2idx, max_seq_len=None, predict=False):
        self.data = data
        self.tokenizer = tokenizer
        self.code2idx = code2idx
        self.predict = predict
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        if self.predict:
            text, file = self.data[idx]
        else:
            text, codes = self.data[idx]
        tokenized_text = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)
        if self.predict:
            # When predicting, we don't have targets, but we want the name
            # of the document, for which we are predicting, here `file`
            return input_ids, attention_mask, file
        code_indeces = torch.tensor([self.code2idx[code] for code in codes])
        # target will be a 'multi hot' tensor
        targets = torch.zeros(len(self.code2idx)).scatter_(0, code_indeces, 1.0)
        return input_ids, attention_mask, targets

    def __len__(self):
        return len(self.data)


def read_topic_codes(path, code_filter=None):
    codes = []
    code2desc = dict()
    with open(path) as fh:
        for line in fh:
            if line.startswith(";"):
                continue
            code, description = line.strip().split("\t")
            if code_filter and code not in code_filter:
                continue
            codes.append(code)
            code2desc[code] = description
    code2idx = {code: i for i, code in enumerate(codes)}
    idx2code = {i: code for i, code in enumerate(codes)}
    return code2idx, idx2code, code2desc


def read_document(path, code_filter=None, include_codes=True):
    tree = ET.parse(path)
    text = "".join(
        filter(None, [tree.find("headline").text] + list(tree.find("text").itertext()))
    )
    if not include_codes:
        # This is executed when documents are read for prediction
        return text, str(path)
    codes = [
        code.get("code")
        for codes in tree.find("metadata").findall("codes")
        for code in codes
        if codes.get("class") == "bip:topics:1.0"
    ]
    if code_filter:
        codes = list(code_filter.intersection(codes))
    if not codes:
        return None
    return text, codes
