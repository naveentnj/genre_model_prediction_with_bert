
import torch

class ClassificationDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, item):
        text = str(self.data[item]["synopsis"])
        # Need to use int conversion
        target = int(self.data[item]["genre"])
        # I used word truncate instead of truncation
        inputs = self.tokenizer(text, max_length = 20, padding = "max_length", truncation = True)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return{
            "input_ids" : torch.tensor(ids, dtype = torch.long),
            "attention_mask" : torch.tensor(mask, dtype = torch.long),
            "labels" : torch.tensor(target, dtype = torch.long)
        }