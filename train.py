from dataset import ClassificationDataset
import config
from typing import Any
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn import metrics
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/9373
# downgrading transformers from 4.26.1 to 4.19.2 works
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    accuracy = metrics.accuracy_score(labels, predictions)
    return {"accuracy" : accuracy}

def train():
    df = load_dataset(config.DATASET_NAME)
    df = df.class_encode_column("genre")

    df_train = df["train"]
    df_test = df["test"]

    temp_df = df["train"].train_test_split(test_size = 0.25, stratify_by_column = "genre")

    df_train = temp_df["train"]
    df_val = temp_df["test"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast = True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", \
                                                               num_labels = len(df_train.features["genre"]._int2str))
    
    train_dataset = ClassificationDataset(df_train, tokenizer)
    val_dataset = ClassificationDataset(df_val, tokenizer)
    test_dataset = ClassificationDataset(df_test, tokenizer)

    args = TrainingArguments( 
        "model",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate = config.LEARNING_RATE,
        per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = config.VALID_BATCH_SIZE,
        num_train_epochs = 1,
        weight_decay = 0.01,
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        report_to = "none",
        save_total_limit = 1 
    )

    trainer = Trainer(
        model,
        args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    
    trainer.train()
    preds = trainer.predict(test_dataset).predictions
    preds = np.argmax(preds, axis = 1)


    submission = pd.DataFrame({"id":df_test["id"], "genre": preds})
    submission.loc[:, "genre"] = submission.genre.apply(lambda x : df_train.features["genre"].int2str(x) )

    submission.to_csv("result/submission.csv", index = False)

if __name__ == "__main__":
    train()
