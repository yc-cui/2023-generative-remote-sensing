import os
from PIL import Image
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import evaluate
import transformers

transformers.logging.set_verbosity_info()

os.chdir("/data/cyc/2023-generative-remote-sensing")


def load_custom_dataset(flists):
    # read all files and concatenate in a list
    lines = []
    for file in flists:
        with open(file, "r") as f:
            lines += f.readlines()

    # split by ','
    lines = [line.strip().split(',') for line in lines]

    # create dictionary
    dataset_dict = {"image": [line[0] for line in lines], "label": [int(line[1]) for line in lines]}

    # create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset

train_set = load_custom_dataset(["datasets/new/AID_test0.2/new.flist", "datasets/data/cls/AID_test0.2/train.flist"])

test_set = load_custom_dataset(["datasets/data/cls/AID_test0.2/test.flist"])



from datasets import load_dataset

food = load_dataset("food101", split="train[:5000]")
food = food.train_test_split(test_size=0.2)

data_list = glob("datasets/decomp/AID/*/*")
target_list = list(map(lambda x: x.split("/")[-2], data_list))
unique_lst = np.unique(target_list)
id2label = {str(index) : value for index, value in enumerate(unique_lst)}
label2id = {value : str(index) for index, value in enumerate(unique_lst)}

from transformers import AutoImageProcessor
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(examples):
    examples["pixel_values"] = [train_transforms(Image.open(img).convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def preprocess_val(examples):
    examples["pixel_values"] = [val_transforms(Image.open(img).convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

train_set = train_set.with_transform(preprocess_train)
test_set = test_set.with_transform(preprocess_val)



from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

acc_metric = evaluate.load("accuracy")
f_metric = evaluate.load("f1")
r_metric = evaluate.load('recall')
p_metric = evaluate.load('precision')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=1)
    acc_at_5 = sum([l in p for l, p in zip(labels, np.argsort(-logits)[:,0:5])]) / len(labels)
    acc = acc_metric.compute(predictions=pred, references=labels)
    f1_macro = f_metric.compute(predictions=pred, references=labels, average="macro")
    f1_micro = f_metric.compute(predictions=pred, references=labels, average="micro")
    f1_weighted = f_metric.compute(predictions=pred, references=labels, average="weighted")
    r_macro = r_metric.compute(predictions=pred, references=labels, average="macro")
    r_micro = r_metric.compute(predictions=pred, references=labels, average="micro")
    r_weighted = r_metric.compute(predictions=pred, references=labels, average="weighted")
    p_macro = p_metric.compute(predictions=pred, references=labels, average="macro")
    p_micro = p_metric.compute(predictions=pred, references=labels, average="micro")
    p_weighted = p_metric.compute(predictions=pred, references=labels, average="weighted")
    
    return {
        "accuracy": acc["accuracy"],
        "accuracy@5": acc_at_5,
        "f1_macro": f1_macro["f1"],
        "f1_micro": f1_micro["f1"],
        "f1_weighted": f1_weighted["f1"],
        "r_macro": r_macro["recall"],
        "r_micro": r_micro["recall"],
        "r_weighted": r_weighted["recall"],
        "p_macro": p_macro["precision"],
        "p_micro": p_micro["precision"],
        "p_weighted": p_weighted["precision"],
    }

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id.keys()),
    id2label=id2label,
    label2id=label2id,
)

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "cls"
os.environ["WANDB_DIR"] = "tasks/logs/cls/AID_test0.2"
os.environ["WANDB_TAGS"] = "cls,gen10000,AID,test0.2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %env WANDB_MODE=offline
# %env WANDB_PROJECT=cls
# %env WANDB_DIR=tasks/logs/cls
# %env CUDA_VISIBLE_DEVICES=0

os.makedirs("tasks/logs/cls/AID_test0.2", exist_ok=True)

training_args = TrainingArguments(
    output_dir="tasks/logs/cls/AID_test0.2",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    num_train_epochs=200,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)


train_results = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


