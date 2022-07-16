from baseline import *
from model_v1 import PeModelV1

# prepare
tokenizer = AutoTokenizer.from_pretrained(
    config["datamodule"]["dataset"]["base_model_name"],
    use_fast=config["datamodule"]["dataset"]["use_fast_tokenizer"]
)

# tokenize
token = tokenizer.encode_plus(
    "test",
    truncation=True,
    add_special_tokens=True,
    max_length=config["datamodule"]["dataset"]["max_length"],
    padding="max_length"
)
ids = torch.tensor([token["input_ids"]])
masks = torch.tensor([token["attention_mask"]])

# model
model = PeModel(config["model"])
model(ids, masks)

# model_v1
modelv1 = PeModelV1(config["model"])
modelv1(ids, masks)
