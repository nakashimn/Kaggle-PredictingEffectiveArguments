config = {
    "n_splits": 3,
    "random_seed": 57,
    "label": "discourse_effectiveness",
    "group": "essay_id",
    "labels": [
        "Ineffective",
        "Adequate",
        "Effective"
    ],
    "types": [
        "Claim",
        "Concluding Statement",
        "Counterclaim",
        "Evidence",
        "Lead",
        "Position",
        "Rebuttal"
    ],
    "experiment_name": "fp-roberta-v0",
    "path": {
        "traindata": "/kaggle/input/feedback-prize-effectiveness/train.csv",
        "trainessay": "/kaggle/input/feedback-prize-effectiveness/train/",
        "testdata": "/kaggle/input/feedback-prize-effectiveness/test.csv",
        "testessay": "/kaggle/input/feedback-prize-effectiveness/test/",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/fp-roberta-v0/"
    },
    "modelname": "best_loss",
    "pred_ensemble": True,
    "train_with_alldata": False
}
config["model"] = {
    "base_model_name": "/kaggle/input/roberta-base",
    "dim_feature": 768,
    "num_class": 3,
    "dropout_rate": 0.2,
    "freeze_base_model": False,
    "loss":{
        "name": "FocalLoss",
        "params":{
            "gamma": 2.0
        }
    },
    "optimizer":{
        "name": "optim.RAdam",
        "params":{
            "lr": 1e-5
        },
    },
    "scheduler":{
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params":{
            "T_0": 20,
            "eta_min": 1e-4,
        }
    }
}
config["earlystopping"] = {
    "patience": 5
}
config["checkpoint"] = {
    "dirpath": config["path"]["temporal_dir"],
    "monitor": "val_loss",
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 100,
    "accumulate_grad_batches": 1,
    "fast_dev_run": False,
    "deterministic": True,
    "num_sanity_val_steps": 0,
    "resume_from_checkpoint": None,
    "precision": 32
}
config["kfold"] = {
    "name": "StratifiedKFold",
    "params": {
        "n_splits": config["n_splits"],
        "shuffle": True,
        "random_state": config["random_seed"]
    }
}
config["datamodule"] = {
    "dataset": {
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "use_fast_tokenizer": True,
        "max_length": 512,
        "discourse_effectiveness": {l : i for i, l in enumerate(config["labels"])},
        "discourse_type": {tp : i for i, tp in enumerate(config["types"])}
    },
    "train_loader": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False
    },
    "pred_loader": {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": False,
        "drop_last": False
    }
}
config["Metrics"] = {
    "label": config["labels"]
}