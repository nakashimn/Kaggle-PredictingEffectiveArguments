config = {
    "n_splits": 2,
    "pseudo": True,
    "pseudo_confidential_threshold": 0.9,
    "random_seed": 57,
    "label": "discourse_effectiveness",
    "group": "essay_group",
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
    "experiment_name": "fp-distilbert-v0-pseudo-debug",
    "path": {
    "traindata": "/kaggle/input/debug-psuedo/train_and_pseudo.csv",
        "trainessay": "/kaggle/input/debug-psuedo/train/",
        "testdata": "/kaggle/input/feedback-prize-effectiveness/test.csv",
        "testessay": "/kaggle/input/feedback-prize-effectiveness/test/",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/fp-distilbert-v0-pseudo-debug/"
    },
    "modelname": "best_loss",
    "pred_ensemble": True,
    "train_with_alldata": False
}
config["model"] = {
    "base_model_name": "/kaggle/input/distilbertbaseuncased",
    "dim_feature": 768,
    "num_class": 3,
    "dropout_rate": 0.5,
    "freeze_base_model": False,
    "loss": {
        "name": "PseudoLoss",
        "params": {
            "LossFunction": "nn.CrossEntropyLoss",
            "alpha": 3,
            "epoch_th_lower": 1,
            "epoch_th_upper": 10
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
    "monitor": "pseudo_labeled_ratio",
    "mode": "max",
    "patience": 1,
    "stopping_threshold": 0.9
}
config["checkpoint"] = {
    "dirpath": config["path"]["temporal_dir"],
    "monitor": "pseudo_labeled_ratio",
    "save_top_k": 1,
    "mode": "max",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 2,
    "accumulate_grad_batches": 1,
    "fast_dev_run": False,
    "deterministic": True,
    "num_sanity_val_steps": 0,
    "resume_from_checkpoint": None,
    "precision": 16
}
config["kfold"] = {
    "name": "StratifiedGroupKFold",
    "params": {
        "n_splits": config["n_splits"],
        "shuffle": True,
        "random_state": config["random_seed"]
    }
}
config["datamodule"] = {
    "dataset":{
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "labels": config["labels"],
        "use_fast_tokenizer": True,
        "max_length": 512,
        "discourse_effectiveness": {l : i for i, l in enumerate(config["labels"])},
        "discourse_type": {tp : i for i, tp in enumerate(config["types"])}
    },
    "train_loader": {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False
    },
    "pred_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": False,
        "drop_last": False
    },
    "pseudo_confidential_threshold": config["pseudo_confidential_threshold"]
}
config["Metrics"] = {
    "label": config["labels"]
}
