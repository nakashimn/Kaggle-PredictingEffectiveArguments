from baseline import *
from model_v1 import PeModelV1

config["mode"] = "train"
config["n_splits"] = 3
config["experiment_name"] = "fp-distilbert-v1"
config["path"]["model_dir"] = "/kaggle/input/model/fp-distilbert-v1/"
config["pred_ensemble"] = True
config["model"] = {
    "name": "PeModelV1",
    "base_model_name": "/kaggle/input/distilbertbaseuncased",
    "dim_feature": 768,
    "num_class": 3,
    "dropout_rate": 0.2,
    "freeze_base_model": False,
    "loss": {
        "name": "nn.CrossEntropyLoss",
        "params": {
            "weight": None
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
config["model"]["conv_0"] = {
    "params": {
        "in_channels": config["model"]["dim_feature"],
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "bias": True
    }
}
config["model"]["conv_1"] = {
    "params": {
        "in_channels": 128,
        "out_channels": config["model"]["num_class"],
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "bias": True
    }
}
config["earlystopping"] = {
    'patience': 3
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
    }
}

if __name__=="__main__":

    # preprocessor
    text_cleaner = TextCleaner()

    fix_seed(config["random_seed"])

    if config["mode"]=="train":

        # logger
        mlflow_logger = create_mlflow_logger(config)
        mlflow_logger.log_hyperparams(config)

        # Setting Dataset
        df_train = pd.read_csv(config["path"]["traindata"])
        df_train["essay"] = df_train["essay_id"].apply(fetchEssay, args=(config["path"]["trainessay"],))
        df_train = text_cleaner.clean(df_train, "discourse_text")
        df_train = text_cleaner.clean(df_train, "essay")

        # Training
        trainer = Trainer(
            eval(config["model"]["name"]),
            PeDataModule,
            PeDataset,
            AutoTokenizer,
            ValidResult,
            MinLoss,
            df_train,
            config,
            None,
            mlflow_logger
        )
        trainer.run()

        # Validation Result
        confmat = ConfusionMatrix(
            trainer.val_probs.values,
            trainer.val_labels.values,
            config["Metrics"]
        )
        fig_confmat = confmat.draw()
        fig_confmat.savefig(f"{config['path']['temporal_dir']}/confmat.png")
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            f"{config['path']['temporal_dir']}/confmat.png"
        )

        f1score = F1Score(
            trainer.val_probs.values,
            trainer.val_labels.values,
            config["Metrics"]
        )
        f1scores = f1score.calc()
        mlflow_logger.log_metrics({
            "macro_f1_score": f1scores["macro"],
            "micro_f1_score": f1scores["micro"]
        })

        logloss = LogLoss(
            trainer.val_probs.values,
            trainer.val_labels.values,
            config["Metrics"]
        )
        log_loss = logloss.calc()
        mlflow_logger.log_metrics({
            "logloss": log_loss
        })

        # output
        print(f"macro_f1_score: {f1scores['macro']:.04f}")
        print(f"micro_f1_score: {f1scores['micro']:.04f}")
        print(f"logloss: {log_loss:.04f}")

        # update model
        update_model(config)


    if config["mode"]=="test":

        # Setting Dataset
        df_test = pd.read_csv(config["path"]["testdata"])
        df_test["essay"] = df_test["essay_id"].apply(fetchEssay, args=(config["path"]["testessay"],))
        df_test = text_cleaner.clean(df_test, "discourse_text")
        df_test = text_cleaner.clean(df_test, "essay")

        # Prediction
        if config["pred_ensemble"]:
            cls_predictor = PredictorEnsemble
        else:
            cls_predictor = Predictor
        predictor = cls_predictor(
            eval(config["model"]["name"]),
            PeDataModule,
            PeDataset,
            AutoTokenizer,
            df_test,
            config,
            None
        )
        predictor.run()

        # output
        submission = pd.concat([
            df_test["discourse_id"],
            pd.DataFrame(predictor.probs, columns=config["labels"])
        ], axis=1)
        submission.to_csv("submission.csv", index=None)
