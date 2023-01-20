import time
from omegaconf import OmegaConf
import argparse
import pandas as pd
from runner import Runner


def main(cfg):
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    ("save_params" / "hparams.yaml").write_text(yaml_conf)

    train_df = pd.read_json("data/mp_train_clean.json")
    val_df = pd.read_json("data/mp_val_clean.json")

    model = Runner(cfg)
    model.train(train_df, val_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))

    main(cfg)
