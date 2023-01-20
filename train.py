import time
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import os
import pandas as pd
from runner import Runner


def main(cfg):
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    save_dir = Path("save_params")
    os.makedirs(save_dir, exist_ok = True)
    (save_dir / "config.yaml").write_text(yaml_conf)

    train_df = pd.read_json("data/mp_train_clean.json")
    val_df = pd.read_json("data/mp_val_clean.json")

    model = Runner(cfg)
    model.train(train_df, val_df, cfg.filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--filename', required=True)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg.filename = args.filename
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))

    main(cfg)
