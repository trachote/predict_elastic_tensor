import time
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import os
import pandas as pd
from runner import Runner
from utils.datamodule import split_dataset
import warnings

def main(cfg):
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    save_dir = Path(cfg.out_dir)
    os.makedirs(save_dir, exist_ok = True)
    (save_dir / "config.yaml").write_text(yaml_conf)

    if not (os.path.exists(f"{cfg.out_dir}/mp_train.json") and os.path.exists(f"{cfg.out_dir}/mp_val.json")):
        df = pd.read_json("data/mp_dataset.json")
        edge_style = cfg.dataset.edge_style
        print(">>>>> Dataset:\n\tCreate new dataset\n")
        train_df, val_df = split_dataset(df,
                                         val_ratio=0.1,
                                         edge_style=edge_style,
                                         seed=0)
    else:
        print(f">>>>> Dataset:\n\tUse dataset from {cfg.out_dir} directory\n")
        train_df = pd.read_json(f"{cfg.out_dir}/mp_train.json")
        val_df = pd.read_json(f"{cfg.out_dir}/mp_val.json")
    #train_df = pd.read_json("data/mp_train_clean.json")
    #val_df = pd.read_json("data/mp_val_clean.json")

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)
    
    if cfg.save_dataset:
        train_df.to_json(f"{cfg.out_dir}/mp_train.json")
        val_df.to_json(f"{cfg.out_dir}/mp_val.json")
    
    if not cfg.echo_warning:
        warnings.filterwarnings('ignore')

    model = Runner(cfg)
    model.train(train_df, val_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--suffix', type=str, default='pretrained')
    parser.add_argument('--save_dataset', type=bool, default=False)
    parser.add_argument('--echo_warning', type=bool, default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=30)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg.out_dir = args.out_dir
    cfg.suffix = args.suffix
    cfg.save_dataset = args.save_dataset
    cfg.echo_warning = args.echo_warning
    cfg.early_stopping_patience = args.early_stopping_patience
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))

    main(cfg)
