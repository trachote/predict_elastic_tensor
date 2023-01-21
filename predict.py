import time
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import os
import pandas as pd
from runner import Runner
import pickle

def main(cfg):
    save_dir = Path("save_params")
    pred_df = pd.read_json("data/" + cfg.df_filename + ".json")
    pred_df = pred_df[pred_df['edge_size'] <= 100]
    model = Runner(cfg)
    uij, cij = model.predict(pred_df, cfg.filename, cfg.batch_size, cfg.use_gt_label)
    
    pred_filename = 'predict_' + cfg.filename + '.pickle'
    with open(save_dir / pred_filename, 'wb') as f:
        pickle.dump({'strain_energy_tensor': uij, 'elastic_tensor': cij}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True)
    parser.add_argument('--df_filename', required=True)
    parser.add_argument('--batch_size', type=int, default=21)
    parser.add_argument('--use_gt_label', default=False)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load('save_params/config.yaml')
    cfg.filename = args.filename
    cfg.df_filename = args.df_filename
    cfg.batch_size = args.batch_size
    cfg.use_gt_label = args.use_gt_label
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))

    main(cfg)
