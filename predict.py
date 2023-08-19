import time
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import os
import pandas as pd
from runner import Runner
import pickle
import warnings

def main(cfg):
    save_dir = Path(cfg.ckpt_dir)        
    pred_df = pd.read_json(cfg.json_path)
    
    if not cfg.echo_warning:
        warnings.filterwarnings('ignore')
        
    model = Runner(cfg)
    uij, cij = model.predict(pred_df, cfg.limit_edge_size, cfg.use_gt_label)
    
    pred_filename = 'predictions.pickle'
    with open(save_dir / pred_filename, 'wb') as f:
        pickle.dump({'strain_energy_tensor': uij, 'elastic_tensor': cij}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--suffix', type=str, default='pretrained')
    parser.add_argument('--batch_size', type=int, default=21)
    parser.add_argument('--use_gt_label', default=False)
    parser.add_argument('--limit_edge_size', default=False)
    parser.add_argument('--echo_warning', type=bool, default=False)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(f'{args.ckpt_dir}/config.yaml')
    cfg.ckpt_dir = args.ckpt_dir
    cfg.json_path = args.json_path
    cfg.batch_size = args.batch_size
    cfg.use_gt_label = args.use_gt_label
    cfg.limit_edge_size = args.limit_edge_size
    cfg.echo_warning = args.echo_warning
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))

    main(cfg)
