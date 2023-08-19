# Predicting strain energy and elastic tensors
StrainNet code is an implementation of a paper [StrainNet: Predicting crystal structure elastic properties using SE(3)-equivariant graph neural networks](https://arxiv.org/abs/2306.12818).

StrainNet can be employed to train and/or predict a strain energy density in a unit of eV/atom.
21 strain energy density of each crystal structure will be predicted and can be converted to an elastic tensor.

training command:
```
python train.py --config_path conf/config.yaml --out_dir output
```
predicting command:
```
python predict.py --ckpt_dir output --json_path path-to-json-file
```
# SE(3)-Transformers

The SE(3)-Transformers code has been adopted with some modifications from [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://arxiv.org/abs/2006.10503). 

Please cite them as
```
@inproceedings{fuchs2020se3transformers,
    title={SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks},
    author={Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling},
    year={2020},
    booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
}
```
