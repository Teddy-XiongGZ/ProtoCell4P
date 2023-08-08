# ProtoCell4P: An Explainable Prototype-based Neural Network for Patient Classification Using Single-cell RNA-seq

This repo contains the source code for our [manuscript](https://doi.org/10.1093/bioinformatics/btad493) to Bioinformatics.

## Setup
### lupus dataset
- Change directory to `./data/lupus`
- Follow the instruction of `./data/lupus/download.txt`
### cardio dataset
- Change directory to `./data/cardio`
- Follow the instruction of `./data/cardio/download.txt`
### covid dataset
- Change directory to `./data/covid`
- Follow the instruction of `./data/covid/download.txt`

## Usage
- Change directory to `./src`
### Run ProtoCell4P
- Run `sh run.sh`
### Run BaseModel
- Run `sh run_base.sh`
### Run Ablation Studies
- Run `sh run_ablation.sh`

## Citation
If you find our research useful, please consider citing:

```
@article{xiong2023protocell4p,
  title={ProtoCell4P: An Explainable Prototype-based Neural Network for Patient Classification Using Single-cell RNA-seq},
  author={Xiong, Guangzhi and Bekiranov, Stefan and Zhang, Aidong},
  journal={Bioinformatics},
  pages={btad493},
  year={2023},
  publisher={Oxford University Press}
}
```
