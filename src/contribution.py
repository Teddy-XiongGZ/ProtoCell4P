from config import *
import argparse
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def compute_contribution(args):

    print("Checkpoint path:", os.path.join(args.checkpoint_dir,args.checkpoint_name))

    config = Config_eval(
        data = args.data,
        task = args.task,
        model = args.model,
        split_ratio = args.split_ratio,
        batch_size = args.batch_size,
        n_proto = args.n_proto,
        h_dim = args.h_dim,
        z_dim = args.z_dim,
        n_layers = args.n_layers,
        d_min = args.d_min,
        device = args.device,
        seed = args.seed,
        subsample = args.subsample,
        load_ct = args.load_ct,
        checkpoint_dir = args.checkpoint_dir,
        checkpoint_name = args.checkpoint_name
    )

    config.reason(dataset="test")

    n_class = len(config.class_id)
    state_cont = [(n_class * config.CLOG[:,i] - config.CLOG.sum(1)) / (n_class - 1) for i in range(len(config.class_id))]
    for t, ct in enumerate(config.ct_id):
        print(t,ct)
        print((" | ".join(["{:s}: {:.2f}"] * len(config.class_id))).format(*[j for i in range(len(config.class_id)) for j in (config.class_id[i], state_cont[i][config.CT==t].mean().cpu().item())]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", default="lupus")
    parser.add_argument("--task", default=None)
    parser.add_argument("--model", default="ProtoCell")
    parser.add_argument("--split_ratio", nargs='+', type=float, default=[0.5, 0.25, 0.25])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_proto", type=int, default=4)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_min", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subsample", action=argparse.BooleanOptionalAction)
    parser.add_argument("--load_ct", action=argparse.BooleanOptionalAction)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pt")
    parser.add_argument("--type", default="total")
    parser.add_argument("--n_sample", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)    
    args = parser.parse_args()

    args.load_ct = False if args.load_ct is None else args.load_ct
    args.subsample = False if args.subsample is None else args.subsample

    compute_contribution(args)

#     python contribution.py --data lupus --task disease --batch_size 4 --n_proto 4 --seed 1 --model ProtoCell --subsample --load_ct --device cuda:0 --k 20 --type sample --checkpoint_dir '../checkpoint/lupus/disease/ProtoCell_4_proto_1' --checkpoint_name 'best_model.pt'