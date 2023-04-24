import argparse
from config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="PBMC")
    parser.add_argument("--model")
    parser.add_argument("--split_ratio", nargs='+', type=float, default=[0.7, 0.15, 0.15])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_step", type=int, default=1)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_proto", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp_str", type=str, help="special string to identify an experiment")
    parser.add_argument("--task", default=None)
    parser.add_argument("--subsample", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--load_ct", action=argparse.BooleanOptionalAction)
    parser.add_argument("--d_min", type=float, default=1.0)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=1.0)
    parser.add_argument("--lambda_3", type=float, default=1.0)
    parser.add_argument("--lambda_4", type=float, default=1.0)
    parser.add_argument("--lambda_5", type=float, default=1.0)
    parser.add_argument("--lambda_6", type=float, default=1.0)
    parser.add_argument("--keep_sparse", action=argparse.BooleanOptionalAction),
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction),
    parser.add_argument("--lr_pretrain", type=float, default=1e-2)
    parser.add_argument("--max_epoch_pretrain", type=int, default=0)


    args = parser.parse_args()
    
    config = Config(
        data = args.data,
        model = args.model,
        split_ratio = args.split_ratio,
        lr = args.lr,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        test_step = args.test_step,
        h_dim = args.h_dim,
        z_dim = args.z_dim,
        n_layers = args.n_layers,
        n_proto = args.n_proto,
        device = args.device,
        seed = args.seed,
        exp_str = args.exp_str,
        task = args.task,
        subsample = False if args.subsample is None else args.subsample,
        eval = False if args.eval is None else args.eval,
        load_ct = False if args.load_ct is None else args.load_ct,
        d_min = args.d_min,
        lambda_1 = args.lambda_1,
        lambda_2 = args.lambda_2,
        lambda_3 = args.lambda_3,
        lambda_4 = args.lambda_4,
        lambda_5 = args.lambda_5,
        lambda_6 = args.lambda_6,
        keep_sparse = False if args.keep_sparse is None else args.keep_sparse,
        pretrained = False if args.pretrained is None else args.pretrained,
        lr_pretrain = args.lr_pretrain,
        max_epoch_pretrain = args.max_epoch_pretrain
    )
        
    config.train()