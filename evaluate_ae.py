import os
from os import path
from argparse import ArgumentParser

import lightning.pytorch as pl

import engine
from cod.solvers.recon_eval import ReconEvaluator

DEFAULT_OUTPUT_DIR = 'results/'

parser = ArgumentParser('Autoencoder evaluation')
parser.add_argument('model_dir', type=str, help='path to the saved weights dir')
parser.add_argument('--save_dir', '-sd', type=str, default=None, help='path to the output dir')
parser.add_argument('--data', '-d', type=str, default=None, help='name of the data config (e.g., shapenet)')
parser.add_argument('--eval', '-e', type=str, default=None, help='name of the evaluator config')
parser.add_argument('--seed', '-s', type=int, default=123456, help='evaluation random seed')
parser.add_argument('--gpus', '-g', default='[0]',
                    help='GPU to use (num. GPU or gpu ids, follow pytorch-lightning convention). e.g., "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)')


def main():
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    cfg = engine.load_config(path.join(args.model_dir, 'config.yaml'))
    pt_files = [x for x in os.listdir(args.model_dir) if x.endswith('.pt')]
    output_dir = None
    if len(pt_files) > 0:
        ## running from pretrained weights
        checkpoint_path = path.join(args.model_dir, pt_files[0])
    else:
        checkpoint_path = engine.find_best_checkpoint_path(path.join(args.model_dir, 'checkpoints'))
        output_dir = args.model_dir

    model_name = args.model_dir.strip('/').split('/')[-1]
    if args.save_dir is not None:
        output_dir = path.join(args.save_dir, model_name)
    if output_dir is None:
        output_dir = path.join(DEFAULT_OUTPUT_DIR, model_name)

    engine.set_context_from_existing(output_dir)

    model = engine.instantiate(cfg.model)
    if args.data is not None:
        data_cfg = engine.load_config(path.join('config/data', f'{args.data}.yaml'))
    elif 'data' in cfg:
        data_cfg = cfg.data
    else:
        raise Exception('data config should be specified either from command line or config file')
    dm = engine.instantiate(data_cfg)

    eval_cfg = {}
    if args.eval is not None and path.exists(args.eval):
        eval_cfg = engine.load_config(args.eval)
    evaluator = engine.instantiate(eval_cfg, ReconEvaluator, dm=dm, model=model)
    evaluator.restore_checkpoint(checkpoint_path)

    gpus = engine.parse_gpus_str(args.gpus)
    trainer = pl.Trainer(devices=gpus)
    trainer.test(model=evaluator, datamodule=dm)

    ### CD metric
    test_dataset = dm.get_dataset('test')
    test_dataset.use_queries = False
    test_dataset.use_full_surface = True
    evaluator.measure_cd(test_dataset)


if __name__ == '__main__':
    main()