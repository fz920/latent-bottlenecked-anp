# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
####################################################################################


import os
import os.path as osp
import argparse
import yaml
import torch
import time
import matplotlib.pyplot as plt

# Monkey patch collections
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdict import AttrDict
from tqdm import tqdm

from data.image import img_to_task, task_to_img, pred_to_img
from data.celeba import CelebA
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode',
            choices=['train', 'eval', 'eval_multiple_runs', 'visualize', 'attention_map'],
            default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', action='store_true')

    # Data
    parser.add_argument('--max_num_points', type=int, default=200)
    parser.add_argument('--max_num_train_target_points', type=int, default=None) # Sets the max number of training target points
    parser.add_argument('--resolution', type=int, default=32)

    # Model
    parser.add_argument('--model', type=str, default="tnpd")

    # Train
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=5)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # PNP Arguments
    parser.add_argument('--num_latents', type=int, default=None)
    parser.add_argument('--num_latents_per_layer', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--emb_depth', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--dropout', type=int, default=0.0)
    parser.add_argument('--num_layers', type=int, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)


    args = parser.parse_args()

    if args.expid is not None:
        args.root = osp.join(results_path, f'celeba{args.resolution}', args.model, args.expid)
    else:
        args.root = osp.join(results_path, f'celeba{args.resolution}', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/celeba{args.resolution}/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in vars(args).items(): # Override the default arguments
        if key in config and val is not None:
            config[key] = val
            print(f"Overriding argument {key}: {config[key]}")


    if args.pretrain:
        assert args.model == 'tnpa'
        config['pretrain'] = args.pretrain


    model = model_cls(**config)
    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'visualize':
        num_cpoints_ls = [1, int(0.001*args.resolution**2), int(0.005*args.resolution**2),
                          int(0.01*args.resolution**2), int(0.1*args.resolution**2), int(args.resolution**2)]
        pred_dist, task_img = pred_dists(args, model, num_cpoints_ls=num_cpoints_ls)
        visualise_img(args, pred_dist, task_img)
    elif args.mode == 'attention_map':
        visualize_attention_map(args, model)
    else:
        raise NotImplementedError

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_ds = CelebA(train=True, resolution=args.resolution)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in tqdm(range(start_epoch, args.num_epochs+1)):
        model.train()
        for (x, _) in tqdm(train_loader, ascii=True):
            x = x.cuda()
            batch = img_to_task(x,
                max_num_points=args.max_num_points,
                max_num_target_points=args.max_num_train_target_points)
            optimizer.zero_grad()
            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, num_samples=args.train_num_samples)
            else:
                outs = model(batch)
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)



        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)

def gen_evalset(args):

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = CelebA(train=False, resolution=args.resolution)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False, num_workers=4)

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        # No max_num_target_points bc it evaluates on all datapoints. 
        batches.append(img_to_task(
            x, max_num_points=args.max_num_points, 
            t_noise=args.t_noise)
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, f'celeba{args.resolution}')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = 'no_noise.tar' if args.t_noise is None else \
            f'{args.t_noise}.tar'
    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, f'celeba{args.resolution}')
    if not osp.isdir(path):
        os.makedirs(path)
    filename = f'no_noise.tar' if args.t_noise is None else \
        f'_{args.t_noise}.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()

            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line

def pred_dists(args, model, num_cpoints_ls):
    torch.manual_seed(35)
    torch.cuda.manual_seed(35)
    ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    eval_ds = CelebA(train=False, resolution=args.resolution)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=1,
            shuffle=True, num_workers=4)

    # generate one batch per number of context points specified
    eval_batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        for num_cpoints in num_cpoints_ls:
            eval_batches.append(img_to_task(
                x, num_ctx=num_cpoints, target_all=True, pred_all=True,
                t_noise=args.t_noise)
            )
        break

    model.eval()
    pred_dist = []
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            pred_tar = model.predict(batch.xc, batch.yc, batch.xt)
            pred_dist.append(pred_tar)
    return pred_dist, eval_batches  # return the predicted distributions and the context images


def visualise_img(args, pred_dist, eval_batches):
    shape = (3, args.resolution, args.resolution)
    for i, (dist, batch) in enumerate(zip(pred_dist, eval_batches)):
        base_path = f'/rds/user/fz287/hpc-work/MLMI4/lbanp_figures/{args.expid}'
        os.makedirs(base_path, exist_ok=True)

        mean = dist.mean.cpu().detach()
        variance = dist.variance.cpu().detach()

        task_img, _ = task_to_img(batch.xc, batch.yc, batch.xt, batch.yt, shape)

        # Save the task image
        plt.figure(figsize=(4, 4), dpi=200)
        plt.imshow(task_img[0].permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'{base_path}/task_img_{i+1}.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Save the mean image
        mean_img = pred_to_img(batch.xt, mean, shape)[0].permute(1, 2, 0)
        plt.figure(figsize=(4, 4), dpi=200)
        plt.imshow(mean_img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{base_path}/mean_img_{i+1}.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Save the variance image
        var_img = pred_to_img(batch.xt, variance, shape, variance=True)[0].permute(1, 2, 0)
        # def normalize(image):
        #     min_val = image.min()
        #     max_val = image.max()
        #     normalized = (image - min_val) / (max_val - min_val)
        #     return normalized
        # var_img_normalized = normalize(var_img)
        plt.figure(figsize=(4, 4), dpi=200)
        plt.imshow(var_img*5, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{base_path}/var_img_{i+1}.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()

def visualize_attention_map(args, model):
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    eval_ds = CelebA(train=False, resolution=args.resolution)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=1,
            shuffle=True, num_workers=4)

    # generate one batch per number of context points specified
    eval_batches = []
    num_cpoints_ls = [int(32*32)]
    for num_cpoints in num_cpoints_ls:
        for x, _ in tqdm(eval_loader, ascii=True):
            eval_batches.append(img_to_task(
                x, num_ctx=num_cpoints, target_all=True, pred_all=True,
                t_noise=args.t_noise)
            )
            break

    model.eval()
    pred_dist = []
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            pred_tar, weights_ls = model.predict(batch.xc, batch.yc, batch.xt, attn_map=True)
            pred_dist.append(pred_tar)

    for i, attention_weights in enumerate(weights_ls):
        for j in range(attention_weights.shape[1]):
            plt.figure(figsize=(10, 10), dpi=400)
            plt.matshow(attention_weights[0, j, :].reshape(32, 32))
            plt.axis('off')
            plt.savefig(f'/rds/user/fz287/hpc-work/MLMI4/lbanp_figures/attention_map_lbanp/attn_map{j}_layer{i}.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
