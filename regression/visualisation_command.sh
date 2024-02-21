# require gpu
sintr -A MLMI-fz287-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 0:5:0 --qos=INTR

# celeba 128 -- number of latent vectors 8
python celeba.py --mode visualize --expid lbanp_celeba128_8 --model lbanp --num_latents 8 --resolution 128 --max_num_points 1600

# celeba 128 -- number of latent vectors 128
python celeba.py --mode visualize --expid lbanp_celeba128_128 --model lbanp --num_latents 128 --resolution 128 --max_num_points 1600
