set -x
r(){ n=$1; shift; python3 train_tf.py "$@" --out runs/$n.npy > runs/$n.json 2> runs/$n.err; }
r aux_L2_s0  --variant aux  --feats cH/feats_fit.npz
r feed_L2_s0 --variant feed --feats cH/feats_fit.npz
r base_L2_s1 --variant base --seed 1 --feats cH/feats_fit.npz
r aux_L2_s1  --variant aux  --feats cH/feats_fit.npz --seed 1
r feed_L2_s1 --variant feed --feats cH/feats_fit.npz --seed 1
r base_L4_s0 --variant base --layers 4 --feats cH/feats_fit.npz
r aux_L4_s0  --variant aux  --layers 4 --feats cH/feats_fit.npz
r feed_L4_s0 --variant feed --layers 4 --feats cH/feats_fit.npz
r oracleaux_L2_s0  --variant aux  --feats cH/feats_true.npz
r oraclefeed_L2_s0 --variant feed --feats cH/feats_true.npz
r base_L2_s0_8k --variant base --steps 8000 --feats cH/feats_fit.npz
