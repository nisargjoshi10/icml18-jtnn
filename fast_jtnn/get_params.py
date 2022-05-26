import torch
import torch.nn as nn
import torch.nn.functional as F

import jtnn_vae
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from jtmpn import JTMPN
from mpn import MPN
import argparse
import sys
sys.path.append('../')
from fast_jtnn import *


parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
# parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=128)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)


#encoder
jtnn_enc = JTNNEncoder(args.hidden_size, args.depthT, nn.Embedding(vocab.size(), args.hidden_size))
jtnn_enc_params = sum([x.nelement() for x in jtnn_enc.parameters()]) / 1000

mpn = MPN(args.hidden_size, args.depthG)
mpn_params = sum([x.nelement() for x in mpn.parameters()]) / 1000

nn_models = nn.Linear(args.hidden_size, args.latent_size) #multiply with 4 since all the models would have the same no. of parameters
nn_models_params = sum([x.nelement() for x in nn_models.parameters()])*4 / 1000

enc_total = jtnn_enc_params + mpn_params + nn_models_params
print("Encoder total #Params: %dK" % enc_total)

#decoder
jtmpn = JTMPN(args.hidden_size, args.depthG)
jtmpn_params = sum([x.nelement() for x in jtmpn.parameters()]) / 1000

dec = JTNNDecoder(vocab, args.hidden_size, args.latent_size, nn.Embedding(vocab.size(), args.hidden_size))
dec_params = sum([x.nelement() for x in dec.parameters()]) / 1000

A_assm = nn.Linear(args.latent_size, args.hidden_size)
A_assm_params = sum([x.nelement() for x in A_assm.parameters()]) / 1000

dec_total = jtmpn_params + dec_params + A_assm_params
print("decoder total #Params: %dK" % dec_total)

total_params = enc_total + dec_total
print(" total #Params: %dK" % total_params)
