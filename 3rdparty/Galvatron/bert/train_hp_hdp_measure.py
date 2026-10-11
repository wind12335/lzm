"""按 Google HDP 论文(ICLR'18, "A Hierarchical Model for Device Placement")的时延
测量协议,计时 Galvatron 一个训练步的时延。

HDP 原文 §3 "Measuring Reward" 监测的数值:
  r_d    = 完成一个训练步(1 次 forward + 1 次 backward + 1 次参数更新)的秒数;
           每个策略跑 10 步,丢弃前 5 步(warmup),取后 5 步的中位数;
  reward = -sqrt(r_d),非法策略(OOM)reward = -10。

本脚本与 train_hp_layerwise.py 的唯一区别在计时段(模型构建/并行策略完全复用原代码):
  - 每步两侧 torch.cuda.synchronize() 后计时,得到逐 step 时延;
  - warmup 结束时 reset_peak_memory_stats,结束后读 max_memory_allocated/reserved;
  - 每个 rank 打印一行 "HDP_RESULT_JSON {...}",供 benchmark_step_latency.py 汇总。

用法(通常由 benchmark_step_latency.py 驱动,也可手动):
  python -m torch.distributed.launch --nproc_per_node=2 --master_port 9998 train_hp_hdp_measure.py \
      --global_train_batch_size 8 --hidden_size 1280 --num_hidden_layers 32 \
      --num_attention_heads 16 --seq_length 512 --epochs 5 --lr 1e-4 \
      --adam_weight_decay 0.01 --dropout_prob 0.1 --check_loss 0 --profile 0 \
      --apply_strategy 0 --galvatron_config_path ./configs/galvatron_config_8gpus_1280hidden_32layers_8G_full.json
"""
import json
import time

import torch
from torch.optim import Adam
from transformers import BertConfig, BertForPreTraining
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed.rpc as rpc

from dataloader import DataLoaderForBert
from train_hp_layerwise import add_arguments, set_seed, model_forward
from megatron.initialize import initialize_megatron
from megatron import get_args
from hybrid_parallel_model import get_hybrid_parallel_configs, construct_hybrid_parallel_model, overwrite_megatron_args


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    rpc.init_rpc(name="worker%d" % rank, rank=rank, world_size=world_size)

    hybrid_parallel_configs = get_hybrid_parallel_configs(args)

    if local_rank == 0:
        print("Creating Dataloader...")
    dataset = DataLoaderForBert()
    trainloader = DataLoader(dataset=dataset,
                             batch_size=args.global_train_batch_size // world_size,
                             sampler=DistributedSampler(dataset, shuffle=True))

    if local_rank == 0:
        print("Creating Model...")
    config = BertConfig(vocab_size=args.vocab_size,
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers,
                        num_attention_heads=args.num_attention_heads,
                        intermediate_size=args.hidden_size * 4,
                        max_position_embeddings=args.seq_length,
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob)
    overwrite_megatron_args(config, args)
    bert_model = BertForPreTraining(config)
    model = construct_hybrid_parallel_model(model=bert_model,
                                            model_config=config,
                                            training_args=args,
                                            hybrid_parallel_configs=hybrid_parallel_configs)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    warmup, measure = args.warmup_steps, args.measure_steps
    total_steps = warmup + measure
    if local_rank == 0:
        print("[HDP protocol] warmup=%d 步(丢弃), measure=%d 步(取中位数)" % (warmup, measure))

    step_times = []
    it = 0
    for ep in range(args.epochs):
        for batch in trainloader:
            if it == warmup:  # warmup 结束,重置显存峰值统计(其后峰值即训练段真实占用)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(local_rank)
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label = \
                [t.to(device) for t in batch]
            torch.cuda.synchronize()
            t0 = time.time()
            loss, _, _ = model_forward(config, model, input_ids, attention_mask,
                                       token_type_ids, masked_lm_labels, next_sentence_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if it >= warmup:
                step_times.append(time.time() - t0)
            it += 1
            if it >= total_steps:
                break
        if it >= total_steps:
            break

    n = len(step_times)
    step_times.sort()
    r_med = step_times[n // 2] if n % 2 == 1 else (step_times[n // 2 - 1] + step_times[n // 2]) / 2
    result = {
        "rank": rank, "world_size": world_size, "n_measured": n,
        "warmup_steps": warmup, "measure_steps": measure,
        "step_times_s": [round(t, 4) for t in step_times],
        "r_d_median_s": round(r_med, 4),
        "r_d_min_s": round(step_times[0], 4), "r_d_max_s": round(step_times[-1], 4),
        "reward_neg_sqrt": round(-r_med ** 0.5, 4),
        "peak_mem_allocated_MB": round(torch.cuda.max_memory_allocated(local_rank) / 1024 / 1024, 1),
        "peak_mem_reserved_MB": round(torch.cuda.max_memory_reserved(local_rank) / 1024 / 1024, 1),
    }
    print("HDP_RESULT_JSON " + json.dumps(result), flush=True)


def add_hdp_arguments(parser):
    group = parser.add_argument_group(title='hdp measure arguments')
    group.add_argument("--warmup_steps", type=int, default=5,
                       help="HDP 协议:丢弃的 warmup 步数(原文 5)")
    group.add_argument("--measure_steps", type=int, default=5,
                       help="HDP 协议:计时的步数,取中位数(原文 5)")
    return parser


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=lambda p: add_hdp_arguments(add_arguments(p)))
    args = get_args()
    set_seed()
    train(args)
