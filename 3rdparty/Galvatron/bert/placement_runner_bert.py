"""HDP 风格"设备放置"执行器(PyTorch 版)

目的:让任何放置类方法(HDP 复现、你们自创的方法)输出的 device 放置方案,能在
**与 Galvatron 完全相同的框架/协议/硬件**下评测,消除 TF-vs-PyTorch 框架混杂。

放置方案 placement:长度 = 层数 + 3 的列表,依次对应
    [embeddings, layer_0, layer_1, ..., layer_{L-1}, pooler, cls]
每项为 GPU 序号(0-7)或 "cpu"。层与层之间的张量搬运即放置方案的通信代价。

测量协议:与 HDP/我们的 benchmark 完全一致 —— 跑 warmup+measure 步,
每步对所有涉及设备 synchronize 后计时,取中位数,输出一行 HDP_RESULT_JSON。

用法(galvatron 环境,bert/ 目录下):
  python placement_runner_bert.py --placement 0,0,0,0,1,1,1,1,...,7,7,7,7 --bsz 8
  python placement_runner_bert.py --placement_file my_placement.json   # {"placement":[...], "bsz":8}
  python placement_runner_bert.py --device_only 3                      # HDP "GPU Only" 基线(单卡全模型)
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForPreTraining

from dataloader import DataLoaderForBert


def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ToDevice(nn.Module):
    """设备放置的核心:输入张量搬到目标设备再转发(跨界点=通信代价)"""

    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.device = device

    def forward(self, *args, **kw):
        args = [a.to(self.device) if torch.is_tensor(a) else a for a in args]
        kw = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in kw.items()}
        return self.module(*args, **kw)


def parse_placement(placement_str, placement_file, num_layers, device_only):
    if device_only is not None:
        d = str(device_only)
        return [d] * (num_layers + 3)
    if placement_file:
        with open(placement_file) as f:
            spec = json.load(f)
        pl = spec["placement"] if isinstance(spec, dict) else spec
    else:
        pl = placement_str.split(",")
    pl = [str(x).strip() for x in pl]
    assert len(pl) == num_layers + 3, \
        "placement 长度应为 %d(embeddings+各层+pooler+cls), got %d" % (num_layers + 3, len(pl))
    for x in pl:
        assert x == "cpu" or (x.isdigit() and int(x) < torch.cuda.device_count()), "非法设备: %r" % x
    return pl


def to_dev(x):
    return "cpu" if x == "cpu" else torch.device("cuda:%s" % x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=5)
    parser.add_argument("--placement", type=str, default=None, help="逗号分隔的设备列表")
    parser.add_argument("--placement_file", type=str, default=None, help="JSON 放置文件")
    parser.add_argument("--device_only", type=int, default=None,
                        help="HDP 'GPU Only' 基线:整个模型放一张卡")
    args = parser.parse_args()

    set_seed()
    pl = parse_placement(args.placement, args.placement_file,
                         args.num_hidden_layers, args.device_only)
    devs = [to_dev(x) for x in pl]
    used_cuda = sorted({to_dev(x) for x in pl if x != "cpu"}, key=lambda d: d.index)
    if used_cuda:
        print("[placement] embeddings->%s, layer_k->%s...pooler/cls->%s, 使用 GPU {%s}" %
              (pl[0], "/".join(pl[1:1 + 3]) + "...", "/".join(pl[-2:]),
               ",".join(str(d.index) for d in used_cuda)), flush=True)

    # ---- 构建模型并按放置方案落位 ----
    # 词向量绑定:embeddings 与 cls 共享 word_embeddings 权重,只能落在同一张卡;
    # 若放置方案把二者放在不同设备,则自动解绑(decoder 多一份参数,约 +157MB,已注明)
    tied = (pl[0] == pl[-1])
    if not tied:
        print("[placement] embeddings(%s) 与 cls(%s) 不同卡 → 解绑词向量(tie_word_embeddings=False)"
              % (pl[0], pl[-1]), flush=True)
    config = BertConfig(vocab_size=args.vocab_size, hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers,
                        num_attention_heads=args.num_attention_heads,
                        intermediate_size=args.hidden_size * 4,
                        max_position_embeddings=args.seq_length,
                        tie_word_embeddings=tied)
    model = BertForPreTraining(config)
    model.bert.embeddings = model.bert.embeddings.to(devs[0])
    model.bert.encoder.layer = nn.ModuleList(
        [ToDevice(layer.to(devs[1 + i]), devs[1 + i])
         for i, layer in enumerate(model.bert.encoder.layer)])
    model.bert.pooler = model.bert.pooler.to(devs[-2])
    model.cls = model.cls.to(devs[-1])
    if tied:  # 绑定权重被 cls 的 .to 挪走后再归位(同卡,幂等)
        model.bert.embeddings = model.bert.embeddings.to(devs[0])
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    dataset = DataLoaderForBert()
    trainloader = DataLoader(dataset=dataset, batch_size=args.bsz, shuffle=True)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    in_dev = devs[0]   # 输入进入 embeddings 所在设备
    out_dev = devs[-1] # loss 在 cls 所在设备

    def sync_all():
        for d in used_cuda:
            torch.cuda.synchronize(d)

    step_times = []
    it, total = 0, args.warmup_steps + args.measure_steps
    for ep in range(100):
        for batch in trainloader:
            if it == args.warmup_steps:
                sync_all()
                for d in used_cuda:
                    torch.cuda.reset_peak_memory_stats(d)
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label = batch
            input_ids = input_ids.to(in_dev)
            attention_mask = attention_mask.to(in_dev)
            token_type_ids = token_type_ids.to(in_dev)
            masked_lm_labels = masked_lm_labels.to(out_dev)
            next_sentence_label = next_sentence_label.to(out_dev)

            sync_all()
            t0 = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            prediction_scores, seq_relationship_score = \
                outputs.prediction_logits, outputs.seq_relationship_logits
            # 与 train_hp_layerwise.py 相同的手工 loss(ignore_index=-1)
            mlm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), masked_lm_labels.view(-1))
            nsp_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            (mlm_loss + nsp_loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            sync_all()
            if it >= args.warmup_steps:
                step_times.append(time.time() - t0)
            it += 1
            if it >= total:
                break
        if it >= total:
            break

    step_times.sort()
    n = len(step_times)
    r_med = step_times[n // 2] if n % 2 == 1 else (step_times[n // 2 - 1] + step_times[n // 2]) / 2
    peak = max([torch.cuda.max_memory_allocated(d) for d in used_cuda], default=0) / 1024 / 1024
    result = {"rank": 0, "world_size": 1, "n_measured": n,
              "warmup_steps": args.warmup_steps, "measure_steps": args.measure_steps,
              "step_times_s": [round(t, 4) for t in step_times],
              "r_d_median_s": round(r_med, 4),
              "r_d_min_s": round(step_times[0], 4), "r_d_max_s": round(step_times[-1], 4),
              "reward_neg_sqrt": round(-r_med ** 0.5, 4),
              "peak_mem_allocated_MB": round(peak, 1),
              "placement": pl}
    print("HDP_RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
