"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional
import psutil

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import,
    cache_replace, acc_replace)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, print_cpu_mem_usage,
    write_benchmark_log, read_benchmark_log)

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float
    # 是否将 I/O 和计算进行重叠，即IO和计算是否并行
    # Whether to overlap the I/O and compute
    overlap: bool
    # 是否将注意力机制和多层感知机（MLP）视为两个独立的层
    # Whether to separate attention and mlp as two layers
    sep_layer: bool
    # 是否在 CPU 上使用“pinned memory”来存储权
    # Whether to use pinned memory for weights on CPU
    pin_weight: bool
    # 是否在 CPU 上进行注意力机制的计算
    # Whether to compute attention on CPU
    cpu_cache_compute: bool
    # 注意力机制的稀疏性参数，控制注意力矩阵的稀疏性
    # Sparsity of attention weights
    attn_sparsity: float
    # 是否使用分组量化对 权重 进行压缩
    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig
    # 否使用分组量化对 缓存 进行压缩
    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig
    # hh_ratio：“heavy hitter”比率；hh_all：是否对所有数据应用 heavy hitter 策略；
    # hh_long_seq：用于控制长序列数据的处理策略，优化长序列输入的模型推理过程
    # heavy hitter pruning
    hh_ratio: float = 1
    hh_all: bool = False
    hh_long_seq: bool = False
    # 权重、缓存和激活在disk上占比
    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val # 分为 词汇嵌入权重，位置嵌入权重
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst))) # 将输入的嵌入权重存储到buf中

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        # test
        print("******InputEmbed is load_cache used?******")
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate, hh_long_seq=self.policy.hh_long_seq)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val # w_ln：层归一化（Layer Normalization）的权重，b_ln：层归一化的偏置项（bias），w_token：词汇嵌入权重
        if k == 0: # 初始化存储位置后 存储
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        # test
        print("******OutputEmbed is load_cache used?******")
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        # print(h.data[0, 0, :5])
        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        # print(h.data)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None

    def set_task(self, task): # hh_k是 输入长度*重击率
        self.task = task
        self.hh_k = int(task.prompt_len * self.policy.hh_ratio)

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    # 自注意力机制中加载权重
    def load_weight(self, weight_home, weight_read_buf, k): 
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val # q,k,v,output,层归一化的权重和偏置
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
            
    # 配置cache的分配，全gpu、全cpu、全disk还是混合分配，device类型
    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed
        # 非混合分配时才可以压缩cache
        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        # 初始化cache并存储在cache_home，根据设备类型确定 初始化cache的调用方法
        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy, self.hh_k, self.policy.hh_all) # 在模型中已经设置好对应层里的hh_k
        cache_home.store(cache)
    # 根据不同策略选择合适的路径，将缓存从存储设备（如 GPU、CPU 或混合设备）加载到计算设备（如注意力计算设备），以优化注意力机制的计
    def load_cache(self, cache_home, cache_read_buf, i):
        # test
        print("******Att is load_cache used?******")
        if i == 0:  # prefill, no cache
            return
        # test
        print("******att load_cache: the value of cache home val:",cache_home.val)
        print("******att load_cache: the type of cache home val:",type(cache_home.val))

        k_home, v_home, acc = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        pos = min(self.hh_k * 2 - 1, self.task.prompt_len) + 1
        # 根据不同策略加载cache
        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            if self.policy.hh_all:
                indices = (slice(0, pos),
                           slice(0, k_home.shape[1]))
            else:
                indices = (slice(0, pos + i),
                           slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                    acc.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf, acc_buf = dst.next_attention_compute_workspace()
            if self.policy.hh_all:
                indices = (slice(0, pos),
                           slice(0, k_home.shape[1]))
            else:
                indices = (slice(0, pos + i),
                           slice(0, k_home.shape[1]))

            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                general_copy(acc_buf, indices, acc, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False), (acc_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, pos + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")
    # 模型生成过程中产生的cache（如注意力计算的键和值）存储起来
    def store_cache(self, cache_home, cache_write_buf, i):
        # # shape: (s, b * n_head, head_dim)
        
        # test
        print("******att store_cache: the value of cache home val:",cache_home.val)
        print("******att store_cache: the type of cache home val:",type(cache_home.val))
        
        k_home, v_home, acc = cache_home.val
        k_new, v_new, acc_new, kick_ind = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return
        
        # 将生成过程中的注意力cache（如键、值）存储下来，并根据策略选择适当的方法来更新或替换cache
        if i == 0:  # prefill 
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1])) # 根据生成的新数据的形状，初始化cache的索引

        else:  # decoding  # 根据策略 self.policy.hh_all 和 self.hh_k决定更新cache方法
            if self.policy.hh_all: 
                oldest = ((i - 1) % (self.hh_k - 1)) - (self.hh_k - 1)
                cache_replace(k_home, kick_ind, k_new, self.hh_k, oldest)
                cache_replace(v_home, kick_ind, v_new, self.hh_k, oldest)
                acc_replace(acc, kick_ind, acc_new, self.hh_k, oldest)
                return # 用循环缓冲区的方式，替换掉cache中最旧的数据

            if self.hh_k is None:
                pos = self.task.prompt_len + i # 计算出cache索引位置
            else:
                pos = min(self.hh_k * 2 - 1, self.task.prompt_len) + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)
        if self.policy.hh_all:
            general_copy(acc, indices, acc_new, None) # 根据策略替换cache中最旧的数据或者直接更新对应位置的cache

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head

        donate = [False] * 14 # 跟踪可以在计算后被释放的张量，14？？
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop() # 最后一个batch，获取并清除权重
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val # 获取权重当前值

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            # print("hidden", h.data)
            h, new_k_cache, new_v_cache, acc = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config,
                self.hh_k, self.policy.hh_all)
            # print(new_k_cache.shape)
            # print(new_k_cache.data[:, :2, :2])
            cache_write_buf.store((new_k_cache, new_v_cache, acc, None))
        else:  # decoding
            # print("hidden", h.shape)
            # print(h.data)
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]), (acc, _) = cache_read_buf.pop()
            # if self.layer_id == 10:
            #     print(k_cache.shape)
            #     cnt = min(self.hh_k * 2, self.task.prompt_len)
            #     print(v_cache.data[:cnt + i][-5:, 0, :2])
            if self.policy.hh_all is not None:
                cnt = min(self.hh_k * 2 - 1, self.task.prompt_len + i)
                mask = mask.device.slice_attention_mask(mask, cnt + 1)
            h, new_k_cache, new_v_cache, acc, kick_ind = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, acc, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,
                self.hh_k, self.policy.hh_all)
            # if self.layer_id == 10:
            #     print(h.data)
            cache_write_buf.store((new_k_cache, new_v_cache, acc, kick_ind))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val # input,output,归一化层 的权重和偏置
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home): # MLP层不用加载cache
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        # test
        print("******MLP is load_cache used?******")
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val # 两部分拆分？
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0: # 在非首批次的情况下，权重已经加载并存储，后续只需要利用缓冲区中的权重，而不需要每次都重新加载
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home) # 调用attention层的cache初始化方法

    def load_cache(self, cache_home, cache_read_buf, i):
        # test
        print("******Transformer is load_cache used?******")
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k): # cache、weight读缓存，cache写缓存
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop() # 清除weight读缓存
        else:
            read_buf1, read_buf2 = weight_read_buf.val # 获取 att和MLP的weight读缓存值

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k) # att计算
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k) # MLP计算，cache无关


class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        # 初始化模型的各个层，输入嵌入层、隐藏层、输出嵌入层（各层分别进行配置），存储所有层、记录层数目
        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)
        # 设置 激活状态 存储的位置
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()
        # 创建用于异步执行的 CUDA 流，用于加载权重、加载缓存和存储缓存，可以提高并行处理的效率
        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        # cache_home, cache_read_buf, cache_write_buf分别存储 每层 每个GPU批次 的 缓存数据 以及读、写缓冲区
        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder) # 存储每层的 权重读取 缓冲区
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder) # 存储注意力掩码

        self.task = None
        self.init_all_weights() # 初始化所有的权重

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
        self.hh_k = int(task.prompt_len * self.policy.hh_ratio)

    # 根据层索引初始化层的权重
    def init_weight(self, j):
        # 把存储权重路径self.path和*-np拼接，然后再替换可能的~，最后取绝对路径，得到存储权重的路径expanded_path
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        # 拼接expanded_path和"decoder.embed_positions.weight"
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        # 若 check_path不存在 且 路径中不包含某个占位符DUMMY_WEIGHT ，则根据 模型名 和 设定的权重路径 下载模型权重
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)
        # 调用每一层的（不同类型的层）init_weight()方法，参数为对应第j层的权重和expanded_path
        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return
        # 不同类型的层加载权重，IO和CPU并行或串行
        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream): # 允许异步加载权重，提高性能
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k) # 同步加载权重，可能适用于调试或不需要异步计算的情况

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    # 初始化cache，即考虑将cache如何分配、存储，只需attention层实现该方法
    def init_cache(self, j, k):
        # # # test
        # print("******before init_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
        # print("******before init_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
        # print("******before init_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])
        # # # test
        # print("******after init_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
        # print("******after init_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
        # print("******after init_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))

    def load_cache(self, i, j, k, overlap=True):
        # test
        print("******test: load_cache-1******")
        
        # Handle corner cases
        if i == 0:  # prefill, no cache
            # test
            print("******test: load_cache-2******")
            return
        if k == self.num_gpu_batches:
            # test
            print("******test: load_cache-3******")
            k = 0
            j += 1
        if j == self.num_layers:
            # test
            print("******test: load_cache-4******")
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return
            
        # # # test
        # print("******before load_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
        # print("******before load_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
        # print("******before load_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))

        # 不同类型的层加载cache，IO和CPU并行或串行
        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
            # # test
            # print("******test1: i ,j ,k  ******",i, j, k)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
            # # test
            # print("******test2: i ,j ,k  ******",i, j, k)

        # # # test
        # print("******after load_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
        # print("******after load_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
        # print("******after load_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))
    
    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # # test
        # print("******the value of self cache home: ", self.cache_home)
        # print("******the type of self cache home: ", type(self.cache_home))  # list      
        # print("******the value of self cache write buf: ", self.cache_write_buf)
        # print("******the type of self cache write buf: ", type(self.cache_write_buf))  # list     
        # print("******the value of input i: ", i)
        # print("******the type of input i: ", type(i))  # int

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:

            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            # # test
            print("******before store_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
            print("******before store_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
            # print("******before store_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))

            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
            
            # # test
            print("******after store_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
            print("******after store_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
            # print("******after store_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()
    # 将隐藏状态加载到适当的位置
    def load_hidden(self, i, j, k):
        # # test
        # print("******test: load_hidden******")
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute # 目标设备 dst 设置为第 j 层的计算设备
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids # 从输入加载
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token # 从上次生成的token加载
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer # 从上一层加载
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output  # 最后一层
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop: # 任务有停止标识
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids # 更新输出序列id
        else:  # move to home  # 中间层
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home) # 移动隐藏状态到存储设备
    # 更新隐藏层、清除内存缓冲、运行层计算
    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()
        
    # 初始化所有权重
    def init_all_weights(self):
        # 大小与模型的层数相同，元素类型为ValueHolder，用于保存对应层权重
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            # 对于每一层j，初始化该层权重(模型层面)
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            # cnt = min(self.hh_k * 2, self.task.prompt_len)
            # if i == 1:
            #     mask.val = mask.val.device.slice_attention_mask(mask.val, cnt)
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            #if self.policy.hh_all:
            #    mask.val = mask.val.device.slice_attention_mask(mask.val, cnt + 1)
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0): 
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len # 确定生成语句的长度即推理迭代次数

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)  # np.full()生成一个具有指定形状元素值全1的数组
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool) # 标记每个batch是否停止
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs) # 给output_ids赋值
        assert gpu_batch_size * num_gpu_batches == len(task.inputs) # 断言确保批次大小符合预期
        
        # # test
        # print("*******the value of self.config.pad_token_id:", self.config.pad_token_id)
        # print("*******the value of init output_ids:", self.output_ids)
        # print("*******the length of init output_ids:", len(self.output_ids))
        # print("*******the length of init output_ids[0]:", len(self.output_ids[0]))
        # print("*******the value of init task.inputs:", task.inputs)
        # print("*******the length of init task.inputs:", len(task.inputs))
        # print("*******the length of init task.inputs[0]:", len(task.inputs[0]))
        
        # 中间张量，存储第i个token、第j层、第k个批次的值
        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches): # 每个批次的attention不同
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder) # 存放隐藏状态
        # 初始化cache
        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k) # 初始化即分配、存储cache
        if self.policy.cpu_cache_compute: # 在CPU计算attention
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy, self.hh_k)
        print_cpu_mem_usage("after init cache")

        # Generate
        if debug_mode is None: # 无调试模式
            if not overlap: # IO和CPU串行
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute # IO和CPU并行 
                if num_gpu_batches == 1: # 1个batch
                    self.generation_loop_overlap_single_batch()
                else: # 多个batch
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch": # 更少层数和批次用于调试
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown": # 无 I/O 并行，并提供分批次的执行时间分析
            # No overlap, fewer batches, execution time breakdown 
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")
        
        # 推理之后的后处理，释放cache，若是cpu推理删除注意力计算工作空间，对应Init cache
        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids
    # 无调试模式，串行
    def generation_loop_normal(self):
        for i in range(self.execute_gen_len): # 逐次推理
            timers("generate").start() # 开始计时
            for k in range(self.num_gpu_batches): # 按batch数更新注意力掩码（用于处理序列生成时，哪些部分应该被关注）
                self.update_attention_mask(i, k)
            for j in range(self.num_layers): 

                # print("step", i, "layer", j-1)
                for k in range(self.num_gpu_batches): # 加载j层、k-batch的权重
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches): # 加载 j层、k-batch的cache、隐藏层， 计算 j层、k-batch， 存储 j层、k-batch的隐藏层、cache
                    # test
                    print("******load_cache_start")
                    print("******before load_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
                    print("******before load_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
                    # print("******before load_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))
                    
                    self.load_cache(i, j, k, overlap=False)
                    
                    # # test
                    print("******after load_cache: the value of self cache_home[",j,"][",k,"]: ", self.cache_home[j][k])
                    print("******after load_cache: the value of self cache_home[",j,"][",k,"].val: ", self.cache_home[j][k].val)
                    # print("******after load_cache: the type of self cache_home[",j,"][",k,"]: ", type(self.cache_home[j][k]))
                    print("******load_cache_stop")
                    # # test
                    # print("******load_hidden_start")
                    # print("******before load_hidden: the value of self hidden[",j,"][",k,"]: ", self.hidden[j][k])
                    self.load_hidden(i, j, k)
                    # # test
                    # print("******after load_hidden: the value of self hidden[",j,"][",k,"]: ", self.hidden[j][k])
                    # print("******load_hidden_stop")
                    self.compute_layer(i, j, k) # 计算j层、k-batch
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
                    # # test
                    # print("******i, j, k******: ", i, j, k)
            timers("generate").stop() # 停止计时

    # 无 I/O 并行，并提供分批次的执行时间分析
    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()
        # 设置执行批次数和初始化计时器，准备跟踪生成过程中的各个环节
        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")
        # 逐次推理
        for i in range(self.execute_gen_len):
            if i == 0: # 第一步迭代
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else: # 第二次及以后迭代
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k) # 更新k-batch掩码
            # 逐层计算
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync) # 加载权重 开始
                for k in range(self.num_gpu_batches): # 逐 层和batch 加载权重
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync) # 加载权重 结束

                for k in range(self.num_gpu_batches): # 逐 层和batch 加载cache、隐藏层，计算，存储隐藏层、cache
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1) # 一层所有批次算完为什么batch更新？
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}") # 打印预填充批次数
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}") # 打印解码批次数
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s") # 打印每层加载权重的平均时间
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s") # 打印每个函数的平均执行时间

    # 无调试模式，单batch 并行
    def generation_loop_overlap_single_batch(self): # 重叠 I/O 和计算来提高效率
        # Prologue
        for k in range(self.num_gpu_batches): 
            self.load_weight(0, 0, k) # 加载第0次、0层、所有批次的权重
        self.sync() # 确保所有 GPU 在继续之前都已完成权重加载

        # Generate
        for i in range(self.execute_gen_len): # 因为只有1个batch
            timers("generate").start()
            self.update_attention_mask(i, 0) # 更新i次推理，0-batch掩码？？
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0) # 连续加载权重、cache、隐藏层？计算，存储cache、隐藏层，同步
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped): # 停止条件检查
                break
    
    # 无调试模式，多batch并行
    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k) # 加载第0次、0层、所有batch的权重
        self.load_hidden(0, 0, 0) # 加载初始的隐藏状态?
        self.sync() # 同步所有 GPU

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k) # 更新i次推理，k-batch掩码（多batch）
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k) # 加载第 j+1 层的权重
                    self.load_cache(i, j, k+1) # 加载缓存
                    self.store_hidden(i, j, k-1) # 存储前一个batch的隐藏状态
                    self.load_hidden(i, j, k+1) # 加载下一batch的隐藏状态
                    self.compute_layer(i, j, k) # 执行当前层的计算
                    self.store_cache(i, j, k-1)  # 存储当前层、前一个batch的缓存
                    self.sync() # 同步所有 GPU
            timers("generate").stop() 
        # 在所有生成步骤完成后，存储最后一步的隐藏状态，以便后续处理或输出
        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    # 调试模式，更少层数和批次，单batch
    def  generation_loop_debug_single_batch(self):
        execute_num_batches = 10
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()
        # 算CPU可用内存
        lowest_avail_mem = float('inf')
        cpu_avail_mem = []
        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
                if i > 0 or j:
                    avail = psutil.virtual_memory().available / GB
                    lowest_avail_mem = min(lowest_avail_mem, avail)
                    cpu_avail_mem.append((i, j, avail))

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # print("(token, layer, available mem)")
        # for mem_info in cpu_avail_mem:
        #     print(f"({mem_info[0]}, {mem_info[1]}, {mem_info[2]:.2f})"),
        print(f"lowest available cpu mem: {lowest_avail_mem:.2f}")

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[execute_num_batches // 2:])
        for i in range(self.execute_gen_len): # genarate时间就是prefill+decoding时间
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    # 调试模式，更少层数和批次，多batch
    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k) # 加载第0次、0层、所有batch的权重
        self.load_hidden(0, 0, 0) # 加载初始的隐藏状态?
        self.sync()
        # 算CPU可用内存
        lowest_avail_mem = float('inf')
        cpu_avail_mem = []
        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start() # prefill开始时间
            for k in range(self.num_gpu_batches): # 更新i次推理，k-batch掩码（多batch）
                self.update_attention_mask(i, k) 
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start() # decoding开始时间
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
                    if i > 0 or (j + k == 0): # 从 第二次迭代 或 第一层第一批 开始监控内存
                        avail = psutil.virtual_memory().available / GB
                        lowest_avail_mem = min(lowest_avail_mem, avail)
                        cpu_avail_mem.append((i, j, k, avail))

                if i > 0:
                    timers("decoding_gpu_batch").stop() # 每层decoding结束记录本batch结束时间？？
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # print("(token, layer, batch, available mem)")
        # for mem_info in cpu_avail_mem:
        #     print(f"({mem_info[0]}, {mem_info[1]}, {mem_info[2]}, {mem_info[3]:.2f})"),
        print(f"lowest available cpu mem: {lowest_avail_mem:.2f}")

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()

# 组合log文件名字
def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    # prompts = ["Paris is the capital city of"]
    prompts = ["Artificial Intelligence (AI) refers to the development of computer systems or machines that can perform tasks that typically require human intelligence. These tasks include problem-solving, learning, understanding natural language, recognizing patterns, perception, and decision-making. AI systems are designed to process vast amounts of data and draw conclusions or make decisions based on that data. There are two main categories of AI: Narrow AI and General AI. Narrow AI, also known as Weak AI, is designed for "]
    prompts = ["As I sit here on my porch, sipping my coffee and watching the world go by, I can not help but feel a sense of wonder at the sheer complexity of everything around us. From the smallest particle to the grandest galaxy, the universe is a tapestry of infinite detail and beauty. And yet, for all its complexity, there is a simplicity to it all that is truly awe-inspiring. Everything is connected, in ways that we can not even begin to fathom. Every action has a reaction, every cause has an effect. And yet, even with all the knowledge that we have amassed, there is still so much that we do not understand. There are mysteries that have eluded us for centuries, and may continue to do so for centuries to come. But that does not stop us from trying to unravel them. It does not stop us from exploring the depths of our own consciousness, or the vast expanse of the cosmos. It does not stop us from seeking answers to the biggest questions of all. Who are we? Why are we here? What is the meaning of life? These are questions that have plagued us since the dawn of time, and yet we continue to search for answers. Perhaps it is in the search itself that we find meaning. Perhaps it is in the journey, rather than the destination, that we discover the true nature of our existence. And so, as I sit here on my porch, watching the world go by, I am content to simply marvel at the beauty and complexity of it all, and to embrace the mystery that lies at the heart of our being."]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len, add_special_tokens=False).input_ids
    # input_ids = tokenizer(prompts, add_special_tokens=False).input_ids
    return (input_ids[0],) * num_prompts # 重复多次


def run_flexgen(args):
    # # test
    # print("******args: ", args)

    # 解析 模型名 获得对应模型的tokenizer
    # if args.model == "facebook/galactica-30b":
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left", use_fast=True)

    # 修改成本地存在模型时的情况, 原来的情况只需要考虑opt-30b, opt系列和gpt系列, tokenizer可通用
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    elif args.model == "facebook/opt-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    else:
        # tokenizer = AutoTokenizer.from_pretrained("/opt/lw/Models/opt-1.3b", padding_side="left")
        # # test
        # print("the value of args model: ", args.model)
        # tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        tokenizer = AutoTokenizer.from_pretrained("/opt/lw/Models/opt-1.3b", padding_side="left")
        # # test
        # print("the value of tokenizer: ", tokenizer)

    num_prompts = args.num_gpu_batches * args.gpu_batch_size # 1*1
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len # 512,32,none
    
    # 将prompts进行tokenize
    # Task and policy
    warmup_inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    
    # test
    # print("******the value of prompt_len and num_prompts: ", prompt_len, num_prompts) # 512 1
    # print("******the value of inputs", inputs) # 1, 1个list元素
    # print("******the value of inputs[0]", inputs[0]) # inputs第一个list元素
    # print("******the length of inputs", len(inputs))

    prompt_len = len(inputs[0]) 
    print(prompt_len) # 512
    
    # 选计算设备，GPU，CPU和Disk，配置硬件运行环境
    gpu = TorchDevice("cuda:0")
    # gpu = TorchDevice("cuda:1")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    # 制定优化策略，权重分配、稀疏、hh率等配置
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False),
                    hh_ratio=args.hh_ratio,
                    hh_all=args.hh_all,
                    hh_long_seq=args.hh_long_seq)
    # 缓存压缩和稀疏注意力机制不能同时使用, 还未实现
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"
    # 根据模型名称生成模型的超参数配置
    opt_config = get_opt_config(args.model)
    # 公式估算 模型、缓存、隐藏层大小并打印
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")
    # 初始化权重
    print("init weight...")
    model = OptLM(opt_config, env, args.path, policy)
    print_cpu_mem_usage("after init weight") # 打印CPU内存使用情况

    try:
        # 预热生成
        print("warmup - generate")
        # output_ids = model.generate(
        #     warmup_inputs, max_new_tokens=4, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        # ****推理生成****
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()
    # 计算延迟、吞吐量
    # Log output
    prefill_latency = costs[0] # prefill cost
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len) # 预测（求均值再乘解码生成token个数）
    else:
        decode_latency = sum(costs[1:]) # 直接求和
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10) # max防止除以零的错误
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats() # 打印并返回 当前和峰值GPU内存占用
    cpu.print_stats() # 打印并返回 当前和峰值CPU内存占用
    projected = bool(args.debug_mode or cut_gen_len)
    
    # 获取日志文件名字
    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file
    # 写日志
    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")

    # heavy hitter pruning
    parser.add_argument("--hh-ratio", type=float, default=1,
                        help="ratio of the prompt seq length")
    parser.add_argument("--hh-all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--hh-long-seq", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6
    assert int(args.prompt_len * args.hh_ratio) > 0, "Please increase the ratio to keep at least one token"

    run_flexgen(args)
