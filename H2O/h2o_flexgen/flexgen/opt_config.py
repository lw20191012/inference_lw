"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm

# 用于配置神经网络模型的 超参数 ，特别是与模型存储和缓存有关的计算
@dataclasses.dataclass(frozen=True)
class OptConfig:
    name: str = "opt-125m"
    num_hidden_layers: int = 12
    max_seq_len: int = 2048
    hidden_size: int = 768
    n_head: int = 12
    input_dim: int = 768
    ffn_embed_dim: int = 3072 # 嵌入维度
    pad: int = 1 
    activation_fn: str = 'relu'
    vocab_size: int = 50272 # 词汇表的大小
    layer_norm_eps: float = 0.00001 # 层归一化的 epsilon 值，防止数值不稳定
    pad_token_id: int = 1 # 用于填充的 token ID
    dtype: type = np.float16 # 数据类型

    # 该方法 估算 模型在内存中 占用的字节数
    def model_bytes(self):
        h = self.input_dim
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))
    
    # 计算缓存占用的字节数，尤其是自注意力机制中的键和值缓存
    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2
    
    # 计算隐藏状态所需的内存大小
    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_opt_config(name, **kwargs):
    # if "/" in name:
    #     name = name.split("/")[1]
    # name = name.lower()

    # 修改的内容__begin
    # test
    print("******model name before get_opt_config:", name)
    # 原来是默认namespace/model_name从官网下载，拆分路径得到model_name, 
    # 考虑本地路径存在多个/分隔的元素，最后一个元素是model_name
    if "/" in name:
        # name = name.split("/")[1]
        name = name.split("/")[-1]
    name = name.lower()
    # test,已经是模型的名称了
    print("******model name after get_opt_config:", name)
    # 修改的内容__end


    # Handle opt-iml-30b and opt-iml-max-30b
    if "-iml-max" in name:
        arch_name = name.replace("-iml-max", "")
    elif "-iml" in name:
        arch_name = name.replace("-iml", "")
    else:
        arch_name = name

    if arch_name == "opt-125m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=12, n_head=12,
            hidden_size=768, input_dim=768, ffn_embed_dim=768 * 4,
        )
    elif arch_name == "opt-350m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=16,
            hidden_size=1024, input_dim=1024, ffn_embed_dim=1024 * 4,
        )
        raise NotImplementedError("Not implemented because this model "
                                  "has a different architecture")
    elif arch_name == "opt-1.3b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=32,
            hidden_size=2048, input_dim=2048, ffn_embed_dim=2048 * 4,
        )
    elif arch_name == "opt-2.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=2560, input_dim=2560, ffn_embed_dim=2560 * 4,
        )
    elif arch_name == "opt-6.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=4096, input_dim=4096, ffn_embed_dim=4096 * 4,
        )
    elif arch_name == "opt-13b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=40, n_head=40,
            hidden_size=5120, input_dim=5120, ffn_embed_dim=5120 * 4,
        )
    elif arch_name == "opt-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4,
        )
    elif arch_name == "galactica-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4, vocab_size=50000,
        )
    elif arch_name == "opt-66b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=64, n_head=72,
            hidden_size=9216, input_dim=9216, ffn_embed_dim=9216 * 4,
        )
    elif arch_name == "opt-175b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=96, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    elif arch_name == "opt-175b-stage":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_opt_weights_old(model_name, path):
    """Download weights from huggingface."""
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "galactica" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)

# 下载opt模型权重
def download_opt_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    # if "opt" in model_name:
    #     hf_model_name = "facebook/" + model_name
    # elif "galactica" in model_name:
    #     hf_model_name = "facebook/" + model_name

    # folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    

    # 添加的内容__begin
    # test
    print("********the model name when download the weight of model: ", model_name)
    
    # add: 本地存放模型的文件夹中是否存在该模型
    model_dir = "/opt/lw/Models"
    model_path = os.path.join(model_dir, model_name) 
    # add: 原来是从facebook上下载
    # 现在先判断模型是否存在于本地，若是则跳过Hugging Face下载
    if os.path.exists(model_path):
        print(f"Using local model path:{model_path}")
        # return
        hf_model_name = model_path # 本地路径
        folder = hf_model_name # 返回包含.bin文件的文件夹
    else:
        print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
            f"The downloading and cpu loading can take dozens of minutes. "
            f"If it seems to get stuck, you can monitor the progress by "
            f"checking the memory usage of this process.")
        # 在模型名称面前加上facebook
        if "opt" in model_name:
            hf_model_name = "facebook/" + model_name
        elif "galactica" in model_name:
            hf_model_name = "facebook/" + model_name
        
        # test
        print("******huggingface model name:", hf_model_name)
        folder = snapshot_download(hf_model_name, allow_patterns="*.bin") # 从Hugging Face 只下载.bin格式模型文件 到 文件夹folder
    # 添加的内容__end

    bin_files = glob.glob(os.path.join(folder, "*.bin")) # 返回 folder 文件夹中所有 .bin 文件的路径
    # # test 不用该方法下载无法测试？？？
    # print("******the type of bin_files:", type(bin_files))
    # print("******the value of bin_files:", bin_files)
    
    # 新建存储权重目录
    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        # 构造存储路径 param_path，并使用 numpy.save 将每个权重张量保存为 .npy 文件
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
            # decoder.embed_tokens.weight是模型嵌入层权重，也用于语言模型的头部lm_head.weight，故将该文件复制，重命名为 lm_head.weight
            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    args = parser.parse_args()

    download_opt_weights(args.model, args.path)
