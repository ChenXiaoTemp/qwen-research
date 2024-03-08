# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy
from math import ceil, floor

import numpy as np
import torch
from numpy import arange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation import GenerationConfig, SampleDecoderOnlyOutput
from transformers.trainer_utils import set_seed

DEFAULT_CKPT_PATH = 'Qwen/Qwen-1_8B'

_WELCOME_MSG = '''\
Welcome to use Qwen-Chat model, type text to start chat, type :h to show command help.
(欢迎使用 Qwen-Chat 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of Qwen.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
(注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
'''
_HELP_MSG = '''\
Commands:
    :help / :h          Show this help message              显示帮助信息
    :exit / :quit / :q  Exit the demo                       退出Demo
    :clear / :cl        Clear screen                        清屏
    :clear-his / :clh   Clear history                       清除对话历史
    :history / :his     Show history                        显示对话历史
    :seed               Show current random seed            显示当前随机种子
    :seed <N>           Set random seed to <N>              设置随机种子
    :conf               Show current generation config      显示生成配置
    :conf <key>=<value> Change generation config            修改生成配置
    :reset-conf         Reset generation config             重置生成配置
'''


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True, output_attentions=args.output_attentions,
        output_hidden_states=True, return_dict_in_generate=True
    )

    return model, tokenizer, config


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f'History ({len(history)})'.center(terminal_width, '='))
    for index, (query, response) in enumerate(history):
        print(f'User[{index}]: {query}')
        print(f'QWen[{index}]: {response}')
    print('=' * terminal_width)


def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')


def merge_outputs(outputs: SampleDecoderOnlyOutput):
    attentions = outputs.attentions
    words0_attentions = attentions[0][0].numpy()
    new_shape = list(words0_attentions.shape)
    new_shape[-2:] = [outputs.sequences.shape[-1], outputs.sequences.shape[-1]]
    res = []
    row = 0
    for word_idx, words_attentions in enumerate(attentions):
        for idx, layer in enumerate(words_attentions):
            layer = layer.numpy()
            if idx >= len(res):
                res.append(np.zeros(new_shape, dtype=layer.dtype))
            layer_shape = layer.shape
            res[idx][:, :, row:row + layer_shape[-2], :layer_shape[-1]] = layer
        row += layer_shape[-2]

    return res


def draw_attention(labels, attentions, layer, threshold, head=0):
    import networkx as nx
    G = nx.Graph()
    for idx, label in enumerate(labels):
        if idx == 0:
            continue
        G.add_node(label)
    shape = attentions[0].shape
    for x in range(0, shape[2]):
        for y in range(0, shape[3]):
            attention = attentions[layer][0][head][x, y]
            if attention > threshold:
                G.add_edge(labels[x + 1], labels[y + 1], weight=1 - attention)
    nx.draw_spring(G, font_family='SimHei', with_labels=True)


def visualize_outputs(outputs: SampleDecoderOnlyOutput, tokenizer: PreTrainedTokenizer):
    merged_attentions = merge_outputs(outputs)
    attentions = merged_attentions
    sequences = outputs.sequences
    labels = []
    for id in sequences[-1].tolist():
        labels.append(tokenizer.decode([id]))
    import matplotlib.pyplot as plt

    for i in range(len(attentions)):
        row, col = floor(i / 4), i % 4
        plt.subplot(row + 1, col + 1, 1)
        for threshold in arange(0.0001, 0.2, 0.01):
            draw_attention(labels, attentions, i, threshold)
            plt.savefig(f"{i}-{threshold:.2f}.png")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='QWen-Chat command-line interactive chat demo.')
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument('--output_attentions', type=bool, default=True, help="output_attentions")
    args = parser.parse_args()

    history, response = [], ''

    model, tokenizer, config = _load_model_tokenizer(args)

    orig_gen_config = deepcopy(model.generation_config)

    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed

    while True:
        query = "你是谁"

        # Process commands.
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command in ['clear', 'cl']:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif command in ['clear-history', 'clh']:
                print(f'[INFO] All {len(history)} history cleared')
                history.clear()
                _gc()
                continue
            elif command in ['help', 'h']:
                print(_HELP_MSG)
                continue
            elif command in ['history', 'his']:
                _print_history(history)
                continue
            elif command in ['seed']:
                if len(command_words) == 1:
                    print(f'[INFO] Current random seed: {seed}')
                    continue
                else:
                    new_seed_s = command_words[1]
                    try:
                        new_seed = int(new_seed_s)
                    except ValueError:
                        print(f'[WARNING] Fail to change random seed: {new_seed_s!r} is not a valid number')
                    else:
                        print(f'[INFO] Random seed changed to {new_seed}')
                        seed = new_seed
                    continue
            elif command in ['conf']:
                if len(command_words) == 1:
                    print(model.generation_config)
                else:
                    for key_value_pairs_str in command_words[1:]:
                        eq_idx = key_value_pairs_str.find('=')
                        if eq_idx == -1:
                            print('[WARNING] format: <key>=<value>')
                            continue
                        conf_key, conf_value_str = key_value_pairs_str[:eq_idx], key_value_pairs_str[eq_idx + 1:]
                        try:
                            conf_value = eval(conf_value_str)
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            print(f'[INFO] Change config: model.generation_config.{conf_key} = {conf_value}')
                            setattr(model.generation_config, conf_key, conf_value)
                continue
            elif command in ['reset-conf']:
                print('[INFO] Reset generation config')
                model.generation_config = deepcopy(orig_gen_config)
                print(model.generation_config)
                continue
            else:
                # As normal query.
                pass

        # Run chat.
        set_seed(seed)
        try:
            response, history, outputs = model.chat(tokenizer, query, history=history, generation_config=config)
            print(f"\nUser: {query}")
            print(f"\nQwen-Chat: {response}")
            visualize_outputs(outputs, tokenizer)
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue
        history.append((query, response))
        break


if __name__ == "__main__":
    main()
