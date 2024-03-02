import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import numpy as np
import random


def construction_rps(input_file, rp_file):

    f_rp = open(rp_file, 'r')
    promote = f_rp.readlines()
    promote = random.choice(promote)
    instruction = "Instruction:"
    response = "Response:"
    ress = '\nThe answer is: \n'
    f = open(input_file, 'r')
    f = json.load(f)
    rating_prompt_list = []
    for item in f:
        ins = item['conversations'][0]['value']
        res = item['conversations'][1]['value']
        rating_prompt = promote + '\n' + instruction + ins + '\n' + response + res + ress
        rating_prompt_list.append(rating_prompt)
    return rating_prompt_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, required=True, help='model name in the hub or local path')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='input file')
    parser.add_argument('--rating-prompt-file', '-rp', type=str, required=True, help='input file')
    parser.add_argument('--output-file', '-o', type=str, required=True, help='output file')
    parser.add_argument('--k', '-k', type=int, required=True, help='parameter')
    parser.add_argument('--proportion', '-proportion', type=float, required=True, default=0.2, help='parameter')
    parser.add_argument('--alpha', '-alpha', type=float, required=True, default=0.2, help='parameter')
    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    input_file = args.input_file
    rp_file = args.rating_prompt_file
    output_file = args.output_file
    proportion = args.proportion
    alpha = args.alpha
    k = args.k
    rps = construction_rps(input_file, rp_file)
    print('okk')
    print(f'Loading Mater Model weights from path: {model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    print(model.hf_device_map)

    to_use_fast = False
    if "bloom" in model_name_or_path:
        to_use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pro = []
    for idx, p in enumerate(rps):
        tokenized = tokenizer(p, padding=True, return_tensors="pt")
        tokenized.input_ids = tokenized.input_ids.cuda()
        tokenized.attention_mask = tokenized.attention_mask.cuda()
        with torch.no_grad():
            try:
                outputs = model(**tokenized)
                predictions = outputs[0]
                logits = predictions[:, -1, :]
                softmax_logits = torch.softmax(logits.float(), dim=-1)
                for index in range(1):
                    tmp_res = [float(softmax_logits[index][29896]), float(softmax_logits[index][29906]),
                               float(softmax_logits[index][29941]), float(softmax_logits[index][29946]),
                               float(softmax_logits[index][29945])]
                    pro.append(tmp_res)
            except Exception as ex:
                print(ex)
    pro_softmax = []
    for item in pro:
        tmp_pro_softmax = item
        tmp0_pro_softmax = []
        tmp1_pro_softmax = []
        for idx, item in enumerate(tmp_pro_softmax):
            tmp0_pro_softmax.append(np.exp(tmp_pro_softmax[idx] / sum(tmp_pro_softmax)))
        for jdx, item in enumerate(tmp0_pro_softmax):
            tmp1_pro_softmax.append(tmp0_pro_softmax[jdx] / sum(tmp0_pro_softmax))
        pro_softmax.append(tmp1_pro_softmax)

    data_num = len(pro_softmax)
    token_level_score = []
    for idx in range(data_num):
        score_base = np.argmax(pro_softmax[idx])
        tmp_sum = 0
        for tmp_idx in range(len(pro_softmax[idx])):
            tmp_sum += pro_softmax[idx][score_base] - pro_softmax[idx][tmp_idx]
        tmp_sum = tmp_sum / (k - 1)
        token_score = (score_base + 1) * tmp_sum
        token_level_score.append(token_score)

    score_dict = {}
    for idx, score in enumerate(token_level_score):
        score_dict[idx] = score
    score_dict_sort = sorted(score_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    final_it_data = []
    f = open(input_file, 'r')
    f = json.load(f)
    for item in score_dict_sort:
        id = item[0]
        final_it_data.append(f[id])
    final_it_data = final_it_data[0:int(len(final_it_data) * proportion)]
    f_o = open(output_file, 'w')
    json.dump(final_it_data, f_o)
    f_o.close()

