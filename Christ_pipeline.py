import argparse

from typing import Dict, Tuple, List, Any
import random
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
import os
import sys
import time
import json
import time 
import math
import statistics
from scipy import special
from scipy.spatial import distance
import numpy as np

import torch
from peft import PeftModel    
from compact_text import CompactText
from DISC import DISC, ChristWatermark, ChristWatermarkMultiKey
def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str, default="llama-7b")

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="../data/alpaca_data.json")
    parser.add_argument('--prompt_type', type=str, default="alpaca", 
                        help='type of prompt formatting. Choose between: alpaca, oasst, guanaco')
    parser.add_argument('--prompt', type=str, nargs='+', default=None, 
                        help='prompt to use instead of prompt_path, can be a list')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    
    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose between: none (no watermarking), christ, DISC , multikeychrist, OZ')
    # parser.add_argument('--method_detect', type=str, default='same',
    #                     help='Statistical test to detect watermark. Choose between: same (same as method), openai, openaiz, openainp, maryland, marylandz')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=4.0, 
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # multibit
    parser.add_argument('--payload', type=int, default=0, help='message')
    parser.add_argument('--payload_max', type=int, default=0, 
                        help='maximal message, must be inferior to the vocab size at the moment')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=1,# default = None  
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--do_eval', type=utils.bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None,
                        help='split the prompts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat prompts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat prompts as a whole')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)
    
    return parser

def format_prompts(prompts: List[Dict], prompt_type: str) -> List[str]:
    """
    This function forms the prompts as a list of strings, with "instructions"
    and "input" filled from the prompt data that is loaded

    Parameters
    ----------
    prompts : List[Dict]
        A list of prompts, e.g. one prompt of alpaca data is 
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
        }.
    prompt_type : str
       Options are: alpaca, oasst, guanaco.

    Returns
    -------
    List[str]
        List of prompts with their "instruction" and "input" filled from the 
        list of prompts that are filled. For the example above one prompt in the
        list of prompts will be 
        
        prompts = ['Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n']
    """
    if prompt_type=='alpaca':
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }
    elif prompt_type=='guanaco':
        PROMPT_DICT = {
            "prompt_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Input:\n{input}\n\n### Assistant:"
            ),
            "prompt_no_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Assistant:"
            )
        }
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts(json_path: str, prompt_type: str, nsamples: int=None) -> List[str]:
    """
    This function returns a list of prompts(str)

    Parameters
    ----------
    json_path : str
        path ot the prompt file data.
    prompt_type : str
        Options are: alpaca, oasst, guanaco.
    nsamples : int, optional
       Number of prompts to load, if it is set as None all
       the prompts are loaded. The default is None.

    Returns
    -------
    List[str]
        List of prompts.

    """
    with open(json_path, "r") as f:
        prompts = json.loads(f.read())
    new_prompts = prompts
    # new_prompts = [prompt for prompt in prompts if len(prompt["output"].split()) > 5]
    new_prompts = new_prompts[10:10+nsamples]
    # new_prompts = random.sample(new_prompts, k= nsamples)
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, prompt_type)
    return new_prompts

def Christgentest(
        model,
        tokenizer,
        prompts,
        keys,
        gen_length = 200,
        Rlambda = 5, 
        flagR = True,
        return_debug = False,
        verbose = True
):
    christ_watermark = ChristWatermark(model, tokenizer) 
    watermarked_texts = []
    for j in range(len(keys)):
        for i in range(len(prompts)):
            print(f"Generating watermarked text for prompt {i+1} of {len(prompts)}, gen_length = {gen_length}, Rlambda = {Rlambda}, flagR = {flagR}")
            watermarked_texts.append(christ_watermark.generate( 
                [prompts[i]],
                key = keys[j],
                max_gen_len = gen_length, 
                Rlambda = Rlambda, 
                flagR = flagR, 
                return_debug= return_debug, 
                verbose = verbose
                )[0])
            print(f"Decoding watermarked text for prompt {i+1} of {len(prompts)}, gen_length = {gen_length}, Rlambda = {Rlambda}, flagR = {flagR}")
            print(f"watermarked_texts[i]: {watermarked_texts[i]}")
            watermarked_texts[i] = christ_watermark.decode( 
                decoding_key = keys[j], 
                text = watermarked_texts[i], 
                Rlambda=Rlambda, 
                flagR=flagR,
                return_debug=return_debug,
                verbose = verbose
            )

    # # Print detailed attributes of each watermarked text object
    # for idx, obj in enumerate(watermarked_texts):
    #     print(f"--- WatermarkedText object {idx} ---")
    #     for attr, val in vars(obj).items():
    #         print(f"{attr}: {val}")    
    # Save watermarked_texts as a JSON file
    serializable = []
    for obj in watermarked_texts:
        # scalar fields
        n_val     = obj.n.item()     if torch.is_tensor(obj.n)     else obj.n
        nstar_val = obj.nstar.item() if torch.is_tensor(obj.nstar) else obj.nstar

        # 1D or 2D tensors -> Python lists
        P1_list         = obj.P1.tolist()            if torch.is_tensor(obj.P1)            else obj.P1
        random_vals     = obj.random_values
        if torch.is_tensor(random_vals):
            random_vals = random_vals.tolist()
        elif isinstance(random_vals, list):
            random_vals = [
                x.tolist() if torch.is_tensor(x) else x
                for x in random_vals
            ]

        serializable.append({
            "prompt":                obj.prompt,
            "text":                  obj.text,
            "token_ids":             obj.token_ids,
            "watermarked":           obj.watermarked,
            "P1":                    P1_list,
            "R":                     obj.R,
            "R_detected":            obj.R_detected,
            "random_values":         random_vals,
            "entropies":             obj.entropies.tolist() if torch.is_tensor(obj.entropies) else obj.entropies,
            "empirical_entropies":   obj.empirical_entropies.tolist() if torch.is_tensor(obj.empirical_entropies) else obj.empirical_entropies,
            "avg_entropy":           obj.avg_entropy,
            "avg_emp_entropy":       obj.avg_emp_entropy,
            "n":                     n_val,
            "nstar":                 nstar_val,
            "random_values_at_decode": obj.random_values_at_decode,
            "generation_key":        obj.generation_key,
            "watermarked_btkns":    obj.watermarked_btkns,
            "watermarked_tkns":     obj.watermarked_tkns,
            "decoded_message":      obj.decoded_message,
            "best_p_value":         obj.best_p_value,
            "p_value":              obj.p_value,
            "score":                obj.score,
            "best_score":           obj.best_score,
            "normalized_score":     obj.normalized_score,
            "best_normalized_score": obj.best_normalized_score,
            "tkn_scores":           obj.tkn_scores,
            "decoded_message":      obj.decoded_message,
            "detected_watermarked": obj.detected_watermarked,
            "watermarked_btkns_indx": obj.watermarked_btkns_indx,
        })
    return serializable

def christ_multibit_test(
    model,
    tokenizer,
    prompts,
    keys,
    gen_length = 200,
    Rlambda = 5, 
    nbits = 5,
    payload = 0,
    flagR = False,
    verbose = True,
    return_debug = False
    ):
    
    christ_watermark_multikey = ChristWatermarkMultiKey(model, tokenizer) 
    watermarked_texts = []
    for i in range(len(prompts)):
        print(f"Generating watermarked text for prompt {i+1} of {len(prompts)}, gen_length = {gen_length}, Rlambda = {Rlambda}, flagR = {flagR}, payload = {payload}, nbits = {nbits}")
        watermarked_texts.append(christ_watermark_multikey.generate(
            prompts = [prompts[i]],
            keys = keys,
            payload = payload,
            m_bits = nbits,
            max_gen_len = gen_length,
            Rlambda = Rlambda,
            flagR = flagR,
            return_debug = return_debug,
            verbose = verbose
        ))
        print(f"watermarked_texts[i]: {watermarked_texts[i]}")
        # Print detailed attributes of each watermarked text object
        for idx, obj in enumerate(watermarked_texts):
            print(f"--- WatermarkedText object {idx} ---")
            for attr, val in vars(obj).items():
                print(f"{attr}: {val}")    
        print(f"Decoding watermarked text for prompt {i+1} of {len(prompts)}, gen_length = {gen_length}, Rlambda = {Rlambda}, flagR = {flagR}, payload = {payload}, nbits = {nbits}")
        watermarked_texts[i] = christ_watermark_multikey.decode(
            decoding_keys = keys,
            text = watermarked_texts[i],
            Rlambda = Rlambda,
            flagR = flagR,
            return_debug = return_debug,
            verbose = verbose
        )

    serializable = []
    for obj in watermarked_texts:
        # Create a dictionary for each BinarizedWatermarkedTextMultiKey object
        obj_dict = {
            'nbits': obj.nbits,
            'payload': obj.payload,
            'encoding_keys': obj.encoding_keys,
            'decoding_keys': obj.decoding_keys,
            'detected_watermarked': obj.detected_watermarked,
            'detected_watermarked_keys': obj.detected_watermarked_keys,
            'detected_message': obj.detected_message,
            'max_scores': obj.max_scores,
            'scores': obj.scores,
            'BinarizedWatermarkedText_dict': {}
        }
        
        # Convert BinarizedWatermarkedText_dict to a serializable format
        for key, text_list in obj.BinarizedWatermarkedText_dict.items():
            obj_dict['BinarizedWatermarkedText_dict'][key] = []
            for text_obj in text_list:
                # Convert each BinarizedWatermarkedText object to a dictionary
                text_dict = {
                    'n': text_obj.n.item() if torch.is_tensor(text_obj.n) else text_obj.n,
                    'nstar': text_obj.nstar.item() if torch.is_tensor(text_obj.nstar) else text_obj.nstar,
                    'P1': text_obj.P1.tolist() if torch.is_tensor(text_obj.P1) else text_obj.P1,
                    'random_values': text_obj.random_values.tolist() if torch.is_tensor(text_obj.random_values) else text_obj.random_values,
                    'text': text_obj.text,
                    'token_ids': text_obj.token_ids.tolist() if torch.is_tensor(text_obj.token_ids) else text_obj.token_ids,
                    'watermarked': text_obj.watermarked,
                    'score': text_obj.score,
                    'normalized_score': text_obj.normalized_score,
                    'tkn_scores': text_obj.tkn_scores,
                    'best_score': text_obj.best_score,
                    'best_normalized_score': text_obj.best_normalized_score,
                    'p_value': text_obj.p_value,
                    'best_p_value': text_obj.best_p_value,
                    'decoded_message': text_obj.decoded_message,
                    'generation_key': text_obj.generation_key,
                    'detection_key': text_obj.detection_key,
                    'watermarked_tkns': text_obj.watermarked_tkns,
                    'watermarked_btkns': text_obj.watermarked_btkns,
                    'watermarked_btkns_indx': text_obj.watermarked_btkns_indx,
                    'detected_watermarked': text_obj.detected_watermarked,
                    'R_detected': text_obj.R_detected,
                    'R': text_obj.R
                }
                obj_dict['BinarizedWatermarkedText_dict'][key].append(text_dict)
        
        serializable.append(obj_dict)
    return serializable    
    
   
def BERtestDISC(nrun, model, tokenizer, prompt, nbits = 5, FPR = 1e-2, length = 30, Rlambda = 5, flagR = False, h=4):
    perfdict, wm_text_obj  = DISCgentest(nrun, model, tokenizer, prompt, nbits = nbits, FPR = FPR, length = length, Rlambda = Rlambda, flagR = flagR, h = h)
    
    nbiterrors = 0
    ncorrect= 0
    nfalseNegative = 0   
    for i in range(nrun):
        key = perfdict[i]['key']
        tokens = perfdict[i]['tokens']
        t0 = time.time()
        payload, nstar, Mstar, pstar, p, scores, indvScores, Ydetect = extract_payload_DISC(key, wm_text_obj, model, tokenizer, nbits, skip_prefix=0, FPR= FPR, h = h, flagR = flagR, verbose = False, deltailedData = True, flagTokens = True)
        t1 = time.time()
        if payload:
            payload = ("0"*nbits + bin(payload)[2:])[-nbits:]
            if payload == message:
                ncorrect +=1 
                nbiterror = 0
            else: 
                nbiterror = distance.hamming(list(payload),list(message))* nbits    
        else:
            nfalseNegative += 1
            nbiterror = nbits
        nbiterrors += nbiterror    
        
        perfdict[i]['extractedMessage'] = payload
        perfdict[i]['nbits'] = nbits
        perfdict[i]['detected'] = False if not payload else True
        perfdict[i]['nErrors'] = nbiterror
        perfdict[i]['nstar'] = nstar
        perfdict[i]['Mstar'] = Mstar
        perfdict[i]['pstar'] = pstar
        perfdict[i]['p'] = p
        perfdict[i]['scores'] = scores
        perfdict[i]['indvScores'] = indvScores
        perfdict[i]['Ydetect'] = Ydetect
        perfdict[i]['detectT'] = t1 - t0

        print(f"run {i+1}, response: {perfdict[i]['text']}")
        
    return nbiterror/(nbits* nrun), nfalseNegative/nrun, perfdict

def generate_watermarked_response_DISC(
    key,
    model,
    tokenizer,
    prompt,
    message,
    nbits=5,
    length=30,
    Rlambda=5,
    flagR=False,
    h=4,
    verbose=True,
    deltailedData=True
):
    """
    Generate a watermarked response using the DISC method.

    Parameters
    ----------
    key : float
        The shared key for watermarking.
    model : torch.nn.Module
        The language model used for generation.
    tokenizer : PreTrainedTokenizer
        The tokenizer associated with the model.
    prompts : List[str]
        A list of prompts to generate from.
    message : str
        The binary string payload to embed (e.g., '01010').
    nbits : int
        Number of bits in the message.
    length : int
        Length of the generated output (in tokens).
    Rlambda : float
        Threshold for switching to PRF-based sampling.
    flagR : bool
        Whether to use the R mode (stochastic embedding).
    h : int
        Context window size.
    verbose : bool
        Print debug info.
    deltailedData : bool
        Not used here but reserved for compatibility.

    Returns
    -------
    Tuple[str, List[int], List[float], List[int], List[float], float, float, float, float, int]
        Returns:
        - Generated string (text),
        - Token ids,
        - P1 values,
        - Y (random PRF values),
        - Token entropies,
        - Avg entropy,
        - Empirical entropy,
        - Avg empirical entropy,
        - n (number of bits generated).
    """
    # Convert binary string message to integer
    payload = int(message, 2)

    # Use the first prompt (or adjust if batching)
    prompt = prompts[0]

    # Initialize DISC object
    disc = DISC(model=model, tokenizer=tokenizer)

    # Call DISC.generate
    wm_text = disc.generate(
        key=key,
        prompt=prompt,
        payload=payload,
        m_bits=nbits,
        length=length,
        Rlambda=Rlambda,
        prob_mode='R' if flagR else None,
        h=h,
        flagPerm=False,
        verbose=verbose
    )

    return wm_text.text, wm_text.token_ids, wm_text.P1, wm_text.random_values, wm_text.entropies, wm_text.empirical_entropies, wm_text.avg_entropy, wm_text.avg_emp_entropy, wm_text.n, wm_text  # <- REmpEntropy


def extract_payload_DISC(
    key,
    wm_text_obj,  # must be BinarizedWatermarkedText or BinarizedText
    model,
    tokenizer,
    nbits=5,
    skip_prefix=1,
    FPR=1e-5,
    h=4,
    flagR=False,
    verbose=True,
    deltailedData=True,
    flagTokens=True,
):
    """
    Extract payload from watermarked text using DISC.
    
    Parameters
    ----------
    key : float
        Random seed used for encoding.
    wm_text_obj : BinarizedWatermarkedText
        The object returned by `DISC.generate()`.
    model : LlamaForCausalLM
        The language model.
    tokenizer : LlamaTokenizer
        The tokenizer used.
    nbits : int
        Number of payload bits.
    skip_prefix : int
        Number of tokens to skip at the beginning (e.g., <s> in LLaMA).
    FPR : float
        Desired false positive rate.
    h : int
        Context size.
    flagR : bool
        Whether using random embedding mode.
    verbose : bool
        If True, prints debug information.
    deltailedData : bool
        Whether to return full scoring and decoding details.
    flagTokens : bool
        If True, `wm_text_obj` is assumed to already contain token IDs.

    Returns
    -------
    payload : str
        Extracted binary payload string.
    nstar : int
        Detected starting index for the watermark.
    Mstar : int
        Index of deltaM with the best score.
    pstar : float
        Best p-value.
    p : list
        List of p-values for each candidate message.
    scores : list
        Final score for each candidate.
    indvScores : list
        Token-level scores for each candidate.
    Ydetect : list
        List of PRF values used during decoding.
    """
    decoder = DISC(model, tokenizer)
    decoded = decoder.decode(
        key=key,
        text=wm_text_obj,
        nbits=nbits,
        FPR=FPR,
        prob_mode='R' if flagR else None,
        context_mode='type2',
        h=h,
        verbose=verbose,
        flagPerm=False
    )

    if not getattr(decoded, "watermarked", False):
        return None, None, None, None, None, None, None, None

    payload = decoded.decoded_message
    binary_payload = ("0" * nbits + bin(payload)[2:])[-nbits:]

    if deltailedData:
        return (
            binary_payload,
            getattr(decoded, "n", None),
            getattr(decoded, "decoded_message", None),
            getattr(decoded, "best_p_value", None),
            getattr(decoded, "p_value", None),
            getattr(decoded, "score", None),
            getattr(decoded, "tkn_scores", None),
            getattr(decoded, "random_values_at_decode", None),
        )
    else:
        return binary_payload

def DISCgentest(
    nrun, model, tokenizer, prompts, 
    nbits=5, FPR=1e-2, length=30, Rlambda=5, 
    flagR=False, h=4, verbose=False
):
    """
    Generate nrun samples with DISC watermarking and collect performance info.

    Returns
    -------
    perfdict : dict
        Dictionary containing output info for each run:
        {
            run_id: {
                'prompt': ...,
                'tokens': ...,
                'text': ...,
                'key': ...,
                'message': ...,
                'decoded_message': ...,
                'detected': ...,
                'nErrors': ...,
                ...
            }
        }
    """

    disc = DISC(model, tokenizer)
    perfdict = {}

    for i in range(nrun):
        prompt = prompts[i % len(prompts)]
        message = CompactText.text_to_bits("OZ")[:nbits]  # fixed or random message
        payload = int(message, 2)
        key = random.random()

        # Generate watermarked text
        wm_text = disc.generate(
            key=key,
            prompt=prompt,
            payload=payload,
            m_bits=nbits,
            length=length,
            Rlambda=Rlambda,
            prob_mode='R' if flagR else 'random_embedding',
            h=h,
            verbose=verbose
        )
        # Try decoding
        decoded = disc.decode(
            key=key,
            text=wm_text,
            nbits=nbits,
            FPR=FPR,
            prob_mode='R' if flagR else None,
            h=h,
            verbose=False
        )

        # Evaluate
        bit_error = None
        detected = getattr(decoded, 'watermarked', False)
        if detected and hasattr(decoded, 'decoded_message'):
            extracted = decoded.decoded_message
            extracted_bin = ("0" * nbits + bin(extracted)[2:])[-nbits:]
            bit_error = distance.hamming(list(extracted_bin), list(message)) * nbits
        else:
            extracted_bin = None
            bit_error = nbits  # total bit error if not detected

        # Save result
        perfdict[i] = {
            'prompt': prompt,
            'tokens': wm_text.token_ids,
            'text': wm_text.text,
            'key': key,
            'message': message,
            'extractedMessage': extracted_bin,
            'nbits': nbits,
            'detected': detected,
            'nErrors': bit_error,
            'nstar': getattr(decoded, 'n', None),
            'Mstar': getattr(decoded, 'decoded_message', None),
            'pstar': getattr(decoded, 'best_p_value', None),
            'p': getattr(decoded, 'p_value', None),
            'scores': getattr(decoded, 'score', None),
            'indvScores': getattr(decoded, 'tkn_scores', None),
            'Ydetect': getattr(decoded, 'random_values_at_decode', None),
            'detectT': None  # Can add timing if needed
        }

    return perfdict, wm_text

if __name__ == '__main__':   
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # # grab state for later:
    # state = torch.cuda.get_rng_state_all()
    # torch.save(state, "rng_state.pt")
    
    # # reload the exact RNG state:
    # state = torch.load("rng_state.pt")
    # torch.cuda.set_rng_state_all(state)
    method = "christ"
    model_name = "llama-7b" 
    # prompt_path = "./data/alpaca_data.json"
    # prompt_type = "alpaca"
    nsamples = 1
    # build model
    if model_name == "llama-7b":
        model_name = "huggyllama/llama-7b"#"meta-llama/Llama-2-7b-hf"
        model_name_or_path = pre_path + "/llama-2-7b"# "meta-llama/Llama-2-7b-hf"
        adapters_name = None
    elif model_name == "llama-7b-chat":
        model_name = "huggyllama/llama-7b-chat"
        model_name_or_path = pre_path + "/llama-2-7b-chat"
        adapters_name = None
    elif model_name == "llama-13b-chat":
        model_name = "huggyllama/llama-13b-chat"
        model_name_or_path = pre_path + "/llama-2-13b-chat"
        adapters_name = None
    elif model_name == "llama-70b-chat":
        model_name = "huggyllama/llama-70b-chat"
        model_name_or_path = pre_path + "/llama-2-70b-chat"
        adapters_name = None    
    elif model_name == "guanaco-7b":
        model_name = "huggyllama/llama-7b"
        model_name_or_path = pre_path + "/llama-2-7b"
        adapters_name = 'timdettmers/guanaco-7b'
    elif model_name == "guanaco-13b":
        model_name = "huggyllama/llama-13b"
        model_name_or_path = pre_path + "/llama-2-13b"
        adapters_name = 'timdettmers/guanaco-13b'
    elif model_name == "gpt2":
        model_name = "gpt2"   
        adapters_name = None
    # config = AutoConfig.from_pretrained(
    #     model_name_or_path
    #     )    

    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if adapters_name is not None:
        model = PeftModel.from_pretrained(model, adapters_name)
    for param in model.parameters():
        param.requires_grad = False
    
    ngpus = torch.cuda.device_count()
    print(f"Using {ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")
    
    # model, tokenizer = start_model("gpt2")  # Requires a LLamma token ID
    
    # load prompts
    start_idx = 10
    end_idx = 13
    with open('./data/c4_subset_10000_15_20words_prompts.json', 'r') as f:
        prompts = json.load(f)[start_idx:end_idx]   
        
    ## This section is for generating one response with Chrsit, DISC and OZ method
    keys = [35317]
    # message = CompactText.text_to_bits("OZ")
    nbits = 3
    payload = 0
    Rlambda = 5
    flagR = False
    nrun = 1
    gen_length = 80
    method = "christ"
    method = "multikeychrist"

    if method == "christ":
        serializable = Christgentest(
            model,
            tokenizer,
            prompts,
            keys= keys,
            gen_length = gen_length,
            Rlambda = Rlambda, 
            flagR = flagR,
            return_debug = True,
            verbose = True
        )
        file_name = "watermarked_" + method + "_" + "len_prompts" + str(len(prompts)) + "_start_idx" + str(start_idx) + "_end_idx" + str(end_idx) + "_len_key" + str(len(keys)) + "_flagR" + str(flagR) + "_gen_length" + str(gen_length) + "_Rlambda" + str(Rlambda) + ".json"
        with open(file_name,"w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Watermarked texts saved to {file_name}")
        
    elif method == "multikeychrist":
        key = keys[0]
        # Create a list of keys starting from 'key' and incrementing by 1
        keys_multikey = [key + i for i in range(2**nbits)]  # create 2^nbits unique keys starting from key

        serializable = christ_multibit_test(
            model,
            tokenizer,
            prompts,
            keys=keys_multikey,
            gen_length=gen_length,
            Rlambda=Rlambda,
            nbits=nbits,
            payload=0,  # default payload
            flagR=flagR,
            verbose=True,
            return_debug=True
        )
        file_name = "watermarked_" + method + "_" + "len_prompts" + str(len(prompts)) + "_start_idx" + str(start_idx) + "_end_idx" + str(end_idx) + "_len_key" + str(len(keys)) + "_flagR" + str(flagR) + "_gen_length" + str(gen_length) + "_Rlambda" + str(Rlambda) + "_payload" + str(payload) + "_nbits" + str(nbits) + ".json"
        with open(file_name,"w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Watermarked texts saved to {file_name}")
       
    # resOZ, eccOZ, tokensOZ = generate_payloaded_response_OZ(key, model, tokenizer, prompts , message, 210, threshold=1.7, bit_limit=None, temperature=1.4)
    # resChrist, tokensChrist, P1vecChrist, YChrist, entropyChrist, empEntropyChrist, avgEntropyChrist, avgEmpEntropyChrist, nChrist  = generate_watermarked_response_Christ(key, model, tokenizer, prompts, length=30, Rlambda = 5, flagR = False, verbose = True, deltailedData = True)
    # for prompt in prompts:
        # resDISC, tokensDISC, P1vecDISC,YDISC, entropyDISC, empEntropyDISC, avgEntropyDISC, avgEmpEntropyDISC, REmpEntropy, nDISC, wm_text_obj = generate_watermarked_response_DISC(key, model, tokenizer, prompt,'00000', nbits = nbits, length = 30, Rlambda = 5, flagR = False, h=4, verbose= True, deltailedData = True)
        # payloadDISC,nstarDISC, MstarDISC, pstarDISC, pDISC, scoresDISC, indvScoresDISC, YDISCdetect = extract_payload_DISC(key, wm_text_obj, model, tokenizer, nbits, skip_prefix=1, FPR= 1e-5, h = 4, flagR= False, verbose = True, deltailedData = True)
        # if payloadDISC:
        #     payloadDISC = ("0"*nbits + bin(payloadDISC)[2:])[-nbits:]
        # ber, fnr, perfdict = BERtestDISC(nrun, model, tokenizer, prompt, nbits, length=100, Rlambda=Rlambda)
        # print(f"ber is {ber}, fnr is {fnr}\n")
    # print("OZ watermarked text:", resOZ)
    # assert(CompactText.bits_to_text(payload[:len(message)]) == "OZ")
    # print("sent symbols:",eccOZ.stream)
    # print("the message that will be decoded:",DynamicECC.decode(eccOZ.stream), ",payload=", message)
    # payload = extract_payload_OZ(key, tokensOZ, tokenizer, threshold=1.7, bit_limit=None, skip_prefix=1)
    # print("Christ watermarked text:", resChrist)
    # watermarkStatus, nstarChrist, scoresChrist,nscoreChrist, indvScoresChrist, YChrsitdetect  = detect_watermark_Christ(key, tokensChrist, tokenizer, Rlambda, skip_prefix=1, flagR = False, deltailedData = True)
   
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")    
# if __name__ == '__main__':
#     args = get_args_parser().parse_args()
#     main(args)
     
