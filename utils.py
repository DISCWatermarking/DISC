# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import math
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

class GrayCodeConverter:
    @staticmethod
    def int2gray(n: int, m_bits: int) -> int:
        """
        Convert an integer n into its m-bit Gray code integer representation.

        Parameters
        ----------
        n : int
            A non-negative integer.
        m_bits : int
            Number of bits to consider for binary representation.

        Returns
        -------
        int
            The Gray code representation as an integer.
        """
        if n >= (1 << m_bits):
            raise ValueError(f"Integer {n} exceeds {m_bits}-bit capacity")
        return n ^ (n >> 1)

    @staticmethod
    def gray2int(g: int, m_bits: int) -> int:
        """
        Convert an m-bit Gray code integer g back to its original integer.

        Parameters
        ----------
        g : int
            Gray code integer representation.
        m_bits : int
            Number of bits used in the encoding.

        Returns
        -------
        int
            Original binary integer.
        """
        if g >= (1 << m_bits):
            raise ValueError(f"Gray code {g} exceeds {m_bits}-bit capacity")
        result = 0
        for shift in reversed(range(m_bits)):
            result ^= (g >> shift)
        return result

def PRF(key, input):
    # Lazy and insecure implementation, replace with a provably secure PRF for real applications
    random.seed(str(key) + "||" + str(input))
    return random.random()

def entropyfnc(prob_list):
    """
    Compute the Shannon entropy for a probability distribution or a batch of distributions.

    Args:
        prob_list (torch.Tensor): A probability distribution tensor. If a 2D tensor is provided, each row is assumed to be a distribution.

    Returns:
        If the input is 1D, returns a float representing the entropy.
        If the input is 2D (or has more dimensions), returns a torch.Tensor of entropies.
    """
    if not isinstance(prob_list, torch.Tensor):
        prob_list = torch.tensor(prob_list, dtype=torch.float32)
    else:
        prob_list = prob_list.to(dtype=torch.float32)
    
    epsilon = 1e-10  # Small constant to avoid log(0)
    if prob_list.ndim == 1:
        return -torch.sum(prob_list * torch.log(prob_list + epsilon)).item()
    else:
        # Compute entropy along the last dimension for each distribution
        return -torch.sum(prob_list * torch.log(prob_list + epsilon), dim=-1).reshape(-1)


def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')


def start_model(model_name="gpt2"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer



def strbintobin(bin_string, blen=0):
    """
    Convert a binary string (e.g. '101') into a list of integer bits (e.g. [1, 0, 1]),
    optionally zero-padding on the left so that the final list has length = blen.

    Parameters
    ----------
    bin_string : str
        A string consisting of '0' and '1' characters. For example, '101'.
    blen : int, optional
        If > 0, the string is left-padded with '0' up to length blen.

    Returns
    -------
    list of int
        A list of bits, e.g. [1, 0, 1].
    """
    # 1) Left-pad the binary string to 'blen' characters, if blen > 0.
    if blen > 0:
        bin_string = bin_string.zfill(blen)

    # 2) Convert each character ('0' or '1') to an integer (0 or 1).
    return [int(ch) for ch in bin_string]




  
    