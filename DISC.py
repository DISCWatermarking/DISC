############################################################
# The following code implements multi-bit watermarking     #
# algorithm DISC from "Multi-Bit Distortion-Free           #
# Watermarking for Large Language Models",                   #
#                                                          #
# It also implements Algorithm 3 from "Excuse me, sir?     #
# Your language model is leaking (information)", authored  #
# by Zamir (Tel Aviv University, orzamir@tauex.tau.ac.il). #
#                                                          #
# Additionally, the watermarking algorithm from            #
# "Undetectable Watermarks for Language Models", authored  #
# by Christ et al., is implemented. A multi-bit variation  #
# of that algorithm, which utilizes multiple keys, is also #
# implemented. Some parts of this code are extensions and  #
# modifications of the code written by Or Zamir, available #
# at https://github.com/OrZamir/steg.                      #
############################################################

import argparse
from typing import Dict, List, Union, Optional

import random
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
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    DynamicCache
    )
from sentence_transformers import SentenceTransformer

from dynamic_ecc import DynamicECC
from utils import PRF, start_model, \
         entropyfnc, GrayCodeConverter, strbintobin
from compact_text import CompactText

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Text:
    """
    A base container class for text data, possibly watermarked or not.
    Using cooperative multiple inheritance with **kwargs.
    """

    def __init__(
        self,
        prompt=None,
        text=None,
        token_ids=None,
        generation_key=None,
        detection_key=None,
        random_values=None,
        random_values_at_decode=None,
        watermarked=False,
        score=None,
        normalized_score=None,
        tkn_scores=None,
        best_score=None,
        best_normalized_score=None,
        p_value=None,
        best_p_value=None,
        decoded_message=None,
        **kwargs
    ):
        """
        Base attributes for any text.
        Parameters
        ----------
        prompt : str, optional
            The prompt used for generation.
        text : str
            The generated text.
        token_ids : list of int, optional
            The token IDs associated with the text.
        generation_key : any, optional
            Key used for generation.    
        detection_key : any, optional
            Key used for detection.
        random_values : list of float, optional
            Random draws used for each bit (if any).
        watermarked : bool, optional
            Whether the text is watermarked.
        score : float, optional
            Score for detection.
        normalized_score : float, optional
            Normalized detection score.
        tkn_scores : list of float, optional
            Scores for each token decision.
        best_score : float, optional
            Best detection score encountered.
        best_normalized_score : float, optional
            Best normalized detection score encountered.
        p_value : float, optional
            p-value for detection.
        best_p_value : float, optional
            Best p-value (if multiple tests).
        decoded_message : str, optional
            Decoded watermark message (if applicable).
        """
        self.prompt = prompt
        self.text = text
        self.token_ids = [] if token_ids is None else token_ids
        self.generation_key = generation_key
        self.detection_key = detection_key
        self.random_values = [] if random_values is None else random_values
        self.random_values_at_decode = [] if random_values_at_decode is None else random_values_at_decode
        self.watermarked = watermarked
        self.score = score
        self.normalized_score = normalized_score
        self.tkn_scores = [] if tkn_scores is None else tkn_scores
        self.best_score = best_score
        self.best_normalized_score = best_normalized_score
        self.p_value = p_value
        self.best_p_value = best_p_value
        self.decoded_message = decoded_message

        # Call the next class in the MRO
        super().__init__(**kwargs)

class BinarizedText(Text):
    """
    Adds a 'P1' field (probabilities of bit=1) for binarized text.
    """

    def __init__(self, P1=None, **kwargs):
        """
        Parameters
        ----------
        P1 : list of float
            A list of probabilities for each bit decision, if applicable.
        """
        self.P1 = [] if P1 is None else P1
        # Continue up the MRO chain for other attributes
        super().__init__(**kwargs)

    @classmethod
    def from_text(cls, text_obj, P1=None):
        """
        Create a BinarizedText from an existing Text.

        Parameters
        ----------
        text_obj : Text
            An existing Text object.
        P1 : list of float, optional
            Probabilities for bit=1. If omitted, defaults to empty list.

        Returns
        -------
        BinarizedText
        """
        # Use the fields from text_obj and add P1
        return cls(
            P1=P1,
            prompt=text_obj.prompt,
            text=text_obj.text,
            token_ids=text_obj.token_ids,
            generation_key=text_obj.generation_key,
            detection_key=text_obj.detection_key,
            random_values=text_obj.random_values,
            random_values_at_decode=text_obj.random_values_at_decode,
            watermarked=text_obj.watermarked,
            score=text_obj.score,
            normalized_score=text_obj.normalized_score,
            tkn_scores=text_obj.tkn_scores,
            best_score=text_obj.best_score,
            best_normalized_score=text_obj.best_normalized_score,
            p_value=text_obj.p_value,
            best_p_value=text_obj.best_p_value,
            decoded_message=text_obj.decoded_message
        )
    @classmethod
    def from_binarized_watermarked_text(cls, bwt_obj):
        """
        Create a BinarizedText object from an existing BinarizedWatermarkedText object.

        Parameters
        ----------
        bwt_obj : BinarizedWatermarkedText
            The specialized watermarked text object to be converted.

        Returns
        -------
        BinarizedText
            A new BinarizedText instance with fields copied from bwt_obj.
        """
        return cls(
            # BinarizedText-specific field
            P1=bwt_obj.P1,
            # Fields inherited from Text
            prompt=bwt_obj.prompt,
            text=bwt_obj.text,
            token_ids=bwt_obj.token_ids,
            generation_key=bwt_obj.generation_key,
            detection_key=bwt_obj.detection_key,
            random_values=bwt_obj.random_values,
            random_values_at_decode=bwt_obj.random_values_at_decode,
            watermarked=bwt_obj.watermarked,
            score=bwt_obj.score,
            normalized_score=bwt_obj.normalized_score,
            tkn_scores=bwt_obj.tkn_scores,
            best_score=bwt_obj.best_score,
            best_normalized_score=bwt_obj.best_normalized_score,
            p_value=bwt_obj.p_value,
            best_p_value=bwt_obj.best_p_value,
            decoded_message=bwt_obj.decoded_message      
        )    

class WatermarkedText(Text):
    """
    Holds metadata and final text for watermarked outputs.
    Inherits from the base Text class and adds watermark-specific attributes.
    """

    def __init__(
        self,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        watermarked_tkns=None,
        **kwargs
    ):
        """
        entropies : list of float, optional
            Entropy of each token's distribution.
        empirical_entropies : list of float, optional
            -log(prob) for each chosen token.
        avg_entropy : float, optional
            Average Shannon entropy.
        avg_emp_entropy : float, optional
            Average empirical entropy.
        embedded_message : str, optional
            Any embedded watermark message.
        watermarked_tkns : list of int, optional
            Indices of tokens marked as watermarked.    
        """
        self.entropies = [] if entropies is None else entropies
        self.empirical_entropies = [] if empirical_entropies is None else empirical_entropies
        self.avg_entropy = avg_entropy
        self.avg_emp_entropy = avg_emp_entropy
        self.embedded_message = embedded_message
        self.watermarked_tkns = [] if watermarked_tkns is None else watermarked_tkns
        # Continue up the MRO chain
        super().__init__(**kwargs)

    @classmethod
    def from_text(
        cls,
        text_obj,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        watermarked_tkns=None,
    ):
        """
        Create a WatermarkedText from an existing Text object.

        Parameters
        ----------
        text_obj : Text
            An existing Text object to be converted.
        entropies : list of float, optional
            Shannon entropies for each token's distribution.
        empirical_entropies : list of float, optional
            -log(prob of chosen token).
        avg_entropy : float, optional
            Average Shannon entropy over tokens.
        avg_emp_entropy : float, optional
            Average empirical entropy.
        embedded_message : str, optional
            The embedded watermark message.
        watermarked_tkns : list of int, optional
            Indices of tokens marked as watermarked.    

        Returns
        -------
        WatermarkedText
        """
        return cls(
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            watermarked_tkns=watermarked_tkns,
            prompt=text_obj.prompt,
            text=text_obj.text,
            token_ids=text_obj.token_ids,
            random_values=text_obj.random_values,
            random_values_at_decode=text_obj.random_values_at_decode,
            watermarked=text_obj.watermarked,
            score=text_obj.score,
            normalized_score=text_obj.normalized_score,
            tkn_scores=text_obj.tkn_scores,
            best_score=text_obj.best_score,
            best_normalized_score=text_obj.best_normalized_score,
            p_value=text_obj.p_value,
            best_p_value=text_obj.best_p_value,
            decoded_message=text_obj.decoded_message,
            generation_key=text_obj.generation_key,
            detection_key=text_obj.detection_key,
        )    

class BinarizedWatermarkedText(BinarizedText, WatermarkedText):
    """
    A specialized class that inherits from both BinarizedText and WatermarkedText
    in a cooperative multiple inheritance manner.
    """

    def __init__(self, n=None, nstar=None, watermarked_btkns=None, watermarked_btkns_indx=None, detected_watermarked=None, R=None, R_detected=None, **kwargs):
        """
        R: list of bits per sample, cleaned of pad_id elements.
       
        n : int, optional
            Extra field, e.g. number of bits used before switching from random sampling.
        nstar : int, optional
            The index of the first binary token of the watermark.
        watermarked_btkns : list of int, optional
            Binary token IDs specifically marked as watermarked.
        watermarked_btkns_indx : list of int, optional
            Indices of the watermarked bits in the watermarked_btkns list.
        detected_watermarked : bool, optional
            Whether a watermark has been detected.
        """
        self.n = 0 if n is None else n
        self.nstar = 0 if nstar is None else nstar
        self.watermarked_btkns = [] if watermarked_btkns is None else watermarked_btkns
        self.watermarked_btkns_indx = [] if watermarked_btkns_indx is None else watermarked_btkns_indx
        self.detected_watermarked = False if detected_watermarked is None else detected_watermarked
        self.R = [] if R is None else R
        self.R_detected = [] if R_detected is None else R_detected
        # Use super() to traverse MRO in a single chain
        super().__init__(**kwargs)
    @classmethod
    def from_binarized_text(
        cls,
        bin_text_obj,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        n=None,
        nstar = None, 
        watermarked_btkns=None,
        watermarked_btkns_indx=None,
        detected_watermarked=None,
        R_detected=None,
        R=None,
    ):
        """
        Convert a BinarizedText into a BinarizedWatermarkedText by adding
        watermark fields.
        """
        return cls(
            text=bin_text_obj.text,
            token_ids=bin_text_obj.token_ids,
            random_values=bin_text_obj.random_values,
            P1=bin_text_obj.P1,
            watermarked=bin_text_obj.watermarked,
            score=bin_text_obj.score,
            normalized_score=bin_text_obj.normalized_score,
            tkn_scores=bin_text_obj.tkn_scores,
            best_score=bin_text_obj.best_score,
            best_normalized_score=bin_text_obj.best_normalized_score,
            p_value=bin_text_obj.p_value,
            best_p_value=bin_text_obj.best_p_value,
            decoded_message=bin_text_obj.decoded_message,
            generation_key=bin_text_obj.generation_key,
            detection_key=bin_text_obj.detection_key,
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            n=n,
            nstar = nstar, 
            watermarked_btkns=watermarked_btkns,
            watermarked_btkns_indx=watermarked_btkns_indx,
            detected_watermarked=detected_watermarked,
            R_detected=R_detected,
            R=R,
        )

    @classmethod
    def from_watermarked_text(
        cls,
        wtr_text_obj,
        P1=None,
        n=None,
        nstar = None,
        watermarked_btkns=None,
        watermarked_btkns_indx=None,
        detected_watermarked=None,
        R_detected=None,
        R=None,
    ):
        """
        Convert a WatermarkedText into a BinarizedWatermarkedText by adding
        binarized fields (P1, etc.).
        """
        return cls(
            text=wtr_text_obj.text,
            token_ids=wtr_text_obj.token_ids,
            random_values=wtr_text_obj.random_values,
            watermarked=wtr_text_obj.watermarked,
            score=wtr_text_obj.score,
            normalized_score=wtr_text_obj.normalized_score,
            tkn_scores=wtr_text_obj.tkn_scores,
            best_score=wtr_text_obj.best_score,
            best_normalized_score=wtr_text_obj.best_normalized_score,
            p_value=wtr_text_obj.p_value,
            best_p_value=wtr_text_obj.best_p_value,
            decoded_message=wtr_text_obj.decoded_message,
            entropies=wtr_text_obj.entropies,
            empirical_entropies=wtr_text_obj.empirical_entropies,
            avg_entropy=wtr_text_obj.avg_entropy,
            avg_emp_entropy=wtr_text_obj.avg_emp_entropy,
            embedded_message=wtr_text_obj.embedded_message,
            detection_key=wtr_text_obj.detection_key,
            watermarked_tkns=wtr_text_obj.watermarked_tkns,
            P1=P1,
            n=n,
            nstar = nstar,
            watermarked_btkns=watermarked_btkns,
            watermarked_btkns_indx=watermarked_btkns_indx,
            detected_watermarked=detected_watermarked,
            R_detected=R_detected,
            R=R,
        )

    @classmethod
    def from_text(
        cls,
        text_obj,
        P1=None,
        entropies=None,
        empirical_entropies=None,
        avg_entropy=None,
        avg_emp_entropy=None,
        embedded_message=None,
        n=None,
        nstar = None,
        watermarked_tkns=None,
        watermarked_btkns=None,
        watermarked_btkns_indx=None,
        detected_watermarked=None,
        R_detected=None,
        R=None,
    ):
        """
        Convert a plain Text into a BinarizedWatermarkedText directly,
        adding both binarized and watermark fields.
        """
        return cls(
            text=text_obj.text,
            token_ids=text_obj.token_ids,
            random_values=text_obj.random_values,
            watermarked=text_obj.watermarked,
            score=text_obj.score,
            normalized_score=text_obj.normalized_score,
            tkn_scores=text_obj.tkn_scores,
            best_score=text_obj.best_score,
            best_normalized_score=text_obj.best_normalized_score,
            p_value=text_obj.p_value,
            best_p_value=text_obj.best_p_value,
            decoded_message=text_obj.decoded_message,
            generation_key=text_obj.generation_key,
            detection_key=text_obj.detection_key,
            watermarked_tkns=watermarked_tkns,
            watermarked_btkns=watermarked_btkns,
            P1=P1,
            entropies=entropies,
            empirical_entropies=empirical_entropies,
            avg_entropy=avg_entropy,
            avg_emp_entropy=avg_emp_entropy,
            embedded_message=embedded_message,
            n=n,
            nstar = nstar,
            watermarked_btkns_indx=watermarked_btkns_indx,
            detected_watermarked=detected_watermarked,
            R_detected=R_detected,
            R=R,
        )

class BinarizedWatermarkedTextMultiKey:
    """
    A specialized class for handling watermarked text with multiple keys.
    Extends BinarizedWatermarkedText to include multi-key specific attributes.
    """
    def __init__(
        self,
        nbits=None,
        payload=None,
        encoding_keys=None,
        decoding_keys=None,
        BinarizedWatermarkedText_dict=None,
        detected_watermarked=None,
        detected_watermarked_keys=None,
        detected_message=None,
        max_scores=None,
        scores=None
    ):
        """
        Initialize a BinarizedWatermarkedTextMultiKey object.

        Parameters
        ----------
        nbits : int
            Number of bits in the payload
        payload : str
            The payload to be embedded
        encoding_keys : list
            List of keys used for encoding
        decoding_keys : list
            List of keys used for decoding
        BinarizedWatermarkedText_dict : dict
            Dictionary mapping keys to their corresponding BinarizedWatermarkedText objects
        detected_watermarked : dict
            Dictionary mapping nstar keys to boolean values indicating if watermark was detected
        detected_watermarked_keys : dict
            Dictionary mapping nstar keys to the index of the detected message
        detected_message : dict
            Dictionary mapping nstar keys to the index of the detected message
        max_scores : dict
            Dictionary mapping nstar keys to tuples of (max_score, key)
        scores : dict
            Dictionary mapping nstar keys to lists of (score, key) tuples
        **kwargs : dict
            Additional arguments passed to BinarizedWatermarkedText constructor
        """
        self.nbits = nbits
        self.payload = payload
        self.encoding_keys = encoding_keys
        self.decoding_keys = decoding_keys
        self.BinarizedWatermarkedText_dict = BinarizedWatermarkedText_dict
        self.detected_watermarked = detected_watermarked
        self.detected_watermarked_keys = detected_watermarked_keys
        self.detected_message = detected_message
        self.max_scores = max_scores
        self.scores = scores

        
    
###############################################
# BINARIZED TEXT GENERATOR (NO WATERMARK)
###############################################
class BinarizedLLM:
    """
    This class wraps a language model and a tokenizer to generate text in a 'binarized' fashion.
    """
    @torch.no_grad()
    def __init__(self,
                model, 
                tokenizer,
                max_gen_len: int =200,
                temperature: float = 0.8,
                top_p: float = 0.95,
                seed: int = 0,
                ):
        """
        Initialize with a model and its tokenizer.
        
        Parameters
        ----------
        model : PreTrainedModel
            The language model to be used (e.g., GPT-like model).
        tokenizer : PreTrainedTokenizer
            The tokenizer corresponding to the model.
        """
        # model config
        self.max_seq_len = model.config.max_sequence_length
        # self.pad_id = model.config.pad_token_id
        self.eos_id = model.config.eos_token_id
        self.model = model
        self.tokenizer = tokenizer
        self.temerature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.evolving_rng_seed = seed
        self.evolving_rng = torch.Generator(device=self.model.device)
        self.evolving_rng.manual_seed(self.evolving_rng_seed)
        # Setup for binarization: pre-compute the number of bits needed, 
        # plus token <-> id dictionaries.
        self.blen, self.token_to_id, self.id_to_token = self._setup_binarization()

    @torch.no_grad()
    def _setup_binarization(self):
        """
        Prepares the dictionaries for binarizing the LLM tokens.
        
        Returns
        -------
        blen : int
            Number of binary tokens equivalent to a real token 
            (ceiling of log2 of vocabulary size).
        token_to_id : dict
            {token_string : vocab_id}.
        id_to_token : dict
            {vocab_id : token_string}.
        """
        vocab_size = len(self.tokenizer)
        # print(f"vocab_size = {vocab_size}")
        blen = math.ceil(math.log2(vocab_size))
        token_to_id = self.tokenizer.get_vocab()
        id_to_token = {v: k for (k, v) in token_to_id.items()}
        return blen, token_to_id, id_to_token

    @torch.no_grad()
    def _aux_tokenize(self, text, skip_prefix=0):
        """
        Tokenize a text prompt into model-ready format.
        This method was supposed to resolve an issue with tokenization in LLama models.
        I think this can be removed or updated in the future.

        Parameters
        ----------
        text : str
            The text prompt to tokenize.
        skip_prefix : int, optional
            Number of tokens to skip from the beginning.

        Returns
        -------
        list
            The tokenized prompt as a PyTorch tensor (batch dimension included).
        """        
        if "llama" in self.tokenizer.name_or_path:
            text = "-" + text.split(" ", 1)[0] + text.split(" ", 1)[1]
            skip_prefix = 2  # Adjust for leading <s> token
            tokens = self._tokenize(text)[0][skip_prefix:]
        return tokens.tolist()
    
    @torch.no_grad()
    def _tokenize(self, 
                prompts: List[str]
                ) -> torch.tensor:
        
        """
        Tokenize a text prompt into model-ready format.
        
        Parameters
        ----------
        prompt : str
            The prompt to tokenize.
        Returns
        -------
        torch.Tensor
            The tokenized prompt as a PyTorch tensor.
        """
        # bsz = len(prompts) # batch size 
        # prompt_tokens = [self.tokenizer.encode(x, 
        #                              return_tensors='pt', 
        #                              truncation=True, 
        #                              add_special_tokens=False,
        #                              max_length=self.max_seq_len)
        #                             for x in prompts]
        # min_prompt_size = min([len(t) for t in prompt_tokens])
        # max_prompt_size = max([len(t) for t in prompt_tokens])
        # total_len = min(self.max_seq_len, self.max_gen_len + max_prompt_size)
        return self.tokenizer.encode(prompts, 
                                     return_tensors='pt', 
                                     truncation=True, 
                                     add_special_tokens=False,
                                     )
                                   
    @torch.no_grad()
    def _detokenize(self, token_ids):
        """
        Convert token IDs back to human-readable text.
        
        Parameters
        ----------
        token_ids : list or torch.Tensor
            The sequence of token IDs. Can be a single sequence or a batch (list of sequences).
        
        Returns
        -------
        str or List[str]
            The decoded text(s), with special tokens skipped.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def _binarize_next(self, probs, ind=0, blen=16, prefix=0):
        """
        Given the probability distribution over the vocabulary, 
        compute the partial sums p0 and p1 corresponding to the 
        next binary bit.
        
        Supports both single distribution (1D tensor) and batch (2D tensor).
        
        Parameters
        ----------
        probs : torch.Tensor
            Probability distribution over the entire vocabulary for a single 
            sample (1D) or a batch of distributions (2D, shape: [B, V]).
        ind : int, optional
            Index of the binary bit we're deciding (0-based).
        blen : int, optional
            Total number of bits needed to choose one real token.
        prefix : int or torch.Tensor, optional
            Accumulated bits so far for the current token. Can be scalar (for all samples)
            or a tensor of shape (B,).
        
        Returns
        -------
        (p0, p1) :
            If probs is 1D: Two torch.Tensors each containing a single value.
            If probs is 2D: Two torch.Tensors of shape (B,) with the partial probabilities.
        """
        if probs.ndim == 1:
            p0 = torch.tensor(0.0, device=probs.device)
            p1 = torch.tensor(0.0, device=probs.device)
    
            start_id = prefix << (blen - ind)
            end_id = min((prefix + 1) << (blen - ind), probs.size(0))
    
            for vocab_id in range(start_id, end_id):
                if (vocab_id >> (blen - ind - 1)) & 1 == 0:
                    p0 += probs[vocab_id]
                else:
                    p1 += probs[vocab_id]
    
            return p0, p1
        elif probs.ndim == 2:
            batch_size, vocab_size = probs.shape
            # If prefix is a scalar, use same prefix for all samples.
            if isinstance(prefix, int):
                start = prefix << (blen - ind)
                end = min((prefix + 1) << (blen - ind), vocab_size)
                p0 = torch.zeros(batch_size, device=probs.device)
                p1 = torch.zeros(batch_size, device=probs.device)
                for vocab_id in range(start, end):
                    bit = (vocab_id >> (blen - ind - 1)) & 1
                    if bit == 0:
                        p0 += probs[:, vocab_id]
                    else:
                        p1 += probs[:, vocab_id]
                return p0, p1
            else:
                # Assume prefix is a tensor of shape (batch_size,).
                p0_list = []
                p1_list = []
                for b in range(batch_size):
                    pref_b = prefix[b].item() if torch.is_tensor(prefix) else prefix[b]
                    start_b = pref_b << (blen - ind)
                    end_b = min((pref_b + 1) << (blen - ind), vocab_size)
                    p0_val = torch.tensor(0.0, device=probs.device)
                    p1_val = torch.tensor(0.0, device=probs.device)
                    for vocab_id in range(start_b, end_b):
                        if (vocab_id >> (blen - ind - 1)) & 1 == 0:
                            p0_val += probs[b, vocab_id]
                        else:
                            p1_val += probs[b, vocab_id]
                    p0_list.append(p0_val)
                    p1_list.append(p1_val)
                p0 = torch.stack(p0_list)
                p1 = torch.stack(p1_list)
                return p0, p1
        else:
            raise ValueError("probs must be a 1D or 2D tensor")

    @torch.no_grad()
    def generate(self, 
                 prompts: List[str],
                 max_gen_len: int = 30, 
                 length=30,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 return_debug: bool = False) -> List[BinarizedText]:
        """
        Watermarked response generation using binarization of tokens.
        This method is now fully vectorized for GPU efficiency.

        Parameters
        ----------
        prompts : List[str]
            The prompt text to start generation.
        max_gen_len : int, optional
            Maximum number of tokens to generate (default is 30).
        length : int, optional
            Number of real tokens to generate (kept for compatibility).
        temperature : float, optional
            Sampling temperature (default is 0.8).
        top_p : float, optional
            Top-p sampling parameter (default is 0.95).
        return_debug : bool, optional
            If True, additional debug information is returned.

        Returns
        -------
        List[BinarizedText]
            A list of BinarizedText objects containing the generated text,
            token IDs, and optionally P1 values and random draws.
    """
        bsz = len(prompts) # batch size 
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).to(device).long()
            # Initialize attention mask as a PyTorch tensor
        attn_mask = torch.zeros((bsz, total_len), dtype=torch.bool, device=device)
        for k, t in enumerate(prompt_tokens):
            attn_mask[k, : len(t)] = 1  # Mark prompt tokens as valid
        input_text_mask = tokens != self.pad_id

        # Initialize lists to store output information
        generated_texts = []
        all_token_ids = []
        P1_values = [[] for _ in range(bsz)] if return_debug else None
        random_values_all = [[] for _ in range(bsz)] if return_debug else None

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos],
                use_cache=True,
                past_key_values=outputs.past_key_values if prev_pos > 0 else None,
                attention_mask=attn_mask[:, :cur_pos]  # Apply updated attention mask
            )
            next_toks, P1_batch, rand_vals_batch  = self.sample_next(outputs.logits[:, -1, :], 
                                        temperature,
                                        top_p,
                                        return_debug)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            # Update attention mask for newly generated tokens
            attn_mask[:, cur_pos] = 1  # Mark new tokens as valid for attention
            prev_pos = cur_pos

            # Store debug values if requested
            if return_debug:
                for i in range(bsz):
                    P1_values[i].extend(P1_batch[i])
                    random_values_all[i].extend(rand_vals_batch[i])

        # Vectorized processing to cut sequences at the EOS token for all batch elements
        total_len = tokens.size(1)                  # total token length per sample
        bsz = tokens.size(0)
        # Create indices for each position (shape: [bsz, total_len])
        indices = torch.arange(total_len, device=tokens.device).unsqueeze(0).expand(bsz, total_len)
        # Create an EOS mask (True where token equals the EOS token)
        eos_mask = (tokens == self.eos_id)
        # For each sample, replace indices where EOS is not found with total_len and take the min;
        # if no EOS token is found in a sample, this will return total_len.
        masked_indices = torch.where(eos_mask, indices, torch.full_like(indices, total_len))
        eos_idx = masked_indices.min(dim=1).values         # shape: (bsz,)
        # Create a mask: for each sample, positions less than its eos_idx are valid
        batch_range = torch.arange(total_len, device=tokens.device).unsqueeze(0).expand(bsz, total_len)
        valid_mask = batch_range < eos_idx.unsqueeze(1)
        # Clone the tokens and replace tokens after EOS with the pad ID
        processed_tokens = tokens.clone()
        processed_tokens[~valid_mask] = self.pad_id
        
        # Use batch decoding on the processed tokens; skip special tokens so pad IDs are removed
        generated_texts = self.tokenizer.batch_decode(processed_tokens.tolist(), skip_special_tokens=True)
        # Also update all_token_ids with the processed tokens (as lists)
        all_token_ids = processed_tokens.tolist()

        # Create and return BinarizedText objects
        results = []
        for i in range(bsz):
            results.append(BinarizedText(
                prompt = prompts[i],
                text=generated_texts[i],
                token_ids=all_token_ids[i],
                P1=P1_values[i] if return_debug else None,
                random_values=random_values_all[i] if return_debug else None,
                watermarked=False
            ))

        return results
        

    @torch.no_grad()
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        return_debug: bool = False  # Optional: Return P1 and random_values_all
    ) -> tuple:
        """ Vanilla sampling with temperature and top p.
        
        Sampling the next token using binary random selection with optional debug tracking.
        
        Args:
            logits: (bsz, vocab_size) - Logits for the last token in each batch item.
            ngram_tokens: (bsz, ngram) - Context tokens used in seeding.
            temperature: Controls randomness in sampling.
            top_p: Nucleus sampling threshold.
            return_debug: If True, returns P1 and random_values_all; otherwise, only returns tokens.

        
        Returns:
            If `return_debug=False`: Returns `next_tokens` (bsz,)
            If `return_debug=True`: Returns tuple `(next_tokens, prob_bit_1, random_values, entropy, ementropy)`
        """
        bsz, vocab_size = logits.shape  # Get batch size and vocab size
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # Restore to original probability order
            probs_original_order = torch.zeros_like(probs)
            probs_original_order.scatter_(-1, probs_idx, probs_sort)
            
            # Vectorized over the batch: maintain a token_ids vector for all samples
            token_ids = torch.zeros(bsz, dtype=torch.long, device=self.model.device)
            if return_debug:
                prob_bit_1_list = []
                random_val_list = []
            for bit_index in range(self.blen):
                # _binarize_next now supports a 2D tensor: p0,p1 are computed for all samples simultaneously.
                p0, p1 = self._binarize_next(probs_original_order, bit_index, self.blen, token_ids)
                total = p0 + p1
                # Avoid division by zero and compute probability for bit=1 in a vectorized manner.
                prob_bit_1 = torch.where(total > 0, p1 / total, torch.zeros_like(total))
                rand_vals = torch.rand(bsz, generator = self.evolving_rng, device = self.model.device)
                if return_debug:
                    prob_bit_1_list.append(prob_bit_1.unsqueeze(1))
                    random_val_list.append(rand_vals.unsqueeze(1))
                # Shift token_ids left and add the new bit (1 if rand < prob_bit_1, else 0)
                token_ids = (token_ids << 1) + (rand_vals < prob_bit_1).long()
            next_tokens = token_ids
        else:
            next_tokens = torch.argmax(logits, dim=-1)
        next_tokens = next_tokens.reshape(-1)
        if return_debug:
            # After the loop over bits, concatenate debug lists to form 2D tensors of shape (bsz, blen)
            rand_val_debug_tensor = torch.cat(random_val_list, dim=1)
            prob_bit_1_debug_tensor = torch.cat(prob_bit_1_list, dim=1)  
            entropy = entropyfnc(probs_original_order)  
            ementropy = -torch.log(probs_original_order[torch.arange(bsz, device=token_ids.device), token_ids] + 1e-15)      
            return next_tokens, prob_bit_1_debug_tensor, rand_val_debug_tensor, entropy, ementropy
        return next_tokens
     
    

###############################################
# GENERAL WATERMARK BASE
###############################################
class Watermark:
    """
    A general (abstract) base class for watermarking. 
    Does NOT rely on binarized logic.
    """
    @torch.no_grad()
    def __init__(self, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
            pad_id: int = -1,
            device = device
        ):
        
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.manual_rng_seed = seed
        self.hashtable = torch.randperm(1000003, device = device)
        self.seeding = seeding 
        self.manual_rng = torch.Generator(device = device)
        self.manual_rng.manual_seed(self.manual_rng_seed)
        self.payload = payload
        self.pad_id = pad_id

    @torch.no_grad()
    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    @torch.no_grad()
    def get_seed_rng(self, input_ids: torch.LongTensor) -> Union[int, List[int]]:
        """
        Seed RNG with hash of input_ids.
        Supports both 1D and 2D input_ids using vectorized operations for GPU efficiency.
        For 2D input_ids (batch mode), only valid tokens (non-padded) are considered.
        Returns an int for 1D input, or a list of ints for 2D input.
        """
        M = 2**64 - 1
        device = input_ids.device
        if input_ids.dim() == 1:
            # Unvectorized, single sequence case
            if self.seeding == 'hash':
                manual_rng_seed = self.manual_rng_seed
                for i in input_ids:
                    manual_rng_seed = (manual_rng_seed * self.salt_key + i.item()) % M
            elif self.seeding == 'additive':
                manual_rng_seed = self.salt_key * torch.sum(input_ids).item()
                manual_rng_seed = self.hashint(torch.tensor(manual_rng_seed, dtype=torch.long, device=device))
                manual_rng_seed = manual_rng_seed.item() if isinstance(manual_rng_seed, torch.Tensor) else manual_rng_seed
            elif self.seeding == 'skip':
                manual_rng_seed = self.salt_key * input_ids[0].item()
                manual_rng_seed = self.hashint(torch.tensor(manual_rng_seed, dtype=torch.long, device=device))
                manual_rng_seed = manual_rng_seed.item() if isinstance(manual_rng_seed, torch.Tensor) else manual_rng_seed
            elif self.seeding == 'min':
                manual_rng_seed = self.hashint(self.salt_key * input_ids)
                manual_rng_seed = torch.min(manual_rng_seed).item()
            else:
                raise ValueError(f"Unknown seeding method: {self.seeding}")
            return manual_rng_seed
        elif input_ids.dim() == 2:
            batch_size, seq_len = input_ids.size()
            if hasattr(self, 'pad_id'):
                valid_mask = input_ids != self.pad_id
                lengths = valid_mask.sum(dim=1)  # shape: (batch,)
            else:
                valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
                lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
            # print(f"lengths = {lengths}")
            if self.seeding == 'hash':
                # Fall back to Python loop for each row (slower but correct)
                seeds = []
                for row in input_ids:
                    # extract only real tokens (drop pad_id)
                    valid_row = row[row != self.pad_id]
                    # call 1D path
                    seeds.append(self.get_seed_rng(valid_row))
                return seeds
            elif self.seeding == 'additive':
                # Sum valid tokens for each row
                row_sum = (input_ids * valid_mask.long()).sum(dim=1)
                seed_tensor = self.salt_key * row_sum
                seed = self.hashint(seed_tensor)
            elif self.seeding == 'skip':
                # For batch mode skip, use the first token of each row (always non-pad) to seed
                tokens_first = input_ids[:, 0].long()
                seed = self.hashint(self.salt_key * tokens_first)
            elif self.seeding == 'min':
                # Fall back to Python loop for each row (slower but correct)
                seeds = []
                for row in input_ids:
                    valid_row = row[row != self.pad_id]
                    seeds.append(self.get_seed_rng(valid_row))
                return seeds
            else:
                raise ValueError(f"Unknown seeding method: {self.seeding}")
            return seed.tolist()
        else:
            raise ValueError("input_ids must be 1D or 2D")
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError("generate() must be implemented by subclasses.")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("decode() must be implemented by subclasses.") 
    
###############################################
# BINARIZED WATERMARK (MULTI-INHERITANCE)
###############################################

class BinarizedWatermark(Watermark, BinarizedLLM):
    """
    A watermark class that relies on binarization of the language model.
    Inherits:
     - Watermark: for general watermark interface
     - BinarizedLLM: for shared binarized token generation logic
    """
    @torch.no_grad()
    def __init__(self, model, tokenizer, seed= 0, key= 35317):
        # Initialize both parents
        Watermark.__init__(self, salt_key= key, seed= seed, device = model.device)
        BinarizedLLM.__init__(self, model, tokenizer, seed= seed)

    @torch.no_grad()
    def generate(self, prompt, length=30, embedded_message=None, decoded_message=None):
        raise NotImplementedError("generate() must be implemented by subclasses.")

    @torch.no_grad()
    def decode(self, watermarked_text, *args, **kwargs):
       raise NotImplementedError("decode() must be implemented by subclasses.") 
    
###############################################
# CHRIST WATERMARK (SUBCLASS OF BINARIZEDWATERMARK)
###############################################  

class ChristWatermark(BinarizedWatermark):
    """
    Implements the 'Christ' method on top of a binarized watermark.
    """   
    @torch.no_grad()
    def __init__(self, model, tokenizer, key= 0, seed= 0):
        """
        key : any (typically str or bytes)
                Key used for pseudo-random generation in watermarking.
        """
        super().__init__(model, tokenizer, key= key, seed= seed)

    @torch.no_grad()
    def convert_R_vectorized( R: torch.Tensor, blen: int):
        """
        Vectorized conversion of the binary portion of R.
        
        Each row in R is assumed to follow:
        [prefix (elements not equal to 0 or 1), followed by binary digits (0 or 1),
        then padding (-1)].
        
        For each row:
        • The valid (non -1) portion is determined.
        • The prefix is defined as elements before the first occurrence of 0 or 1.
        • The remaining binary part is split into complete groups of length `blen`; each complete
            group is interpreted (as bits) into an integer.
        • Leftover binary digits (if fewer than blen) are left unchanged.
        • The new row is reconstructed as: prefix + (converted integer groups) + (leftover),
            then padded with -1 to length L.
        
        Parameters:
        R (torch.Tensor): A 2D tensor of shape [bsz, L] with pad value -1.
        blen (int): The full group size (number of binary digits per token).
        tkn_index, bit_index: Unused (kept for signature compatibility).
        
        Returns:
        torch.Tensor: A new tensor of shape [bsz, L] with binary groups replaced by their integer values.
        """
        bsz, L = R.shape
        device = R.device

        # Compute valid length per row (positions not equal to -1)
        valid_mask = (R != -1)
        valid_len = valid_mask.sum(dim=1)  # shape [bsz]

        # Create column indices for each row
        idx = torch.arange(L, device=device).unsqueeze(0).expand(bsz, L)
        
        # Identify positions that are binary (0 or 1) in the valid area.
        binary_mask = (((R == 0) | (R == 1)) & valid_mask)
        # For rows with no binary digit, set index to L
        temp = torch.where(binary_mask, idx, L * torch.ones_like(idx))
        first_bin, _ = temp.min(dim=1)  # shape [bsz]
        # Prefix length: first binary index or, if none, the entire valid length.
        prefix_len = torch.where(first_bin < valid_len, first_bin, valid_len)
        
        # Compute binary segment length per row.
        binary_length = valid_len - prefix_len  # shape [bsz]
        max_bin_len = binary_length.max().item()

        # Prepare a tensor to hold the binary part for all rows, padded with -1.
        # For each row, binary part is R[i, prefix_len[i]: valid_len[i]].
        binary_data = torch.full((bsz, max_bin_len), -1, dtype=R.dtype, device=device)
        # Build indices for each row for the binary part.
        # For each row i, desired column indices = prefix_len[i] + [0, 1, ..., max_bin_len-1]
        start = prefix_len.unsqueeze(1)  # shape [bsz, 1]
        arange_bin = torch.arange(max_bin_len, device=device).unsqueeze(0).expand(bsz, max_bin_len)
        desired_idx = start + arange_bin  # shape [bsz, max_bin_len]
        # Mask: only take positions within binary_length.
        valid_positions = (arange_bin < binary_length.unsqueeze(1))
        # Gather using advanced indexing.
        binary_data = torch.where(valid_positions, R.gather(1, desired_idx), torch.full_like(desired_idx, -1))
        
        # For each row, determine the number of complete groups and leftover bits.
        full_groups = (binary_length // blen)  # shape [bsz] of ints
        leftover = binary_length % blen           # shape [bsz]
        max_groups = full_groups.max().item()

        # Process full binary groups. We'll create a tensor of shape [bsz, max_groups*blen]
        if max_groups > 0:
            full_grp_data = torch.full((bsz, max_groups * blen), -1, dtype=R.dtype, device=device)
            for i in range(bsz):
                cnt = full_groups[i].item() * blen
                if cnt > 0:
                    full_grp_data[i, :cnt] = binary_data[i, :cnt]
            # Reshape to [bsz, max_groups, blen]
            groups_tensor = full_grp_data.view(bsz, max_groups, blen)
            # Create weight vector for binary conversion, e.g. for blen=4: [8,4,2,1]
            weights = 2 ** torch.arange(blen - 1, -1, -1, device=device, dtype=R.dtype)
            weights = weights.unsqueeze(0).unsqueeze(0)  # shape [1,1,blen]
            groups_int = (groups_tensor * weights).sum(dim=-1)  # shape [bsz, max_groups]
        else:
            groups_int = torch.empty((bsz, 0), dtype=R.dtype, device=device)

        # Process leftover binary digits: pad each row to max_leftover.
        max_leftover = leftover.max().item() if leftover.numel() > 0 else 0
        if max_leftover > 0:
            leftover_data = torch.full((bsz, max_leftover), -1, dtype=R.dtype, device=device)
            for i in range(bsz):
                cnt = leftover[i].item()
                if cnt > 0:
                    # Starting index for leftover in binary_data: full_groups[i]*blen
                    start_idx = full_groups[i].item() * blen
                    leftover_data[i, :cnt] = binary_data[i, start_idx:start_idx + cnt]
        else:
            leftover_data = torch.empty((bsz, 0), dtype=R.dtype, device=device)

        # Extract prefix for each row: R[i, :prefix_len[i]]
        max_prefix = prefix_len.max().item()
        prefix_data = torch.full((bsz, max_prefix), -1, dtype=R.dtype, device=device)
        for i in range(bsz):
            cnt = prefix_len[i].item()
            if cnt > 0:
                prefix_data[i, :cnt] = R[i, :cnt]

        # Now concatenate: prefix, then groups_int (converted to integers), then leftover_data.
        # For groups_int, note that each row may have fewer than max_groups groups;
        # they are already padded with -1.
        new_valid = torch.cat([prefix_data, groups_int, leftover_data], dim=1)
        
        # Finally, pad or truncate each row to recover the original length L.
        cur_len = new_valid.size(1)
        if cur_len < L:
            pad = -1 * torch.ones((bsz, L - cur_len), dtype=R.dtype, device=device)
            new_valid = torch.cat([new_valid, pad], dim=1)
        else:
            new_valid = new_valid[:, :L]
        
        return new_valid    

    @torch.no_grad()
    def consistent_perm(self, n):
        """
        This function makes a random permutation of a list 

        Parameters
        ----------
        key : TYPE
            key for PSF.
        n : int
            size of the list.

        Returns
        -------
        perm : list
            permuted list, e.g perm = [5,3,2,4,0,1] when n = 6
        inv_perm : list
            mapping from list(0:n) to their indx in perm. we can build perm using 
            inv_perm, e.g. inv_perm = [4,5,2,1,3,0]

        """
        perm = list(range(n))
        perm = torch.randperm(n, generator=self.evolving_rng, device = self.model.device).tolist()
        inv_perm = [0 for _ in range(n)]
        for i in range(n):
            inv_perm[perm[i]] = i
        return perm, inv_perm
        
    @staticmethod
    def apply_perm(vector, perm):
        """
        This function applies a permutation to the input vector.
        It supports a 1D list, a 1D tensor, or a 2D tensor. In the case of a 2D tensor, the permutation
        is applied to the last dimension for each row.

        Parameters
        ----------
        vector : vector : list or torch.Tensor
            The input vector (or batch of vectors as a 2D tensor) to permute.
            a list to be permuted according to perm. e.g 
            vector = [0.3,0.2,0.17,0.13,0.1,0.07,0.03]
        perm : list
            A list of indices representing the permutation, e.g., [4,5,0,3,6,1,2].

        Returns
        -------
        result : list
            permuted vetor. e.g result = [0.1,0.07,0.3,0.13,0.03,0.2,0.17]

        """
        # Convert vector to tensor if not already
        if not torch.is_tensor(vector):
            vector = torch.tensor(vector)
        perm_tensor = torch.tensor(perm, device=vector.device, dtype=torch.long)
        if vector.dim() == 1:
            result = torch.empty_like(vector)
            result[perm_tensor] = vector
            return result
        elif vector.dim() == 2:
            result = torch.empty_like(vector)
            result[:, perm_tensor] = vector
            return result
        else:
            raise ValueError("apply_perm supports only 1D or 2D tensors")
    
    @staticmethod
    def normalize_score(score, length):
        return (score - length)/math.sqrt(length)
    
    @torch.no_grad()
    def compute_score_function(self, y, bit):
        """
        This function calculates the score function for a binary bit and a payload value.

        Parameters
        ----------
        y : float
            random draw from a uniform distribution.
        bit : str
            Binary token ('0' or '1').

        Returns
        -------
        float
            Score value computed as -log(v), where v is determined from a pseudo-random draw.
        """


        # Compute v based on the bit
        v = y if bit == '1' else (1 - y)

        return -math.log(v)
    
    @torch.no_grad()
    def generate(
            self, 
            prompts: List[str],
            key,
            max_gen_len: int = 200, 
            Rlambda: float = 5, 
            temperature: float = 1,
            top_p: float = 0.95,
            return_debug: bool = False, 
            flagR: bool=False, 
            verbose: bool=False) -> List[BinarizedWatermarkedText]:
            
            """
            Watermarked response generation using "Christ" method 
            (adapted from the original generate_watermarked_response_Christ function).


            TODO:
            - The batch geberation need to be updated, because of different length of prompts, R for each sample should start at different points.

            Parameters
            ----------
            
            prompts : List[str]
                The prompt text to start generation.
            length : int, optional
                Number of real tokens to generate.
            Rlambda : float, optional
                Threshold for switching from random sampling to PRF-based sampling.
            flagR : bool, optional
                If True, we allow collection of bits in R until H >= Rlambda.    
            verbose : bool, optional
                If True, prints debug info to the console.
            return_debug : bool, optional
                If True, also returns P1 values and random draws.

            Returns
            -------
            BinarizedWatermarkedText
                A specialized watermarked text object.
            """
            self.salt_key = key
            # Initialize trackers
            bsz = len(prompts) # batch size 
            flagRchosen = [False] * bsz
            H = torch.zeros(bsz, device=self.model.device, dtype=torch.float)
            n = torch.zeros(bsz, device=self.model.device, dtype=torch.long)  
            if bsz > 1:
                entropy_list = None 
                empentropy_list = None 
            else:
                entropy_list = []
                empentropy_list = []
            all_token_ids = [] if return_debug else None
            generated_texts = []
            
            # Tokenize prompt
            prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
            min_prompt_size = min([len(t) for t in prompt_tokens])
            max_prompt_size = max([len(t) for t in prompt_tokens])
            total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

            watermarked_btkns_list =torch.full((bsz, total_len * self.blen), self.pad_id, dtype=torch.long, device=self.model.device) if return_debug else None            
            watermarked_btkns_list_indx = torch.full((bsz, total_len * self.blen), self.pad_id, dtype=torch.long, device=self.model.device) if return_debug else None
            tokens = torch.full((bsz, total_len), self.pad_id).to(self.model.device).long()
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t).to(self.model.device).long()
                # Initialize attention mask as a PyTorch tensor
            attn_mask = torch.zeros((bsz, total_len), dtype=torch.bool, device=self.model.device)
            for k, t in enumerate(prompt_tokens):
                attn_mask[k, : len(t)] = 1  # Mark prompt tokens as valid
            input_text_mask = tokens != self.pad_id

            # We'll use the BinarizedLLM's own binarization attributes
            blen = self.blen  # number of bits to represent a token

            start_pos = min_prompt_size
            prev_pos = 0
            R = torch.full((bsz, total_len * blen), self.pad_id, dtype=torch.long, device=self.model.device)
            cache = DynamicCache()
            for cur_pos in range(start_pos, total_len):
                outputs = self.model.forward(
                    tokens[:, prev_pos:cur_pos],
                    use_cache=True,
                    past_key_values= cache if prev_pos > 0 else None,
                    attention_mask=attn_mask[:, :cur_pos]  # Apply updated attention mask
                )
                if return_debug:
                    next_toks,  R, H, n, flagRchosen, watermarked_btkns, watermarked_btkn_indx, rand_vals, P1, entropies, ementropies  = self.sample_next(outputs.logits[:, -1, :], 
                                                temperature,
                                                top_p,
                                                flagR,
                                                cur_pos - start_pos,
                                                R,
                                                H,
                                                n,
                                                Rlambda,
                                                flagRchosen,
                                                return_debug = return_debug,
                                                verbose = verbose)
                else:
                    next_toks,  R, H, n, flagRchosen, entropies, ementropies  = self.sample_next(outputs.logits[:, -1, :], 
                                                temperature,
                                                top_p,
                                                flagR,
                                                cur_pos - start_pos,
                                                R,
                                                H,
                                                n,
                                                Rlambda,
                                                flagRchosen,
                                                return_debug = return_debug,
                                                verbose = verbose) 
                tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
                # Update attention mask for newly generated tokens
                attn_mask[:, cur_pos] = 1  # Mark new tokens as valid for attention
                prev_pos = cur_pos

                if bsz > 1:
                    if entropy_list is not None:
                        entropy_list = torch.cat((entropy_list, entropies.unsqueeze(1)), dim = 1)
                        empentropy_list = torch.cat((empentropy_list, ementropies.unsqueeze(1)), dim = 1)
                    else:
                        entropy_list = entropies.unsqueeze(1)
                        empentropy_list = ementropies.unsqueeze(1)
                else:
                    entropy_list.append(entropies.item())
                    empentropy_list.append(ementropies.item())
                # Store debug values if requested
                if return_debug:
                    if cur_pos == start_pos:
                        P1_list = P1
                        rand_vals_list = rand_vals
                    else:    
                        P1_list = torch.cat((P1_list,P1) , dim = 1 )
                        rand_vals_list = torch.cat((rand_vals_list,rand_vals), dim =1 )
                    # Vectorized update of watermarked_btken_list using watermarked_btken_new from sample_next (only add indexes not equal to self.pad_id)
                    current_valid_counts = (watermarked_btkns_list != self.pad_id).sum(dim=1)  # shape (bsz,)
                    col_offset = torch.arange(self.blen, device=watermarked_btkns_list.device).unsqueeze(0)  # shape (1, blen)
                    col_indices = current_valid_counts.unsqueeze(1) + col_offset  # shape (bsz, blen)
                    batch_indices = torch.arange(bsz, device=watermarked_btkns_list.device).unsqueeze(1).expand_as(col_indices)
                    mask = watermarked_btkn_indx != self.pad_id
                    watermarked_btkns_list[batch_indices[mask], col_indices[mask]] = watermarked_btkns[mask]
                    watermarked_btkns_list_indx[batch_indices[mask], col_indices[mask]] = watermarked_btkn_indx[mask]
                           
            # Convert tokens to text
            # Vectorized processing to cut sequences at the EOS token for all batch elements
            # Create indices for each position (shape: [bsz, total_len])
            indices = torch.arange(total_len, device=tokens.device).unsqueeze(0).expand(bsz, total_len)
            # Create an EOS mask (True where token equals the EOS token)
            eos_mask = (tokens == self.eos_id)
            # For each sample, replace indices where EOS is not found with total_len and take the min;
            # if no EOS token is found in a sample, this will return total_len.
            masked_indices = torch.where(eos_mask, indices, torch.full_like(indices, total_len))
            eos_idx = masked_indices.min(dim=1).values         # shape: (bsz,)
            # Create a mask: for each sample, positions less than its eos_idx are valid
            batch_range = torch.arange(total_len, device=tokens.device).unsqueeze(0).expand(bsz, total_len)
            valid_mask = batch_range < eos_idx.unsqueeze(1)
            # Clone the tokens and replace tokens after EOS with the pad ID
            processed_tokens = tokens.clone()
            processed_tokens[~valid_mask] = self.pad_id
            # Remove prompt tokens from the final sequences
            prompt_lens = [len(t) for t in prompt_tokens]  # list of prompt lengths per sample
            # For each sample, slice tokens from the prompt length onward
            final_token_ids = [processed_tokens[i, prompt_lens[i]:prompt_lens[i]+ max_gen_len][processed_tokens[i, prompt_lens[i]:prompt_lens[i]+ max_gen_len] != self.pad_id].tolist() for i in range(bsz)]
            # Batch decode the final tokens (excluding prompt tokens)
            generated_texts = self.tokenizer.batch_decode(final_token_ids, skip_special_tokens=True)
            
            # Build final WatermarkedText
            if bsz > 1:
                # Calculate mean along the second dimension (axis=1)
                avg_entropy = torch.mean(entropy_list, dim=1).tolist()
                avg_emp_entropy = torch.mean(empentropy_list, dim=1).tolist()
            else:
                avg_entropy = statistics.mean(entropy_list) 
                avg_emp_entropy = statistics.mean(empentropy_list)

            return [BinarizedWatermarkedText(
                prompt=prompts[i],
                text=generated_texts[i],
                watermarked=True,
                token_ids=final_token_ids[i],
                watermarked_btkns = watermarked_btkns_list[i][watermarked_btkns_list[i] != self.pad_id][:max_gen_len* self.blen].tolist() if return_debug else None,
                watermarked_btkns_indx = watermarked_btkns_list_indx[i][watermarked_btkns_list_indx[i] != self.pad_id][:max_gen_len* self.blen].tolist() if return_debug else None,
                P1=P1_list[i][:max_gen_len* self.blen].tolist() if return_debug else None,
                random_values=rand_vals_list[i][:max_gen_len* self.blen].tolist() if return_debug else None,
                R=R[i][R[i] != self.pad_id].tolist(),  # cleaned R bits without pad_id
                entropies=entropy_list[i][:max_gen_len] if bsz > 1 else entropy_list,
                empirical_entropies=empentropy_list[i][:max_gen_len] if bsz > 1 else empentropy_list,
                avg_entropy=avg_entropy[i] if bsz > 1 else avg_entropy,
                avg_emp_entropy=avg_emp_entropy[i] if bsz > 1 else avg_emp_entropy,
                n= n[i],
                generation_key = self.salt_key
            ) for i in range(bsz)]
    
    @torch.no_grad()
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
        flagR: bool = False,
        tkn_index: int = 0,
        R: torch.Tensor = None,                 # 2D tensor holding generated bits; pad positions are filled with self.pad_id
        H: torch.Tensor = None,                 # Tensor of shape (bsz,) as entropy accumulator
        n: torch.Tensor = None,                 # Tensor of shape (bsz,) as counter for each sample
        Rlambda: float = 5,
        flagRchosen: Optional[List[bool]] = None,
        return_debug: bool = False,           # Optional: Return P1 and random_values_all
        verbose: bool = False
    ) -> tuple:
        """ Vanilla sampling with temperature and top p.
        
        Sampling the next token using binary random selection while updating the bit sequence R, 
        entropy accumulator H, and counter n. For each batch element:
          - If flagR is True, we select the next bit based on random numbers.
          - For elements where flagRchosen is False, if (combined_rand_vals < prob_bit_1), we update H by subtracting log(prob_bit_1+1e-15) and extend R with a 1; otherwise, subtract log(1 - prob_bit_1+1e-15) and extend R with a 0.
          - Additionally, for these elements we increment n by 1, and if H[i] >= Rlambda, we set flagRchosen for that element.
        
        Args:
            logits: The logits tensor (bsz, vocab_size) for the last token.
            ngram_tokens: The context tokens used for seeding.
            temperature: Controls sampling randomness.
            top_p: Nucleus sampling threshold.
            flagR: Whether to use the pseudo-random phase.
            tkn_index: Current token index.
            R: A 2D torch.Tensor of shape (bsz, L) that holds the generated bits (with unused positions filled with self.pad_id).
            H: A tensor of shape (bsz,) holding the entropy accumulator values.
            n: A tensor of shape (bsz,) holding counters for how many bits have been added.
            Rlambda: Entropy threshold parameter.
            key: Additional key parameter.
            flagRchosen: A list of booleans (length bsz) indicating if the pseudo-random phase has been chosen per sample; if None, defaults to all False.
            return_debug: If True, debugging info (e.g. P1 and random values) is returned.
            verbose: Verbosity flag.

        Returns:
            If `return_debug=False`: Returns `next_tokens` (bsz,)
            If `return_debug=True`: Returns tuple `(next_tokens, P1, random_values_all)`
        """
        bsz, vocab_size = logits.shape  # Get batch size and vocab size
        if flagRchosen is None:
            flagRchosen = [False] * bsz
        if H is None:
            H = torch.zeros(bsz, device=logits.device, dtype=torch.float)
        if n is None:
            n = torch.zeros(bsz, device=logits.device, dtype=torch.long)
        watermarked_btkns = torch.full((bsz,self.blen), self.pad_id, dtype=torch.long, device=self.model.device)
        watermarked_btkns_indx = torch.full((bsz,self.blen), self.pad_id, dtype=torch.long, device=self.model.device)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # Restore to original probability order
            probs_original_order = torch.zeros_like(probs)
            probs_original_order.scatter_(-1, probs_idx, probs_sort)

            rand_val_list_debug = []       # each element will be a tensor of shape (bsz,)
            prob_bit_1_list_debug = []     # each element will be a tensor of shape (bsz,)
        
            token_ids = torch.zeros(bsz, dtype=torch.long, device=self.model.device)  # Initialize token ID for this batch as a tensor
            for bit_index in range(self.blen):
                # _binarize_next now supports a 2D tensor: p0,p1 are computed for all samples simultaneously for the current bit.
                p0, p1 = self._binarize_next(probs_original_order, bit_index, self.blen, token_ids)
                total = p0 + p1
                # Avoid division by zero and compute probability for bit=1 in a vectorized manner.
                prob_bit_1 = torch.where(total > 0, p1 / total, torch.zeros_like(total))
                if not flagR: # R is not used
                    seedNR = self.get_seed_rng(torch.tensor([tkn_index, bit_index], dtype=torch.long, device=self.model.device))
                    self.manual_rng.manual_seed(seedNR) 
                    random_val_list_NR = torch.rand(1, generator = self.manual_rng, device = self.model.device) * torch.ones(bsz, dtype=torch.float, device=self.model.device)
                    token_ids = (token_ids << 1) + (random_val_list_NR < prob_bit_1)
                    watermarked_btkns[:,bit_index] = (random_val_list_NR < prob_bit_1)
                    # In non-flagR branch, use random_val_list_NR directly.
                    current_rand = random_val_list_NR
                    current_prob = prob_bit_1
                    watermarked_btkns_indx = torch.ones((bsz,self.blen), dtype=torch.long, device=self.model.device)
                else: # R is used
                    R = self.convert_R_vectorized(R, self.blen, tkn_index, bit_index)
                    random_val_list_RChosen =torch.zeros(bsz, device=self.model.device)
                    # Vectorized extension of R with [tkn_index, bit_index] for the entire batch.
                    # This code assumes R is a tensor of shape (bsz, L) filled with self.pad_id for unused positions.
                    v_count = (R != self.pad_id).sum(dim=1)  # Compute valid count per row, shape: (bsz,)
                    L = R.size(1)  # current number of columns in R
                    # Create a new tensor with 2 extra columns to accommodate the new extension.
                    context = torch.full((bsz, L + 2), self.pad_id, dtype=R.dtype, device=R.device)
                    # For each row, we want to shift elements after the valid portion to the right by 2.
                    # Create a column index matrix for the original R.
                    j = torch.arange(L, device=R.device).unsqueeze(0).expand(bsz, L)
                    # For each row, compute an offset of 2 for j indices that are greater than or equal to the valid count.
                    offset = (j >= v_count.unsqueeze(1)).long() * 2
                    new_positions = j + offset  # New positions for the old values.
                    # Scatter the old values of R into context at the appropriate new positions.
                    context.scatter_(1, new_positions, R)
                    # For each row, insert the extension [tkn_index, bit_index] immediately after the valid elements.
                    rows = torch.arange(R.size(0), device=R.device)
                    context[rows, v_count] = tkn_index
                    context[rows, v_count + 1] = bit_index
                    for i in range(bsz):
                        seedRChosen = self.get_seed_rng(context[i][context[i] != self.pad_id].clone().detach())
                        self.manual_rng.manual_seed(seedRChosen)
                        random_val_list_RChosen[i] = torch.rand(1, generator = self.manual_rng, device = self.model.device)
                    random_val_list_RNotChosen = torch.rand(bsz, generator = self.evolving_rng, device = self.model.device)

                    # Shift token_ids left and add the new bit (1 if rand < prob_bit_1, else 0)
                    flagRchosen_tensor = torch.tensor(flagRchosen, dtype=torch.bool, device=token_ids.device)
                    combined_rand_vals = torch.where(flagRchosen_tensor, random_val_list_RChosen, random_val_list_RNotChosen)
                    # Determine the new bit value (1 if combined_rand_vals < prob_bit_1, else 0)
                    bit_val = (combined_rand_vals < prob_bit_1).long()
                    watermarked_btkns[:,bit_index] = bit_val
                    token_ids = (token_ids << 1) + bit_val
                    
                    # For each batch element where flagRchosen_tensor is False, update H:
                    # If combined_rand_vals[i] < prob_bit_1[i], subtract log(prob_bit_1[i]+1e-15);
                    # otherwise, subtract log(1-prob_bit_1[i]+1e-15)
                    mask = ~flagRchosen_tensor
                    update_vals = torch.where(combined_rand_vals < prob_bit_1, 
                                              torch.log(prob_bit_1 + 1e-15), 
                                              torch.log(1 - prob_bit_1 + 1e-15))
                    H[mask] = H[mask] - update_vals[mask]
                    n[mask] = n[mask] + 1
                    # Also update R for those elements where flagRchosen is false:
                    # Replace the pad value at the next valid position with the new bit value
                    R[rows[mask], v_count[mask]] = bit_val[mask]
                    watermarked_btkns_indx[flagRchosen_tensor, bit_index] = tkn_index * self.blen + bit_index
                    # After updating H, update flagRchosen:
                    update_mask = H >= Rlambda
                    flagRchosen_tensor[update_mask] = True
                    # Convert the updated flagRchosen tensor back to a list if required
                    flagRchosen = flagRchosen_tensor.tolist()
                    if verbose:
                        updated_idx = torch.nonzero(update_mask, as_tuple=False).squeeze()
                        # Ensure updated_idx is always iterable
                        if isinstance(updated_idx, int):
                            updated_idx = [updated_idx]
                        else:
                            updated_idx = updated_idx.tolist()
                        for i in updated_idx:
                            # Convert n[i] and R[i] to appropriate Python types for printing
                            n_val = n[i].item() if hasattr(n[i], 'item') else n[i]
                            R_val = R[i].tolist() if hasattr(R[i], 'tolist') else R[i]
                            print(f"Christ updated for batch {i}: n = {n_val}, R = {R_val}")

                    # For debugging, record current random values and prob_bit_1:
                    current_rand = combined_rand_vals  # shape (bsz,)
                    current_prob = prob_bit_1          # shape (bsz,) 

                # Append debug values (unsqueeze to build a (bsz, 1) tensor)
                rand_val_list_debug.append(current_rand.unsqueeze(1))
                prob_bit_1_list_debug.append(current_prob.unsqueeze(1))           
                                                        
            # After the loop over bits, concatenate debug lists to form 2D tensors of shape (bsz, blen)
            rand_val_debug_tensor = torch.cat(rand_val_list_debug, dim=1)
            prob_bit_1_debug_tensor = torch.cat(prob_bit_1_list_debug, dim=1)    
        else:
            # If temperature is 0, use deterministic argmax
            token_ids = torch.argmax(logits, dim=-1)               
        entropies = entropyfnc(probs_original_order)
        # Calculate empirical "entropy" term
        # (negative log(probability of the chosen token))
        ementropies = -torch.log(probs_original_order[torch.arange(bsz, device=token_ids.device), token_ids] + 1e-15)
        if return_debug:
            return token_ids, R, H, n, flagRchosen, watermarked_btkns, watermarked_btkns_indx, rand_val_debug_tensor, prob_bit_1_debug_tensor, entropies, ementropies
        return token_ids, R, H, n, flagRchosen, entropies, ementropies
    
    @torch.no_grad()
    def decode(
        self, 
        decoding_key, 
        text, 
        Rlambda=5, 
        flagR=False, 
        return_debug=False,
        verbose=False
    ) -> BinarizedWatermarkedText:
        """
        Detects the presence of a Christ watermark in the given text.
        
        Parameters
        ----------
        decoding_key : int
            Key used for pseudo-random generation in watermarking.
        text : BinarizedText or BinarizedWatermarkedText
            The watermarked text object containing the tokenized text.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decoding.
        Rlambda : float, optional
            Rlambda used for detecting a watermark.
        flagR : bool, optional
            If True, use the flagR method for detection.
        return_debug : bool, optional
            If True, return debug information.
        Returns
        -------
        BinarizedWatermarkedText or BinarizedText
            Dependent on the detection outcome, the output is either a watermarked 
            text object or a plain text object.
        """
        self.salt_key = decoding_key
        if not isinstance(text, (BinarizedText, BinarizedWatermarkedText)):
            raise TypeError("text must be an instance of BinarizedText or BinarizedWatermarkedText")
        if not isinstance(text, BinarizedWatermarkedText):
            text = BinarizedWatermarkedText.from_binarized_text(text)
        flag_detected = False
        blen = self.blen
        tokens = text.token_ids
        len_tokens = len(tokens)
        detected_watermarked = {}
        best_score = {}
        best_normalized_score = {}
        if flagR:
            R = []
            R_detected = {}
            scores = {}
            normalized_score = {}
            nstar = {}
            if return_debug:
                tkn_scores = {}
                Y = {}
            # nstar = -1
            # R is assumed at first to have length n* blen + m + 1
            # The minimum length of R is 1 and maximum length of R is len(tokens) * blen - 1
            for n in range(len_tokens):
                if not flag_detected:
                    R = tokens[:n]
                    Rtoken_bits = strbintobin(list("0" * blen + bin(tokens[n])[2:])[-blen:])
                    mend = blen - 1 if n == len(tokens) - 1 else blen
                    
                    for m in range(mend):
                        if not flag_detected:
                            score_tmp = 0
                            R.append(Rtoken_bits[m])
                            print(f"n = {n}, m = {m}, len(R)= {n * blen + m + 1}")
                            for i in range(n * blen + m + 1 ,len(tokens) * blen): # i represents the indx of the last bianry token of the text we consider, the length of the text in binary tokens is i + 1 
                                indtoken = i // blen
                                token_bits = ("0" * blen + bin(tokens[indtoken])[2:])[-blen:]
                                seedR = self.get_seed_rng(torch.tensor(R + [i // blen, i % blen], device = self.model.device).long())
                                self.manual_rng.manual_seed(seedR)
                                y = torch.rand(1, generator= self.manual_rng, device = self.model.device)
                                if return_debug:
                                    if n * blen + m + 1 not in Y:
                                        Y[n * blen + m + 1] = []
                                    Y[n * blen + m + 1].append(y.item())
                                score = self.compute_score_function(
                                    y, token_bits[i % blen]
                                    )
                                score_tmp = score_tmp + score
                                if n * blen + m + 1 not in scores:
                                    scores[n * blen + m + 1] = {}
                                    normalized_score[n * blen + m + 1] = {}
                                if (i+1) % blen == 0:
                                    scores[n * blen + m + 1][(i+1) // blen] = score_tmp
                                    normalized_score[n * blen + m + 1][(i+1) // blen] = self.normalize_score(scores[n * blen + m + 1][(i+1) // blen], i  - n * blen - m)
                                    if normalized_score[n * blen + m + 1][(i+1) // blen] > Rlambda:
                                        flag_detected = True
                                        if (i+1) // blen not in nstar:
                                            nstar[(i+1) // blen] = n * blen + m + 1
                                            detected_watermarked[(i+1) // blen] = True
                                            best_score[(i+1) // blen] = scores[n * blen + m + 1][(i+1) // blen]
                                            best_normalized_score[(i+1) // blen] = normalized_score[n * blen + m + 1][(i+1) // blen]
                                            R_detected[(i+1) // blen] = R
                                            if verbose:
                                                print(f"Christ watermark detected at {(i+1)//blen} tokens with score {best_score[(i+1) // blen]}")
                                                print(f" len(R)= {n * blen + m + 1}, i={(i+1)//blen} tokens, nstar = {nstar[(i+1) // blen]}, normalized_score = {normalized_score[n * blen + m + 1][(i+1) // blen]}")
                                if return_debug:
                                    if n * blen + m + 1 not in tkn_scores:
                                        tkn_scores[n * blen + m + 1] = []
                                    tkn_scores[n * blen + m + 1].append(score)
                
            if return_debug:
                text.random_values_at_decode = Y
                text.tkn_scores = tkn_scores
            text.score = scores
            text.normalized_score = normalized_score
            text.best_score = best_score
            text.best_normalized_score = best_normalized_score
            text.detected_watermarked = detected_watermarked
            text.detection_key = decoding_key
            text.R_detected = R_detected
            text.nstar = nstar
            return text        
            
        else: # flagR = False
            scores = {}
            scores[0] = {}
            normalized_score = {}
            normalized_score[0] = {}
            if return_debug:
                Y = {}
                Y[0] = []
                tkn_scores = {}
                tkn_scores[0] = []
            nstar = {}
            best_score = {}
            best_normalized_score = {}
            detected_watermarked = {}
            score_tmp = 0
            for i in range(len(tokens)):
                token_bits = ("0" * blen + bin(tokens[i])[2:])[-blen:]
                for ind in range(blen):
                    seedNR = self.get_seed_rng(torch.tensor([i , ind], device = self.model.device).long())
                    self.manual_rng.manual_seed(seedNR)
                    y = torch.rand(1, generator= self.manual_rng, device = self.model.device)
                    if return_debug:
                        Y[0].append(y.item())
                    tkn_scores[0].append(self.compute_score_function(
                                y, token_bits[ind]
                            ))
                    score_tmp = score_tmp + tkn_scores[0][-1]  
                
                scores[0][i+1] = score_tmp
                normalized_score[0][i+1] = self.normalize_score(scores[0][i+1], (i +1 ) * blen)
                if verbose:
                    print(f"scores[0][{i+1}] = {scores[0][i+1]}, normalized_score[0][{i+1}] = {normalized_score[0][i+1]}, length = {(i+1) * blen}")
                if normalized_score[0][i+1] > Rlambda:
                    nstar[i+1] = 0
                    best_score[i+1] = scores[0][i+1]
                    best_normalized_score[i+1] = normalized_score[0][i+1]
                    detected_watermarked[i+1] = True
                    if verbose: 
                        print(f"Christ watermark detected at {(i+1)} tokens with score {best_score[(i+1) ]}")
                        print(f" len(R)= {0}, i={(i+1)} tokens, nstar = {nstar[(i+1) ]}, normalized_score = {normalized_score[0][(i+1) ]}")

                
            text.score = scores
            text.normalized_score = normalized_score
            text.best_score = best_score
            text.best_normalized_score = best_normalized_score
            text.detected_watermarked = detected_watermarked
            text.detection_key = decoding_key
            text.nstar = nstar
            if return_debug:
                text.random_values_at_decode = Y
                text.tkn_scores = tkn_scores
            return text
                    
                
                    

############################################################
# CHRIST WATERMARK MULTI-KEY(SUBCLASS OF BINARIZEDWATERMARK)
############################################################      
class ChristWatermarkMultiKey(ChristWatermark):
    """
    Extends ChristWatermark to embed a short m-bit message 
    by choosing one among 2^m different keys.
    """
    @torch.no_grad()
    def __init__(self, model, tokenizer, key =0):
        super().__init__(model, tokenizer, key = key)

    @torch.no_grad()
    def generate(
        self,
        keys,
        payload,
        m_bits,
        prompts: List[str],
        max_gen_len: int = 200, 
        Rlambda=5, 
        temperature: float = 1,
        top_p: float = 0.95,
        return_debug: bool = False, 
        flagR: bool=False, 
        verbose: bool=False
    ) -> BinarizedWatermarkedTextMultiKey:
        """
        Watermarked response generation using multiple keys 
        based on an m-bit message.
        TODO: needs to be extended for batch of prompts.
        TODO: needs to be extended for multiple keys.
        TODO: need to be extended for chunking.
        Parameters
        ----------
        keys : list (or dict) of str
            A collection of distinct keys of length 2^m_bits. 
            E.g., if m_bits=3, then this should have 8 keys.
        payload : int
            The integer in [0, 2^m_bits - 1] representing 
            the message to embed.
        m_bits : int
            The number of bits in the message (so we expect 
            len(keys) == 2^m_bits).
        prompt : str
            The prompt text to start generation.
        length : int, optional
            Number of real tokens to generate.
        Rlambda : float, optional
            Threshold for switching from random sampling to PRF-based sampling.
        flagR : bool, optional
            If True, we allow collection of bits in R until H >= Rlambda.
        flagPerm : bool, optional
            If True, apply a permutation to the probability distribution.
        verbose : bool, optional
            If True, prints debug info to the console.

        Returns
        -------
        BinarizedWatermarkedTextMultiKey
            A specialized watermarked text object.
        """
        text_multikey = BinarizedWatermarkedTextMultiKey()
        # --- Validate input ---
        # Check 2^m_bits == len(keys)
        if len(keys) != 2 ** m_bits:
            raise ValueError(
                f"Expected len(keys) == 2^{m_bits}, but got {len(keys)} instead."
            )
        # Check payload < 2^m_bits
        if not (0 <= payload < 2 ** m_bits):
            raise ValueError(
                f"payload must be in [0, {2**m_bits - 1}], but got {payload}."
            )

        # Select the key based on the message
        chosen_key = keys[payload]

        if verbose:
            print(f"[ChristWatermarkMultiKey] Using key index={payload}, key={chosen_key}")

        # Now call the original generate_christ with the chosen key
        text_multikey.nbits = m_bits
        text_multikey.payload = payload
        text_multikey.encoding_keys = chosen_key
        
        # Initialize the dictionary to store results for each key
        text_multikey.BinarizedWatermarkedText_dict = {}
        
        # Generate text using the chosen key
        text_multikey.BinarizedWatermarkedText_dict[chosen_key] = ChristWatermark.generate(
            self,
            prompts=prompts,
            key=chosen_key,
            max_gen_len=max_gen_len,
            Rlambda=Rlambda,
            temperature=temperature,
            top_p=top_p,
            return_debug=return_debug,
            flagR=flagR,
            verbose=verbose
        )
        
        return text_multikey

        
    @torch.no_grad()
    def decode(
        self, 
        decoding_keys, 
        text, 
        Rlambda=5, 
        flagR=False,
        return_debug=False,
        verbose=False
    ):
        """
        Detects the presence of a Christ watermark in the given text when multiple keys are used.
        
        Parameters
        ----------
        decoding_keys : list
            A collection of distinct keys of length 2^m_bits used for embedding.
        text : BinarizedWatermarkedTextMultiKey
            A specialized watermarked text object
        Rlambda : float, optional
            Threshold for detecting a watermark.
        flagR : bool, optional
            If True, use the flagR method for detection.
        return_debug : bool, optional
            If True, return debug information.
        verbose : bool, optional
            If True, print debug information.
        
        Returns
        -------
        BinarizedWatermarkedText
            A specialized watermarked text object.
        """
        set_detected = []
        scores = []
        max_scores = []
        detected_watermarked = []
        detected_watermarked_keys = []
        detected_message = []

        # Get the first key to determine the number of prompts
        first_key = next(iter(text.BinarizedWatermarkedText_dict))
        num_prompts = len(text.BinarizedWatermarkedText_dict[first_key])
        
        # Iterate over each prompt
        for i in range(num_prompts):
            set_detected.append({})
            scores.append({})
            max_scores.append({})
            detected_watermarked.append({})
            detected_message.append({})
            detected_watermarked_keys.append({})
            # Iterate over each key
            for key in decoding_keys:
                if key not in text.BinarizedWatermarkedText_dict:
                    text.BinarizedWatermarkedText_dict[key] = []
                    text.BinarizedWatermarkedText_dict[key].append(BinarizedWatermarkedText())
                    text.BinarizedWatermarkedText_dict[key][i].token_ids = tokens_ids 
                else:
                    tokens_ids = text.BinarizedWatermarkedText_dict[key][i].token_ids
                text.BinarizedWatermarkedText_dict[key][i] = ChristWatermark.decode(
                    self,
                    decoding_key=key, 
                    text=text.BinarizedWatermarkedText_dict[key][i], 
                    Rlambda=Rlambda, 
                    flagR=flagR,
                    return_debug=return_debug,
                    verbose=verbose
                )
                
                for l_detected in text.BinarizedWatermarkedText_dict[key][i].nstar: # l_detected is the length of text for which the watermark was detedcted 
                    nstar = text.BinarizedWatermarkedText_dict[key][i].nstar[l_detected]
                    normalized_score = text.BinarizedWatermarkedText_dict[key][i].normalized_score[nstar][l_detected]
                    if int(l_detected) not in set_detected[i]:
                        set_detected[i][int(l_detected)]= [(nstar, key, normalized_score)]
                    else:
                        set_detected[i][int(l_detected)].append((nstar, key, normalized_score))
                if verbose:
                    print(f"set_detected[i] = {set_detected[i]}")
        
            for l_detected in set_detected[i]:
                max_scores[i][l_detected] = set_detected[i][l_detected][0]
                for j in range(len(set_detected[i][l_detected])):
                    if set_detected[i][l_detected][j][2] > max_scores[i][l_detected][2]:
                        max_scores[i][l_detected] = set_detected[i][l_detected][j]
                if max_scores[i][l_detected][2] > Rlambda:
                    detected_watermarked[i][l_detected] = True
                    detected_watermarked_keys[i][l_detected] = max_scores[i][l_detected][1]
                    detected_message[i][l_detected] = decoding_keys.index(max_scores[i][l_detected][1])
                else:
                    detected_watermarked[i][l_detected] = False
                    detected_message[i][l_detected] = -1

        text.detected_message = detected_message
        text.decoding_keys = decoding_keys
        text.detected_watermarked = detected_watermarked
        text.detected_watermarked_keys = detected_watermarked_keys
        text.max_scores = max_scores
        text.scores = scores
        return text
        
        
    
###############################################
# DISC WATERMARK (SUBCLASS OF BINARIZEDWATERMARK)
###############################################

class DISC(BinarizedWatermark):
    """
    A specialized class for 'DISC' watermarking or detection,
    inheriting binarized logic from BinarizedWatermark.
    """
    
    @torch.no_grad()
    def __init__(self, model, tokenizer, key= 0, seed= 0):
        """
        key : any (typically str or bytes)
                Key used for pseudo-random generation in watermarking.
        """
        super().__init__(model, tokenizer, key= key, seed= seed)

    @torch.no_grad()
    def watermarked_tkn(self, random_val, prob_bit_1, deltaM = 0, prob_embedding = 1):
        """
        Embeds a watermark into a token based on a random value and a probability.
        Handles both scalar values and 1D tensors.
        
        Parameters
        ----------
        random_val : float or torch.Tensor
            Random value(s) from a uniform distribution.
        prob_bit_1 : float or torch.Tensor
            Probability of binary bit 1.
        deltaM : float, optional
            The shift in the decoder.
        prob_embedding : float, optional
            The probability of embedding the watermark.
            
        Returns
        -------
        torch.Tensor
            A 1D tensor of watermarked tokens (0 or 1).
        """
        # Convert inputs to tensors if they're not already
        if not torch.is_tensor(random_val):
            random_val = torch.tensor([random_val], device=self.model.device)
        elif random_val.dim() == 0:
            random_val = random_val.unsqueeze(0)
        

            
        if not torch.is_tensor(prob_bit_1):
            prob_bit_1 = torch.tensor([prob_bit_1], device=self.model.device)
        elif prob_bit_1.dim() == 0:
            prob_bit_1 = prob_bit_1.unsqueeze(0)
        
        # Initialize result tensor
        watermarked_tkn = torch.zeros_like(random_val, dtype=torch.long)
        
        # Apply watermarking logic elementwise
        if prob_bit_1.numel() == 1 and random_val.numel() > 1:
            # Broadcast single prob_bit_1 to match random_val size
            prob_bit_1 = prob_bit_1.expand_as(random_val)
        
       
        mask_not_embedding = random_val < 1- prob_embedding
        random_not_embedding_bit1 =  mask_not_embedding & (random_val /(1 - prob_embedding) < prob_bit_1)
        watermarked_tkn[random_not_embedding_bit1] = 1
            
        mask_embedding = ~mask_not_embedding
        new_random_val = (random_val - (1-prob_embedding)) / prob_embedding
     
        # Case 1: prob_bit_1 + deltaM < 1
        mask_case1 = (prob_bit_1 + deltaM < 1) & mask_embedding
        # Within case 1, set to 1 where deltaM < random_val < (prob_bit_1 + deltaM)
        mask_case1_bit1 = mask_case1 & (deltaM < new_random_val) & (new_random_val < (prob_bit_1 + deltaM))
        watermarked_tkn[mask_case1_bit1] = 1
            
        # Case 2: prob_bit_1 + deltaM >= 1 (wrap-around logic)
        mask_case2 = (prob_bit_1 + deltaM >= 1) & mask_embedding
        # Within case 2, set to 1 where (random_val < deltaM + prob_bit_1 - 1) or (deltaM < random_val)
        mask_case2_bit1 = mask_case2 & ((new_random_val < deltaM + prob_bit_1 - 1) | (deltaM < new_random_val))
        watermarked_tkn[mask_case2_bit1] = 1
        
        return watermarked_tkn

    
    @torch.no_grad()
    def embed_watermark(self, random_val, prob_bit_1, deltaM):
        """
        Embeds a watermark into a token based on a random value and a probability.
        """
        return (random_val < prob_bit_1).long()

    @torch.no_grad()
    def compute_score_function(self, y, bit, delta, prob_mode=None, prob_embedding= 1):
        """
        This function calculates the score function for a binary bit and a payload value.

        Parameters
        ----------
        y : float
            random draw from a uniform distribution.
        bit : str
            Binary token ('0' or '1').
        delta : float
            The assumed shift in the decoder.
        prob_mode : str, optional
            The probability mode to use (None, 'R', 'random_embedding').
        prob_embedding : float, optional
            The probability of embedding the watermark.
        Returns
        -------
        float
            score value, s(w^b_i, y_i, \delta).
        """
        bit = str(bit)
        if prob_mode != 'random_embedding': # deterministic or R
            if bit == '1':
                if y <= delta:
                    return -math.log(y - delta + 1)
                else:
                    return -math.log(y - delta)
            else:
                if y < delta:
                    return -math.log(delta - y)
                else:
                    return -math.log(delta - y + 1)
        else:
            if y > 1-prob_embedding:
                y = (y - (1-prob_embedding)) / prob_embedding
                if bit == '1':
                    if y <= delta:
                        return -math.log(y - delta + 1)
                    else:
                        return -math.log(y - delta)
                else:
                    if y < delta:
                        return -math.log(delta - y)
                    else:
                        return -math.log(delta - y + 1) 
            else:
                return 0  
                    
            
    @staticmethod    
    def min_score(scores):
        """
        This function returns n* and M* estimated in the decoder.

        Parameters
        ----------
        scores : list or np.ndarray
            A 1D or 2D list/NumPy array of scores.

        Returns
        -------
        int
            n* (row index of the minimum score) or index if 1D.
        int or None
            M* (column index of the minimum score) if 2D.
        float
            The minimum score value.
        """
        scores = np.asarray(scores)  # Convert to NumPy array if not already

        if scores.ndim == 1:  # If it's a 1D list/array
            indmin = np.argmin(scores)  # Get index of the min value
            return int(indmin), scores[indmin]  # No second index in 1D case
        
        elif scores.ndim == 2:  # If it's a 2D list/array
            indmin = np.unravel_index(np.argmin(scores), scores.shape)  # Get row & col indices
            return int(indmin[0]), int(indmin[1]), scores[indmin]

        else:
            raise ValueError("Input scores must be a 1D or 2D list/array.")
        

    
    def generate(
        self, 
        key, 
        prompts: List[str], 
        payload: int = 0, 
        m_bits: int = 5, 
        max_gen_len: int = 200,  
        temperature: float = 1,
        top_p: float = 0.95,
        h: int = 4, 
        Rlambda: float = 5,
        prob_mode: str = None, 
        context_mode: str = 'type2',
        return_debug: bool = False, 
        verbose: bool=False, 
    ):
        """
        Watermarked response generation using the DISC method.
        TODO: needs to be extended for batch of prompts.

        Parameters
        ----------
        key : any
            Key used for pseudo-random generation in watermarking.
        prompt : str
            The initial text prompt to start generation.
        payload: int
            An integer in [0, 2^m_bits - 1] representing the embedded payload.
        m_bits : int
            Number of bits in the payload space.
        length : int
            Number of real tokens to generate.
        Rlambda : float
            Threshold for switching from random sampling to PRF-based sampling.
        prob_mode : str (None, 'R', 'ranom_embedding')
            If 'Deterministic', we use the deterministic DISC approach.
            If 'R', we collect bits in R until H >= Rlambda, then switch approach.
            If 'random_embedding', we embed the watermark randomly.
        context_mode : str
            The context mode to use (type1 or type2).
            type1 uses the last h * log|V| binary tokens,
            type2 uses the last h real tokens + binary index.  
            TODO: Implement type1, for now we only use type2.    
        h : int
            A context window size (in real tokens).
        verbose : bool
            If True, prints debug info.

        Returns
        -------
        BinarizedWatermarkedText
            A specialized watermarked text object.
        """
        # Validate that payload is within [0, 2^m_bits - 1]
        nM = 2**m_bits
        if not (0 <= payload < nM):
            raise ValueError(
                f"payload must be in [0, {nM - 1}], but got {payload}."
            )
        # --------------------------------------------------
        # Compute deltaM from payload
        # --------------------------------------------------
        # 1) Convert integer -> Gray code
        # 2) Scale by 1 / 2^m_bits
        if payload != 0:
            gray_val = GrayCodeConverter.int2gray(payload, m_bits)  # e.g. if payload=5, then gray_val will be 7
            deltaM = gray_val / float(nM)
        else:
            deltaM = 0    

        if verbose:
            print(f"DISC: m_bits={m_bits}, message={payload}, gray={gray_val}, deltaM={deltaM}")
        prob_embedding = 1 - 1/h
        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------
        bsz = len(prompts) # batch size 
        flagRchosen = [False] * bsz
        flagEmbed = [False] * bsz
        H = torch.zeros(bsz, device=self.model.device, dtype=torch.float)
        n = torch.zeros(bsz, device=self.model.device, dtype=torch.long)  
        context = torch.zeros((bsz, h), device=self.model.device, dtype=torch.long)
        if bsz > 1:
            entropy_list = None 
            empentropy_list = None 
        else:
            entropy_list = []
            empentropy_list = []
        all_token_ids = [] if return_debug else None
        generated_texts = []
            
        # Tokenize prompt
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        watermarked_btkns_list =torch.full((bsz, total_len * self.blen), self.pad_id, dtype=torch.long, device=self.model.device) if return_debug else None            
        watermarked_btkns_list_indx = torch.full((bsz, total_len * self.blen), self.pad_id, dtype=torch.long, device=self.model.device) if return_debug else None
        tokens = torch.full((bsz, total_len), self.pad_id).to(self.model.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).to(self.model.device).long()
            # Initialize attention mask as a PyTorch tensor
        attn_mask = torch.zeros((bsz, total_len), dtype=torch.bool, device=self.model.device)
        for k, t in enumerate(prompt_tokens):
            attn_mask[k, : len(t)] = 1  # Mark prompt tokens as valid
        input_text_mask = tokens != self.pad_id

        # We'll use the BinarizedLLM's own binarization attributes
        blen = self.blen  # number of bits to represent a token

        start_pos = min_prompt_size
        prev_pos = 0
        R = torch.full((bsz, total_len * blen), self.pad_id, dtype=torch.long, device=self.model.device)
        cache = DynamicCache()
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos],
                use_cache=True,
                past_key_values= cache if prev_pos > 0 else None,
                attention_mask=attn_mask[:, :cur_pos]  # Apply updated attention mask
            )
            # --------------------------------------------------
            # Initalize context
            # --------------------------------------------------
            for i in range(bsz):
                context[i, :] = torch.zeros(h, dtype=torch.long, device=context.device)
                prompt_len = len(prompt_tokens[i])
                # TODO: This should be implemented in a vectorized way
                if cur_pos >= prompt_len:
                    flagEmbed[i] = True
                    if prompt_len >= h:
                        # If prompt has enough tokens, take the last h
                        context[i, :] = tokens[i, cur_pos - h:cur_pos]
                    else:
                        # If prompt is shorter than h, pad with zeros
                        context[i, :h-prompt_len] = 0
                        context[i, h-prompt_len:] = tokens[i, cur_pos - prompt_len:cur_pos]
                    
            if return_debug:
                next_toks,  R, context, H, n, flagRchosen, watermarked_btkns, watermarked_btkn_indx, rand_vals, P1, entropies, ementropies  = self.sample_next(outputs.logits[:, -1, :], 
                                                temperature,
                                                top_p,
                                                prob_mode,
                                                prob_embedding,
                                                deltaM,
                                                cur_pos - start_pos,
                                                R,
                                                context,
                                                H,
                                                n,
                                                Rlambda,
                                                flagRchosen,
                                                flagEmbed,
                                                return_debug = return_debug,
                                                verbose = verbose)
            else:
                next_toks,  R, context, H, n, flagRchosen, entropies, ementropies  = self.sample_next(outputs.logits[:, -1, :], 
                                                temperature,
                                                top_p,
                                                prob_mode,
                                                prob_embedding,
                                                deltaM,
                                                cur_pos - start_pos,
                                                R,
                                                context,
                                                H,
                                                n,
                                                Rlambda,
                                                flagRchosen,
                                                flagEmbed,
                                                return_debug = return_debug,
                                                verbose = verbose) 
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            # Update attention mask for newly generated tokens
            attn_mask[:, cur_pos] = 1  # Mark new tokens as valid for attention

        flagRchosen = False
        H = 0.0
        n = 0
        R = []                # Bits stored until threshold is reached
        Y = []                # Random draws from PRF
        P1vec = []            # Probability of bit=1 at each step
        entropy_list = []     # Shannon entropies for each next-token distribution
        empentropy = []       # Empirical negative log-likelihood
        current_probability = [] # Probability of the chosen token

        
        
        # Convert the prompt into tokens
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len_tkn = prompt_ids.shape[1]

        # For certain models, we track attention
        attn = torch.ones_like(prompt_ids)

        # We'll use the parent's binarization attributes
        blen = self.blen

        
        past = None

        
        # --------------------------------------------------
        # Generation Loop
        # --------------------------------------------------
        for i in range(length):
            with torch.no_grad():
                if past is not None:
                    output = self.model(
                        prompt_ids[:, -1:], 
                        past_key_values=past, 
                        attention_mask=attn
                    )
                else:
                    output = self.model(prompt_ids)

            # Optionally zero out a specific token's logit
            # (in original code, token_id=29871 is suppressed)
           # output.logits[:, -1, 29871] = -1e20

            # Probability distribution over next token
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :vocab_size], dim=-1
            ).cpu()[0, :]

            # For debugging/logging
            entropy_list.append(entropyfnc(probs.tolist()))

            # Apply permutation to the distribution
            if flagPerm:
                probs = apply_perm(probs, perm)

            # Combine bits to form the next token
            token_id = 0
            current_tkn = []
            for bit_ind in range(blen):
                # Partial prob for bit=0 or 1
                p0, p1 = self._binarize_next(probs, bit_ind, blen, token_id)
                token_id <<= 1
                p0_val = p0.item()
                p1_val = p1.item()
                P1 = p1_val / (p0_val + p1_val) if (p0_val + p1_val) > 0 else 0.0
                P1vec.append(P1)
                if prob_mode == 'R':
                    if (not flagRchosen):
                        # Random sampling until threshold
                        if random.random() < P1:
                            token_id += 1
                            H -= math.log(P1)
                        else:
                            H -= math.log(1 - P1)
                        n += 1
                        R.append(token_id & 1)

                        # Check threshold
                        if H >= Rlambda and n >= h * blen + 1:
                            flagRchosen = True
                            if verbose:
                                print(f"DISC: n= {n}, R={R}")

                    else:
                        watermarked_btkns.append(i * blen + bit_ind)
                        # PRF-based approach with deltaM shift
                        y = PRF(key, R + context)
                        # Insert your custom "if P1 + deltaM < 1" logic
                        if P1 + deltaM < 1:
                            if deltaM < y < (P1 + deltaM):
                                token_id += 1
                        else:
                            # Wrap-around logic
                            if (y < deltaM + P1 - 1) or (deltaM < y):
                                token_id += 1
                    Y.append(y)
                elif prob_mode == 'random_embedding':
                    # Random embedding
                    y = PRF(key, context)
                    if y< 1/h: # 1/h probability of embedding non-watermarked bit
                        if y * h < P1:
                            token_id += 1
                        else:
                            token_id += 0
                    else: # 1-1/h probability of embedding watermarked bit
                        watermarked_btkns.append(i * blen + bit_ind)
                        y = (y - 1/h) / (1 - 1/h)
                        if P1 + deltaM < 1:
                            if deltaM < y < (P1 + deltaM):
                                token_id += 1
                        else:
                            if (y < deltaM + P1 - 1) or (deltaM < y):
                                token_id += 1            
                    Y.append(y)
                elif prob_mode == 'Deterministic':
                    # If the watermarking is deterministic,
                    watermarked_btkns.append(i * blen + bit_ind)
                    y = PRF(key, context)
                    if P1 + deltaM < 1:
                        if deltaM < y < (P1 + deltaM):
                            token_id += 1
                    else:
                        if (y < deltaM + P1 - 1) or (deltaM < y):
                            token_id += 1
                    Y.append(y)

                # update context
                if context_mode == 'type1':
                    context.pop(0)
                    context.append(token_id & 1)
                elif context_mode == 'type2':
                    current_tkn.append(token_id & 1)
                    context.pop(-1)
                    if bit_ind != blen - 1: 
                        context.append((bit_ind + 1) % blen)   
                    else: 
                        context = context[blen:]
                        context = context.extend(current_tkn)
                        context.append(0)


            # Map back from permuted ID to the real vocabulary ID
            real_token_id = inv_perm[token_id]
            
            # Negative log-likelihood
            empentropy.append(-math.log(probs[real_token_id] + 1e-15))
            current_probability.append(probs[real_token_id] * current_probability[-1] if current_probability else probs[real_token_id])    

            # Add the new token
            token_t = torch.tensor([[real_token_id]], device=self.model.device)
            prompt_ids = torch.cat([prompt_ids, token_t], dim=-1)

            past = output.past_key_values
            attn = torch.cat(
                [attn, attn.new_ones((attn.shape[0], 1))],
                dim=-1
            )

        if verbose:
            wtokens = prompt_ids[0][prompt_len_tkn:].tolist()
            print("Watermarked tokens are:", wtokens)

        # Build final text
        new_token_ids = prompt_ids[0][prompt_len_tkn:].cpu()
        generated_text = self._detokenize(new_token_ids)

        mean_entropy = np.average(entropy_list, weights = current_probability)
        mean_emp_entropy = np.average(empentropy, weights = current_probability)

        return BinarizedWatermarkedText(
                prompt = prompt,
                text=generated_text,
                token_ids=new_token_ids.tolist(),
                P1=P1vec,
                random_values=Y,
                entropies=entropy_list,
                empirical_entropies=empentropy,
                avg_entropy=mean_entropy,
                avg_emp_entropy=mean_emp_entropy,
                n=n,
                generation_key = key, 
                watermarked_btkns=watermarked_btkns
        )
    
    @torch.no_grad()
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        temperature: float = 1, # temperature for sampling
        top_p: float = 1, # top p for sampling
        prob_mode: str = None,
        prob_embedding: float = 1,
        deltaM: float = 0,
        tkn_index: int = 0,
        R: torch.Tensor = None,                 # 2D tensor holding generated bits; pad positions are filled with self.pad_id
        context: torch.Tensor = None,           # 2D tensor holding generated bits; pad positions are filled with self.pad_id
        H: torch.Tensor = None,                 # Tensor of shape (bsz,) as entropy accumulator
        n: torch.Tensor = None,                 # Tensor of shape (bsz,) as counter for each sample
        Rlambda: float = 5,
        flagRchosen: Optional[List[bool]] = None,
        flagEmbed: Optional[List[bool]] = None,
        return_debug: bool = False,           # Optional: Return P1 and random_values_all
        verbose: bool = False
    ) -> tuple:
        """ Vanilla sampling with temperature and top p.
        
        Sampling the next token using binary random selection while updating the bit sequence R, 
        entropy accumulator H, and counter n. For each batch element:
          - If flagR is True, we select the next bit based on random numbers.
          - For elements where flagRchosen is False, if (combined_rand_vals < prob_bit_1), we update H by subtracting log(prob_bit_1+1e-15) and extend R with a 1; otherwise, subtract log(1 - prob_bit_1+1e-15) and extend R with a 0.
          - Additionally, for these elements we increment n by 1, and if H[i] >= Rlambda, we set flagRchosen for that element.
        
        Args:
            logits: The logits tensor (bsz, vocab_size) for the last token.
            ngram_tokens: The context tokens used for seeding.
            temperature: Controls sampling randomness.
            top_p: Nucleus sampling threshold.
            prob_mode: Probability mode for watermarking.
            prob_embedding: Probability of embedding a watermarked bit.
            deltaM: The deltaM parameter for watermarking.
            tkn_index: Current token index.
            R: A 2D torch.Tensor of shape (bsz, L) that holds the generated bits (with unused positions filled with self.pad_id).
            context: A 2D torch.Tensor of shape (bsz, h) that holds the generated tokens.
            H: A tensor of shape (bsz,) holding the entropy accumulator values.
            n: A tensor of shape (bsz,) holding counters for how many bits have been added.
            Rlambda: Entropy threshold parameter.
            flagRchosen: A list of booleans (length bsz) indicating if the pseudo-random phase has been chosen per sample; if None, defaults to all False.
            flagEmbed: A list of booleans (length bsz) indicating if the embedding phase has been chosen per sample; if None, defaults to all False.
            return_debug: If True, debugging info (e.g. P1 and random values) is returned.
            verbose: Verbosity flag.

        Returns:
            If `return_debug=False`: Returns `next_tokens` (bsz,)
            If `return_debug=True`: Returns tuple `(next_tokens, P1, random_values_all)`
        """
        # --------------------------------------------------
        # Build initial context from last h tokens 
        # (decode bit pattern from each token)
        # --------------------------------------------------
        # Convert flagEmbed to tensor
        flagEmbed_tensor = torch.tensor(flagEmbed, dtype=torch.bool, device=self.model.device)
        bsz, vocab_size = logits.shape  # Get batch size and vocab size
        if flagRchosen is None:
            flagRchosen = [False] * bsz
        if H is None:
            H = torch.zeros(bsz, device=logits.device, dtype=torch.float)
        if n is None:
            n = torch.zeros(bsz, device=logits.device, dtype=torch.long)
        entropies = torch.zeros(bsz, device=logits.device, dtype=torch.float)
        ementropies = torch.zeros(bsz, device=logits.device, dtype=torch.float)
        watermarked_btkns = torch.full((bsz,self.blen), self.pad_id, dtype=torch.long, device=self.model.device)
        watermarked_btkns_indx = torch.full((bsz,self.blen), self.pad_id, dtype=torch.long, device=self.model.device)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # Restore to original probability order
            probs_original_order = torch.zeros_like(probs)
            probs_original_order.scatter_(-1, probs_idx, probs_sort)

            rand_val_list_debug = []       # each element will be a tensor of shape (bsz,)
            prob_bit_1_list_debug = []     # each element will be a tensor of shape (bsz,)
        
            token_ids = torch.full((bsz,),self.pad_id, dtype=torch.long, device=self.model.device)  # Initialize token ID for this batch as a tensor
            token_ids[flagEmbed_tensor] = 0
            for bit_index in range(self.blen):
                # _binarize_next now supports a 2D tensor: p0,p1 are computed for all samples simultaneously for the current bit.
                p0, p1 = self._binarize_next(probs_original_order, bit_index, self.blen, token_ids)
                total = p0 + p1
                # Avoid division by zero and compute probability for bit=1 in a vectorized manner.
                prob_bit_1 = torch.where(total > 0, p1 / total, torch.zeros_like(total))
                if prob_mode == 'Deterministic': # R is not used and watermarking is deterministic (no random embedding)
                    # TODO: This should be implemented in a vectorized way
                    random_val_list_NR = torch.zeros(bsz, dtype=torch.float, device=self.model.device)
                    watermarked_btkns_indx = torch.zeros((bsz,self.blen), dtype=torch.long, device=self.model.device)
                    for i in range(bsz):
                        if flagEmbed[i]:
                            seedNR = self.get_seed_rng(torch.cat((context[i], torch.tensor([bit_index], dtype=torch.long, device=self.model.device))))
                            self.manual_rng.manual_seed(seedNR) 
                            random_val_list_NR[i] = torch.rand(1, generator = self.manual_rng, device = self.model.device)  
                            token_ids[i] = (token_ids[i] << 1) + watermarked_btkns[i,bit_index]
                            watermarked_btkns[i,bit_index] = self.watermarked_tkn(random_val_list_NR[i] , prob_bit_1[i], deltaM).item()
                            watermarked_btkns_indx[i,bit_index] = 1
                    # In non-flagR branch, use random_val_list_NR directly.
                    current_rand = random_val_list_NR
                    current_prob = prob_bit_1
                    
                elif prob_mode == 'R': # R is used
                    R = ChristWatermark.convert_R_vectorized(R, self.blen)
                    
                    # Get valid R elements mask and counts for all batches
                    valid_R_mask = R != self.pad_id  # [bsz, R_len]
                    valid_R_counts = valid_R_mask.sum(dim=1)  # [bsz]
                    max_valid_R = valid_R_counts.max().item()
                    
                    # Calculate required context length
                    context_len = context.size(1)
                    new_context_len = max_valid_R + context_len + 1  # +1 for bit_index
                    
                    # Initialize tensors
                    new_context = torch.full((bsz, new_context_len), self.pad_id, dtype=R.dtype, device=R.device)
                    random_val_list_RChosen = torch.zeros(bsz, device=self.model.device)
                    random_val_list_RNotChosen = torch.zeros(bsz, device=self.model.device)
                    
                    # Form context for all batches
                    batch_indices = torch.arange(bsz, device=R.device)
                
                    # Create position indices for each batch
                    batch_arange = torch.arange(new_context_len, device=R.device)
                    # Broadcasting: [bsz, 1] < [1, new_context_len] -> [bsz, new_context_len]
                    valid_positions = batch_arange < valid_R_counts.unsqueeze(1)
                    context_positions = (batch_arange >= valid_R_counts.unsqueeze(1)) & (batch_arange < (valid_R_counts + context_len).unsqueeze(1))
                    bit_positions = batch_arange == (valid_R_counts + context_len).unsqueeze(1)
                    
                    # Copy valid R values using masked scatter
                    new_context[valid_positions] = R[valid_R_mask]
                    
                    # Copy context values - create proper indices for context
                    context_indices = batch_arange.unsqueeze(0).expand(bsz, -1)[context_positions] - valid_R_counts.unsqueeze(1)
                    batch_indices = torch.arange(bsz, device=R.device).unsqueeze(1).expand(-1, new_context_len)[context_positions]
                    new_context[context_positions] = context[batch_indices, context_indices]
                    
                    # Add bit_index where needed
                    new_context[bit_positions] = bit_index
                    
                    # Generate random values only for flagEmbed=True elements
                    if flagEmbed_tensor.any():
                        active_indices = torch.where(flagEmbed_tensor)[0]
                        for idx in active_indices:
                            valid_context = new_context[idx][new_context[idx] != self.pad_id].clone().detach()
                            seedRChosen = self.get_seed_rng(valid_context)
                            self.manual_rng.manual_seed(seedRChosen)
                            random_val_list_RChosen[idx] = torch.rand(1, generator=self.manual_rng, device=self.model.device)
                            random_val_list_RNotChosen[idx] = torch.rand(1, generator=self.evolving_rng, device=self.model.device)
                    
                    # Convert flagRchosen to tensor and compute bit values
                    flagRchosen_tensor = torch.tensor(flagRchosen, dtype=torch.bool, device=token_ids.device)
                    combined_rand_vals = torch.where(flagRchosen_tensor, random_val_list_RChosen, random_val_list_RNotChosen)
                    bit_val = self.watermarked_tkn(combined_rand_vals, prob_bit_1, deltaM).long()
                    
                    # Update watermarked_btkns and token_ids
                    watermarked_btkns[flagEmbed_tensor,bit_index] = bit_val[flagEmbed_tensor]
                    token_ids[flagEmbed_tensor] = (token_ids[flagEmbed_tensor] << 1) + bit_val[flagEmbed_tensor]

                    update_vals = torch.zeros(bsz, device=self.model.device)
                    mask = ~flagRchosen & flagEmbed
                    update_vals = torch.where(bit_val, 
                                              torch.log(prob_bit_1 + 1e-15), 
                                              torch.log(1 - prob_bit_1 + 1e-15))
                    H[mask] = H[mask] - update_vals[mask]
                    n[mask] = n[mask] + 1

                    rows = torch.arange(bsz, device=R.device)                    
                    # Also update R for those elements where flagRchosen is false and flagEmbed is true:
                    # Replace the pad value at the next valid position with the new bit value
                    R[rows[mask], valid_R_counts[mask]] = bit_val[mask]
                    watermarked_btkns_indx[flagRchosen_tensor & flagEmbed, bit_index] = tkn_index * self.blen + bit_index
                    # After updating H, update flagRchosen:
                    update_mask = H >= Rlambda & flagEmbed
                    flagRchosen_tensor[update_mask] = True
                    # Convert the updated flagRchosen tensor back to a list if required
                    flagRchosen = flagRchosen_tensor.tolist()
                    if verbose:
                        updated_idx = torch.nonzero(update_mask, as_tuple=False).squeeze()
                        # Ensure updated_idx is always iterable
                        if isinstance(updated_idx, int):
                            updated_idx = [updated_idx]
                        else:
                            updated_idx = updated_idx.tolist()
                        for i in updated_idx:
                            # Convert n[i] and R[i] to appropriate Python types for printing
                            n_val = n[i].item() if hasattr(n[i], 'item') else n[i]
                            R_val = R[i].tolist() if hasattr(R[i], 'tolist') else R[i]
                            print(f"Christ updated for batch {i}: n = {n_val}, R = {R_val}")

                    # For debugging, record current random values and prob_bit_1:
                    current_rand = combined_rand_vals  # shape (bsz,)
                    current_prob = prob_bit_1          # shape (bsz,) 
                elif prob_mode == 'random_embedding':
                    # Initialize tensors
                    new_context = torch.cat((context, torch.tensor([bit_index], dtype=torch.long, device=self.model.device)), dim=1)
                    random_val_list = torch.zeros(bsz, device=self.model.device)
                    
                    # Generate random values only for flagEmbed=True elements
                    if flagEmbed_tensor.any():
                        active_indices = torch.where(flagEmbed_tensor)[0]
                        for idx in active_indices:
                            seedRChosen = self.get_seed_rng(new_context[idx])
                            self.manual_rng.manual_seed(seedRChosen)
                            random_val_list[idx] = torch.rand(1, generator=self.manual_rng, device=self.model.device)
                            
                    bit_val = self.watermarked_tkn(random_val_list, prob_bit_1, deltaM,prob_embedding).long()
                    
                    # Update watermarked_btkns and token_ids
                    watermarked_btkns[flagEmbed_tensor,bit_index] = bit_val[flagEmbed_tensor]
                    token_ids[flagEmbed_tensor] = (token_ids[flagEmbed_tensor] << 1) + bit_val[flagEmbed_tensor]

                    watermarked_btkns_indx[flagEmbed, bit_index] = tkn_index * self.blen + bit_index

                    # For debugging, record current random values and prob_bit_1:
                    current_rand = random_val_list  # shape (bsz,)
                    current_prob = prob_bit_1          # shape (bsz,) 
                
                # Append debug values (unsqueeze to build a (bsz, 1) tensor)
                rand_val_list_debug.append(current_rand.unsqueeze(1))
                prob_bit_1_list_debug.append(current_prob.unsqueeze(1))           
                                                        
            # After the loop over bits, concatenate debug lists to form 2D tensors of shape (bsz, blen)
            rand_val_debug_tensor = torch.cat(rand_val_list_debug, dim=1)
            prob_bit_1_debug_tensor = torch.cat(prob_bit_1_list_debug, dim=1)    
        else:
            # If temperature is 0, use deterministic argmax
            token_ids = torch.argmax(logits, dim=-1)               
        entropies[flagEmbed_tensor] = entropyfnc(probs_original_order[flagEmbed_tensor])
        # Calculate empirical "entropy" term
        # (negative log(probability of the chosen token))
        ementropies[flagEmbed_tensor] = -torch.log(probs_original_order[flagEmbed_tensor, token_ids[flagEmbed_tensor]] + 1e-15)
        if return_debug:
            return token_ids, R, H, n, flagRchosen, watermarked_btkns, watermarked_btkns_indx, rand_val_debug_tensor, prob_bit_1_debug_tensor, entropies, ementropies
        return token_ids, R, H, n, flagRchosen, entropies, ementropies
    
    def decode(
        self, 
        key, 
        text, 
        nbits, 
        FPR=1e-2, 
        h=4, 
        prob_mode=None, 
        context_mode = 'type2',
        verbose=True, 
        flagPerm = False
    ):
        """
        decode the payload embedded in DISC watermarked text.

        Parameters
        ----------
        key : any
            Key used for the pseudo-random function.
        text : BinarizedText or BinarizedWatermarkedText
            The watermarked text object containing the tokenized text.
        nbits : int
            Number of bits in the payload message.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decodeing the payload.
            Defaults to 1 for LLaMA and 0 for GPT-2.
        FPR : float, optional
            False positive rate threshold.
        prob_mode : str (None, 'R', 'random_embedding')
            If None, we use the deterministic DISC approach.
            If 'R', we collect bits in R until H >= Rlambda, then switch approach.
            If 'random_embedding', we embed the watermark randomly.
        flagPerm : bool
            If True, apply a permutation to the probability distribution.
        context_mode : str
            The context mode to use (type1 or type2). type1 uses the last h * log|V| binary tokens,
            type2 uses the last h real tokens + binary index.        
        h : int, optional
            Context size in tokens.
        flagPerm : bool, optional
            If True, applies a permutation to the probability distribution.    
        verbose : bool, optional
            If True, prints debug info.        

        Returns
        -------
        BinarizedWatermarkedText or BinarizedText
            Dependent on the detection outcome, the output is either a watermarked 
            text object or a plain text object.
        """
        # Adjust `skip_prefix` dynamically for LLaMA vs. GPT-2
        # skip_prefix = 1 for llama 2 as a sentence is tokenized with <s> in this model
        # and skip_prefix = 0 for gpt2 as <s> token does not exist at the beginning of a text in this model
        # in Llama 2 when a watermarked text is generated, the watermarked text starts with a token with (unseeable)"-" at 
        # the beginning, for example, in a watermarked response "Here the three ...", the token for "Here" is actually "-Here" 
        # if not flagTokens:
        #     if "llama" in self.tokenizer.name_or_path:
        #         text = "-" + text.split(" ", 1)[0] + text.split(" ", 1)[1]
        #         skip_prefix = 2  # Adjust for leading <s> token
        if not isinstance(text, (BinarizedText, BinarizedWatermarkedText)):
            raise TypeError("text must be an instance of BinarizedText or BinarizedWatermarkedText")

        # Total number of possible payloads
        nM = 2 ** nbits

        # Get token permutation
        vocab_size = len(self.tokenizer)
        if flagPerm:
            perm, inv_perm = consistent_perm(key, vocab_size)
        else:
            perm = range(vocab_size).tolist()    
            inv_perm = range(vocab_size).tolist()

        # Retrieve `blen` from the parent class
        blen = self.blen
        tokens = text.token_ids
        # # Tokenization if text is not already tokenized
        # if not flagTokens:
        #     tokens = self._tokenize(text)[0][skip_prefix:]
        #     if verbose:
        #         print("Received watermarked text tokens:", tokens.tolist())
        # else:
        #     tokens = text
        #     if verbose:
        #         print("Received watermarked text tokens:", tokens)
        
         
        if prob_mode == 'R': # Collect reference bits if R is being used
            # Initialize score matrices
            total_bits = (len(tokens) - h) * blen
            scores = [[0] * nM for _ in range(total_bits)]
            tkn_scores = [[[] for __ in range(nM)] for _ in range(total_bits)]
            p = [[0] * nM for _ in range(total_bits)]
            Y = [[] for _ in range(total_bits)]
            R = []
            # R is assumed at first to have length n* blen + m + 1
            # The minimum length of R is h * blen + 1 and maximum length of R is len(tokens) * blen - 1
            for n0 in range(h):
                Rtoken_bits = ("0" * blen + bin(perm[tokens[n0]])[2:])[-blen:]
                R.extend(strbintobin(list(Rtoken_bits)))

            for n in range(h, len(tokens)):
                Rtoken_bits = strbintobin(list("0" * blen + bin(perm[tokens[n]])[2:])[-blen:])
                mend = blen - 1 if n == len(tokens) - 1 else blen

                for m in range(mend):
                    contextSet = []
                    context = []
                    R.append(Rtoken_bits[m]) # R is set here with length n* blen + m + 1

                    for i in range(n * blen + m + 1, len(tokens) * blen): # a loop over the tokens to form context and the current token
                    # i // blen represents the indx of the real token
                    # i % blen represent the indx of the binary token in the current real token
                    # i is the index of the curent binary token in the overall text
                        indtoken = i // blen
                        token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                        prv_token_bits = ("0" * blen + bin(perm[tokens[indtoken - 1]])[2:])[-blen:]

                        if context_mode== 'type1':
                            # Initialize context window
                            if not context: # this if is to form the initial contex just once and after that just add one binary token to the context and remove the first binary token
                                context = strbintobin(list("0" * blen + bin(perm[tokens[indtoken - h]])[2:])[-blen:])[i % blen:]
                                for indcontext in range(indtoken - h + 1, indtoken):
                                    context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                                context += strbintobin(list("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:])[:i % blen]
                                assert( len(context) == h * blen)
                            else:
                                context.pop(0)
                                context.append(int(token_bits[i % blen - 1]) if i % blen != 0 else int(prv_token_bits[blen - 1]))
                        elif context_mode == 'type2':
                            if not context:
                                for indcontext in range(indtoken - h + 1, indtoken):
                                    context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                                context.append(int(i % blen))
                            else:
                                context.pop(-1)
                                if i % blen != 0:
                                    context.append(int(i % blen))
                                else:
                                    context = context[blen:]
                                    context = context.extend(token_bits)
                                    context.append(0)
                        y = PRF(key, R + context)
                        Y[(n - h) * blen + m].append(y)

                        if context + [int(token_bits[i % blen])] not in contextSet:
                            contextSet.append(context + [int(token_bits[i % blen])])

                            for j in range(nM):  # Iterate over all delta_j values
                                deltaM = j / nM if nbits > 0 else 0
                                
                                tkn_scores[(n - h) * blen + m][j].append(
                                    self.score(key, R + context, token_bits[i % blen], deltaM)
                                )
                                scores[(n - h) * blen + m][j] += tkn_scores[(n - h) * blen + m][j][-1]

                    for inddelta in range(nM):
                        p[(n - h) * blen + m][inddelta] = special.gammaincc(len(contextSet), scores[(n - h) * blen + m][inddelta])

            # Find best scoring payload
            nstar, Mstar, pstar = self.min_score(p)
            nstar = nstar + blen * h # nstar is the index of the first binary token of the watermark
            if verbose:
                print(f"Detected message: {('0' * nbits + bin(gray2int(Mstar))[2:])[-nbits:]}, nstar={nstar}, Mstar={Mstar}")

            # Validate the decodeed message based on the False Positive Rate (FPR)
            if 1 - (1 - nM * pstar) ** ((len(tokens) - h) * blen) < FPR:
                text.random_values_at_decode = Y
                text.score = scores
                text.tkn_scores = tkn_scores
                text.best_score = scores[nstar - h * blen]
                text.watermarked = True
                text.detection_key = key
                text.p_value = p
                text.best_p_value = pstar
                text.decoded_message = gray2int(Mstar)
                if isinstance(text, BinarizedText):
                    return BinarizedWatermarkedText.from_binarized_text(
                                text,
                                n= nstar, 
                                watermarked_btkns= list(range(nstar + 1, len(tokens) * blen))
                            )
                else:
                    text.n = nstar
                    text.watermarked_btkns= list(range(nstar + 1, len(tokens) * blen))
                    return text
            else:
                text.random_values_at_decode = Y
                text.score = scores
                text.tkn_scores = tkn_scores
                text.best_score = scores[nstar - h * blen]
                text.watermarked = False
                text.detection_key = key
                text.p_value = p
                text.best_p_value = pstar
                if isinstance(text, BinarizedText):
                    return text
                else:
                    return BinarizedText.from_binarized_watermarked_text(text) 
                    
        elif prob_mode == 'random_embedding':
            # Initialize score matrices
            scores = [0] * nM 
            tkn_scores = [[] for _ in range(nM)] 
            p = [0] * nM 
            Y = []
            watermarked_btkns = []
            contextSet = []
            context = []
            for i in range(h * blen + 1, len(tokens) * blen): # a loop over the tokens to form context and the current token
            # i // blen represents the indx of the real token
            # i % blen represent the indx of the binary token in the current real token
            # i is the index of the cuurent binary token in the overall text
                indtoken = i // blen
                token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                prv_token_bits = ("0" * blen + bin(perm[tokens[indtoken - 1]])[2:])[-blen:]

                if context_mode== 'type1':
                    # Initialize context window
                    if not context: # this if is to form the initial contex just once and after that just add one binary token to the context and remove the first binary token
                        context = strbintobin(list("0" * blen + bin(perm[tokens[indtoken - h]])[2:])[-blen:])[i % blen:]
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context += strbintobin(list("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:])[:i % blen]
                    else:
                        context.pop(0)
                        context.append(int(token_bits[i % blen - 1]) if i % blen != 0 else int(prv_token_bits[blen - 1]))
                elif context_mode == 'type2':
                    if not context:
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context.append(int(i % blen))
                    else:
                        context.pop(-1)
                        if i % blen != 0:
                            context.append(int(i % blen))
                        else:
                            context = context[blen:]
                            context = context.extend(token_bits)
                            context.append(0)
                y = PRF(key, context)
                Y.append(y)
                if y> 1/h: # 1/h probability of embedding non-watermarked bit
                    watermarked_btkns.append(i)
                    if context + [int(token_bits[i % blen])] not in contextSet:
                        contextSet.append(context + [int(token_bits[i % blen])])    

                        for j in range(nM):  # Iterate over all delta_j values
                            deltaM = j / nM if nbits > 0 else 0
                            tkn_scores[j].append(
                                        self.score(key, context, token_bits[i % blen], deltaM, prob_mode, h)
                                    )
                            scores[j] += tkn_scores[j][-1]  

            for inddelta in range(nM):
                p[inddelta] = special.gammaincc(len(contextSet), scores[inddelta])

            # Find best scoring payload
            Mstar, pstar = self.min_score(p)

            if verbose:
                print(f"Detected message: {('0' * nbits + bin(gray2int(Mstar))[2:])[-nbits:]}, Mstar={Mstar}")

            # Validate the decodeed message based on the False Positive Rate (FPR)
            if 1 - (1 - nM * pstar) ** ((len(tokens) - h) * blen) < FPR:
                text.random_values_at_decode = Y
                text.score = scores
                text.tkn_scores = tkn_scores
                text.best_score = scores
                text.watermarked = True
                text.detection_key = key
                text.p_value = p
                text.best_p_value = pstar
                text.decoded_message = gray2int(Mstar)
                if isinstance(text, BinarizedText):
                    return BinarizedWatermarkedText.from_binarized_text(
                                text,
                                watermarked_btkns= watermarked_btkns
                            )
                else:
                    text.watermarked_btkns = watermarked_btkns
                    return text
        
        elif prob_mode == None:
            # Initialize score matrices
            scores = [0] * nM 
            tkn_scores = [[] for _ in range(nM)] 
            p = [0] * nM 
            Y = []
            for i in range(h * blen + 1, len(tokens) * blen): # a loop over the tokens to form context and the current token
            # i // blen represents the indx of the real token
            # i % blen represent the indx of the binary token in the current real token
            # i is the index of the cuurent binary token in the overall text
                indtoken = i // blen
                token_bits = ("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:]
                prv_token_bits = ("0" * blen + bin(perm[tokens[indtoken - 1]])[2:])[-blen:]

                if context_mode== 'type1':
                    # Initialize context window
                    if not context: # this if is to form the initial contex just once and after that just add one binary token to the context and remove the first binary token
                        context = strbintobin(list("0" * blen + bin(perm[tokens[indtoken - h]])[2:])[-blen:])[i % blen:]
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context += strbintobin(list("0" * blen + bin(perm[tokens[indtoken]])[2:])[-blen:])[:i % blen]
                    else:
                        context.pop(0)
                        context.append(int(token_bits[i % blen - 1]) if i % blen != 0 else int(prv_token_bits[blen - 1]))
                elif context_mode == 'type2':
                    if not context:
                        for indcontext in range(indtoken - h + 1, indtoken):
                            context += strbintobin(list("0" * blen + bin(perm[tokens[indcontext]])[2:])[-blen:])
                        context.append(int(i % blen))
                    else:
                        context.pop(-1)
                        if i % blen != 0:
                            context.append(int(i % blen))
                        else:
                            context = context[blen:]
                            context = context.extend(token_bits)
                            context.append(0)
                y = PRF(key, context)
                Y.append(y)

                if context + [int(token_bits[i % blen])] not in contextSet:
                    contextSet.append(context + [int(token_bits[i % blen])])

                    for j in range(nM):  # Iterate over all delta_j values
                        deltaM = j / nM if nbits > 0 else 0
                        scores[j] += self.score(
                                    key, context, token_bits[i % blen], deltaM
                        )
                        tkn_scores[j].append(
                                    self.score(key, context, token_bits[i % blen], deltaM)
                                )

            for inddelta in range(nM):
                p[inddelta] = special.gammaincc(len(contextSet), scores[inddelta])

            # Find best scoring payload
            Mstar, pstar = self.min_score(p)

            if verbose:
                print(f"Detected message: {('0' * nbits + bin(gray2int(Mstar))[2:])[-nbits:]}, Mstar={Mstar}")

            # Validate the decodeed message based on the False Positive Rate (FPR)
            if 1 - (1 - nM * pstar) ** ((len(tokens) - h) * blen) < FPR:
                text.random_values_at_decode = Y
                text.score = scores
                text.tkn_scores = tkn_scores
                text.best_score = scores
                text.watermarked = True
                text.detection_key = key
                text.p_value = p
                text.best_p_value = pstar
                text.decoded_message = gray2int(Mstar)
                if isinstance(text, BinarizedText):
                    return BinarizedWatermarkedText.from_binarized_text(
                                text,
                                watermarked_btkns= watermarked_btkns
                            )
                else:
                    text.watermarked_btkns= watermarked_btkns
                    return text
            else:
                text.random_values_at_decode = Y
                text.score = scores
                text.tkn_scores = tkn_scores
                text.best_score = scores
                text.watermarked = False
                text.detection_key = key
                text.p_value = p
                text.best_p_value = pstar
                if isinstance(text, BinarizedText):
                    return text
                else:
                    return BinarizedText.from_binarized_watermarked_text(text) 

            
class OZWatermark(BinarizedWatermark):
    """
    Implements the OZ watermarking method for multi-bit steganography 
    using a binarized language model.
    """
    @staticmethod
    def normalize_score(score, length):
        return (score - length)/math.sqrt(length)
    
    @staticmethod
    def compute_score_function(key, prf_input, bit):
        """
        This function calculates the score function for a binary bit and a payload value
        

        Parameters
        ----------
        key : TYPE
            secret key shared between encoder and decoder.
        prf_input : [i, ind , s], 
            i: indx of the real token
            ind: indx of the binary token
            s: codeword symbole ('0','1','<').
        bit : str
            binary tokne ('0' or '1').

        Returns
        -------
        float
            score value, s(w^b_i, y_i, S).

        """
        u = PRF(key, prf_input)
        v = (u if bit == '1' else (1-u))
        return -math.log(v)
    
    def generate(
        self, 
        key, 
        prompt, 
        payload, 
        length=30, 
        threshold=2, 
        bit_limit=None, 
        temperature=1.0, 
        Rlambda=5, 
        flagR=False, 
        h=4, 
        flagPerm=False,
        verbose=True
    ):
        """
        Generate a steganographic response that embeds the given payload.

        Parameters
        ----------
        key : any
            Secret key shared between encoder and decoder.
        prompt : str
            The input prompt for text generation.
        payload : list of bits
            The binary message to be embedded.
        length : int, optional
            Number of real tokens to generate.
        threshold : int, optional
            Used to determine chunk length for hiding symbols.
        bit_limit : int, optional
            Limit on binary bits per token used for embedding.
        temperature : float, optional
            Softmax temperature for sampling.
        Rlambda : float, optional
            Threshold for switching from random sampling to PRF-based sampling.
        flagR : bool, optional
            If True, uses bit-tracking mechanism.
        h : int, optional
            Context size in tokens.
        verbose : bool, optional
            If True, prints debug info.

        Returns
        -------
        BinarizedWatermarkedText
            The generated watermarked text, along with ECC encoding information.
        """
        flagRchosen = False
        H = 0
        n = 0
        R = []
        watermarked_btkns = []
        # Tokenize the prompt
        prompt_ids = self._tokenize(prompt).to(self.model.device)
        prompt_len_tkn = prompt_ids.shape[1]

        attn = torch.ones_like(prompt_ids)

        # Setup permutation
        vocab_size = len(self.tokenizer)
        if flagPerm:
            perm, inv_perm = consistent_perm(key, vocab_size)
        else:
            perm = range(vocab_size).tolist()
            inv_perm = range(vocab_size).tolist()
        # Not necessary, but makes the token indices spread uniformly.
        # This is done for assigning binary numbers of length blen to the tokens, for 
        # example should 0000 mean the first token of the tokenizer? If we use this permutation
        # then 0000 might refer the 100th token of the tokenizer

        # Retrieve blen from parent class
        blen = self.blen

        if bit_limit:
            assert bit_limit <= blen, "bit_limit cannot exceed blen"

        # Initialize ECC (Error Correcting Code for steganography)
        ecc = DynamicECC(payload)
        symbol = ecc.next_symbol() # symbol is the next symbol that decoder sends

        scores = {'0': 0, '1': 0, '<': 0}
        score_length = 0
        past = None
        lapsedt = []

        # Generation loop
        for i in range(length):  # list of real tokens
            with torch.no_grad():
                if past:
                    output = self.model(
                        prompt_ids[:, -1:], past_key_values=past, attention_mask=attn
                    )
                else:
                    output = self.model(prompt_ids)

            # Apply temperature to logits before softmax
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, : vocab_size] / temperature, dim=-1
            ).cpu()[0, :]

            # Apply permutation to the distribution
            if flagPerm:
                probs = apply_perm(probs, perm)

            token_id = 0
            for ind in range(blen):  
                st = time.time()
                p0, p1 = self._binarize_next(probs, ind, blen, token_id)
                et = time.time()
                lapsedt.append(et - st)

                token_id <<= 1  # token_id is the indx of the overall real token 
                # that is generated so far in permuted tokens, eventually it will be the indx of the real token,
                # corresponding to the blen binary tokens 

                P1 = p1.item() / (p0.item() + p1.item())

                # Randomized sampling phase
                if flagR and not flagRchosen:
                    if random.random() < P1:
                        token_id += 1
                        H -= math.log(P1)
                    else:
                        H -= math.log(1 - P1)
                    n += 1
                    R.append(token_id % 2)

                    if H >= Rlambda and n > h * blen:
                        flagRchosen = True
                        if verbose:
                            print(f"OZWatermark: Threshold reached, n={n}, R={R}")

                # PRF-based sampling phase
                elif flagRchosen and flagR:
                    watermarked_btkns.append(i * blen + ind)    
                    if PRF(key, R + [i, ind, symbol]) < P1: # this is y_j < Pj(1)
                        token_id += 1  # w^b_j = 1 
                elif not flagR:
                    if PRF(key, [i, ind, symbol]) < P1: # this is y_j < Pj(1)
                        token_id += 1 # w^b_j = 1 
                    watermarked_btkns.append(i * blen + ind)    

                # Score tracking for ECC decoding
                if (not bit_limit) or (ind < bit_limit):
                    score_length += 1
                    for s in ['0', '1', '<']:
                        if flagR and not flagRchosen:
                            scores[s] += self.compute_score_function(
                                key, [i, ind, s], str(token_id % 2)
                            )
                        elif flagRchosen and flagR:
                            scores[s] += self.compute_score_function(
                                key, R + [i, ind, s], str(token_id % 2)
                            )
                        elif not flagR:
                            scores[s] += self.compute_score_function(
                                key, [i, ind, s], str(token_id % 2)
                            )

                        if self.normalize_score(scores[s], score_length) > threshold:
                            ecc.update(s)
                            symbol = ecc.next_symbol()
                            scores = {'0': 0, '1': 0, '<': 0}
                            score_length = 0
                            break

            # Map back from permuted ID to the real vocabulary ID
            real_token_id = inv_perm[token_id]
            token = torch.tensor([[real_token_id]], device=self.model.device)
            prompt_ids = torch.cat([prompt_ids, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # Check if the full payload has been decoded
            if DynamicECC.decode(ecc.stream)[:len(payload)] == payload:
                generated_text = self._detokenize(prompt_ids[0][prompt_len_tkn:])
                return BinarizedWatermarkedText(
                    prompt=prompt,
                    text=generated_text,
                    token_ids=prompt_ids[0][prompt_len_tkn:].tolist(),
                    embedded_message=payload,
                    decoded_message=DynamicECC.decode(ecc.stream),
                    generation_key=key,
                    watermarked_btkns=watermarked_btkns
                )

        # If payload is not fully embedded, return the generated text anyway
        generated_text = self._detokenize(prompt_ids[0][prompt_len_tkn:])
        return BinarizedWatermarkedText(
            prompt=prompt,  
            text=generated_text,
            token_ids=prompt_ids[0][prompt_len_tkn:].tolist(),
            embedded_message=payload,
            decoded_message=DynamicECC.decode(ecc.stream),
            generation_key=key, 
            watermarked_btkns=watermarked_btkns
        )
    
    def decode(
        self, 
        key, 
        text, 
        threshold=2, 
        bit_limit=None, 
        flagPerm=False,
        verbose=False
    ):
        """
        decode the payload embedded in watermarked text (OZ method).

        Parameters
        ----------
        key : any
            Key used for the pseudo-random function.
        text :  BinarizedText or BinarizedWatermarkedText
            The watermarked text object containing the tokenized text.
        threshold : float, optional
            Score threshold for determining symbols ('0', '1', or '<').
            The default is 2.
        bit_limit : int or None, optional
            Maximum number of binary bits per token used in embedding.
            The default is None.
        skip_prefix : int, optional
            Number of initial tokens to ignore when decodeing payload.
            The default is 0.

        Returns
        -------
        str
            The decodeed payload as a decoded message.
        """

        # Initialize variables
        stream = []
        scores = {'0': 0, '1': 0, '<': 0}
        score_length = 0

        # Get the permutation
        vocab_size = len(self.tokenizer)
        perm, inv_perm = consistent_perm(key, vocab_size)

        # Retrieve blen from parent class
        blen = self.blen
        tokens = text.token_ids

        # Process each token
        for i in range(len(tokens)):
            # Convert token ID to its binary representation in the permuted index'
            if flagPerm:
                token_bits = ("0" * blen + bin(perm[tokens[i]])[2:])[-blen:]
            else:
                token_bits = ("0" * blen + bin(tokens[i])[2:])[-blen:]
            

            for ind in range(blen):
                if (not bit_limit) or (ind < bit_limit):
                    score_length += 1

                    for s in ['0', '1', '<']:
                        scores[s] += self.compute_score_function(key, [i, ind, s], token_bits[ind])

                        if self.normalize_score(scores[s], score_length) > threshold:
                            stream.append(s)
                            scores = {'0': 0, '1': 0, '<': 0}  # Reset scores
                            score_length = 0
                            break
        if verbose:
            print("decodeed stream:", stream)

        # Decode decodeed binary stream into the final payload message
        text.watermarked = True
        text.detection_key = key
        text.decoded_message = DynamicECC.decode(stream)
        if isinstance(text, BinarizedText):
            return BinarizedWatermarkedText.from_binarized_text(
                    text)
        else:
            return text
        
    
    

