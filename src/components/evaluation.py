import gc
import os
import math
import pandas as pd
import transformers
import torch
from typing import Union, List
from src.constants import DEVICE
from collections import Counter

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALELLISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParticipantVisibleError(Exception):
    pass

def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        model_path: str,
        load_in_8_bit: bool = False,
        clear_mem: bool = False
)->float:
    sol_count = solution.loc[:, "text"].str.split().apply(Counter)
    sub_count = submission.loc[:, "text"].str.split().apply(Counter)
    invalid_mask =  sol_count != sub_count
    if invalid_mask.any():
        raise ParticipantVisibleError(
            "At least one submitted string is not a valid permutation of the solution string"
        )
    sub_string = [
        " ".join(s.split()) for s in submission['text'].tolist()
    ]
    pass


class PerplexityCalculator:
    def __init__(
            self,
            model_path: str,
            load_in_8bit: bool = False,
            device_map: str = "auto"
        ):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, 
            padding='right'
        )

        if load_in_8bit:
            if DEVICE.type != "cuda":
                raise ValueError("8 bit quantization requires CUDA device")
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map
            )            
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
                device_map=device_map
            )
        
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.eval()
        
    def get_perplexity(
            self,
            input_texts: Union[str, List[str]],
            batch_size=64
        )->Union[float, List[float]]:
            single_input = isinstance(input_texts, str)
            input_texts = [input_texts] if single_input else input_texts

            loss_input = []

            batches = math.ceil(len(input_texts) / batch_size)#130 input, batchsize=64, we need 3 batches(3*64=192)
            for j in range(batches):
                a = j * batch_size
                b = (j + 1) * batch_size
                input_batch = input_texts[a:b]

                with torch.no_grad():
                    
                    text_with_special = [
                        f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
                        for text in input_batch
                    ]
                    # The user manually adds bos_token and eos_token:
                    # To mimic training behavior
                    # To control exactly what the model sees
                    # To avoid inconsistent or missing tokens from the tokenizer

                    model_inputs = self.tokenizer(
                        text_with_special,
                        return_tensors='pt',
                        add_special_tokens=False,
                        padding=True
                    )

                    if "token_type_ids" in model_inputs:
                        model_inputs.pop("token_type_ids")#protects from TypeError: forward() got an unexpected keyword argument 'token_type_ids' for some models

                    model_inputs = {k:v.to(DEVICE) for k, v in model_inputs.items()}

                    output = self.model(**model_inputs, use_cache=False)
                    #If you're running generation (e.g., sampling), you usually want use_cache=True. But for scoring or training, use_cache=False is the correct choice.
                    logits = output['logits']

                    label = model_inputs["input_ids"]
                    label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID
                    

