from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain.llms  import HuggingFacePipeline
import torch


def get_llm(adapt_model_name, base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(adapt_model_name, device_map='cuda:0', 
                torch_dtype=torch.bfloat16)
    
    pipe=pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        top_p=0.5,
        temperature=0.3,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm