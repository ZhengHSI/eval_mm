from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='/home/workspace/model/MiniCPM-Llama3-V-2_5'),
    'MiniCPM-V-2_6': partial(MiniCPM_V_2_6, model_path='openbmb/MiniCPM-V-2_6'),
    'MiniCPM-V-GPTQ-g128': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-g128'),
    'MiniCPM-V-GPTQ-perchannel-32': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-perchannel'),
    'MiniCPM-V-GPTQ-perchannel-32-smooth': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-perchannel-smooth'),
    'MiniCPM-V-GPTQ-perchannel-32-smooth-lmhead': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-perchannel-smooth-lmhead'),
    'MiniCPM-V-GPTQ-perchannel-64': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-64-perchannel'),
    'MiniCPM-V-GPTQ-perchannel-32-lmhead': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-perchannel-lmhead'),
    'MiniCPM-V-1.2B': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/MiniCPM-V-1B-sft-v2-1B'),
    'MiniCPM-V-GPTQ-perchannel-32-lmhead—no_downproj': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-perchannel-lmhead-no_downproj'),
    'MiniCPM-V-GPTQ-perchannel-32-only_quant_downproj': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/minicpm-gptq-w4-32-perchannel-only_quant_downproj'),
    'MiniCPM-Llama3-V-2_5-flatquant': partial(MiniCPM_V_flatquant, model_path='/home/workspace/model/MiniCPM-Llama3-V-2_5'),
    'MiniCPM-V-1B-sft-v2-1B_vit-w4-perchannel': partial(MiniCPM_V_2_6, model_path='/home/workspace/model/MiniCPM-V-1B-sft-v2-1B_vit-w4-perchannel'),
    'MiniCPM-V-autogptq-w8-pc': partial(MiniCPM_AutoGPTQ, model_path='/home/workspace/model/MiniCPM-3o-1B-sft-v1-pc'),
    'MiniCPM-3o-1B-sft-v1': partial(MiniCPM_V_3o, model_path='/home/workspace/model/MiniCPM-3o-1B-sft-v1'),
}

supported_VLM = {}

model_groups = [
    minicpm_series
]

for grp in model_groups:
    supported_VLM.update(grp)
