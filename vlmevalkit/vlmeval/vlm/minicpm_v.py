import math
import torch
import random
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class MiniCPM_V(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt = 'Study the image carefully and pick the option associated with the correct answer. \
                Focus solely on selecting the option and avoid including any other content.\n' + prompt
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])

        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': prompt}]
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 20
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 100
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams
        )
        default_kwargs.update(self.kwargs)
        res, _, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        return res


class MiniCPM_Llama3_V(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-Llama3-V-2_5', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3
        self.options_system_prompt = ('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt + '\nPlease just indicate your choice.'
            else:
                system_prompt = self.wo_options_system_prompt
            if 'MMMU' in dataset:  # Corner Case
                prompt = system_prompt + '\n' + prompt
                system_prompt = ''
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['MME'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench', 'MMLongBench_DOC'], dataset):
                system_prompt = ''
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def generate_inner(self, message, dataset=None):
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 200
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res

    def chat_inner(self, message, dataset=None):
        max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        msgs = []
        for msg in message:
            content = []
            if len(msg['content']) == 1 and msg['content'][0]['type'] == 'text':
                msg_new = {'role': msg['role'], 'content': msg['content'][0]['value']}
                msgs.append(msg_new)
                continue

            for x in msg['content']:
                if x['type'] == 'text':
                    content.append(x['value'])
                elif x['type'] == 'image':
                    image = Image.open(x['value']).convert('RGB')
                    content.append(image)
            msg_new = {'role': msg['role'], 'content': content}
            msgs.append(msg_new)

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs)

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res


class MiniCPM_V_2_6(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()

        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3

        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'HallusionBench', 'OCRBench', 'ChartQA'], dataset):
            return True
        elif listinstr(['MathVista', 'MMVet', 'MMBench', 'MMStar', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMVet', 'MMBench', 'MMStar', 'AI2D', 'OCRBench'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None):
        max_new_tokens = 2048
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        return res

def get_MiniCPM_V_flatquant():
    import flatquant.model_utils as model_utils
    import transformers
    import flatquant.utils as utils
    import flatquant.args_utils as args_utils
    import flatquant.model_utils as model_utils
    import flatquant.data_utils as data_utils
    import flatquant.eval_utils as eval_utils
    import flatquant.train_utils as train_utils
    import flatquant.flat_utils as flat_utils
    import gptq_utils
    import logging
    import pprint
    from datetime import datetime
    from types import SimpleNamespace

    def create_logger(exp_dir, dist_rank=0, name=''):
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # create formatter
        fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
        color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                    colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

        # create console handlers for master process
        if dist_rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(console_handler)

        # create file handlers
        log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
        return logger

    args = {
        # General Arguments
        'model': '/home/workspace/model/MiniCPM-Llama3-V-2_5',
        'seed': 0,
        'hf_token': None,

        # Activation Quantization Arguments
        'a_bits': 4,
        'a_groupsize': -1,
        'a_asym': False,

        # Weight Quantization Arguments
        'w_bits': 4,
        'w_groupsize': -1,
        'w_asym': False,
        'gptq': False,
        'gptq_mse': False,
        'percdamp': 0.01,
        'act_order': False,

        # FlatQuant calibration Arguments
        'epochs': 15,
        'cali_dataset': 'wikitext2',
        'nsamples': 128,
        'cali_bsz': 4,
        'flat_lr': 5e-3,
        'cali_trans': True,
        'add_diag': True,
        'lwc': True,
        'lac': True,
        'resume': False,
        'save_matrix': True,
        'reload_matrix': True,
        'matrix_path': "/home/workspace/code/git/FlatQuant_mlm/outputs/MiniCPM-Llama3-V-2_5/w4a4/exp-mme/",
        'diag_init': 'sq_style',
        'diag_alpha': 0.3,
        'warmup': False,
        'deactive_amp': False,
        'direct_inv': False,
        'separate_vtrans': False,

        # KV-Cache Quantization Arguments
        'q_bits': 16,
        'q_asym': False,
        'q_groupsize': -1,
        'k_bits': 4,
        'k_asym': True,
        'k_groupsize': 128,
        'v_bits': 4,
        'v_asym': True,
        'v_groupsize': 128,

        # Experiments Arguments
        'output_dir': './outputs',
        'exp_name': 'exp',

        # LM Eval Arguments
        'lm_eval': False,
        'tasks': ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
        'lm_eval_batch_size': 128,
        'distribute_model': True,
    }
        # 处理 a_groupsize
    # 转换为可以通过属性访问的对象
    args = SimpleNamespace(**args)

    # 处理 a_groupsize
    if args.a_groupsize > -1:
        raise NotImplementedError

    # cache path
    args.cache_dir = os.path.join(args.output_dir, ".cache")
    os.makedirs(args.cache_dir, exist_ok=True)

    # output path
    args.model_name = args.model.split("/")[-1]
    args.exp_dir = os.path.join(args.output_dir, args.model_name, f"w{args.w_bits}a{args.a_bits}", args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    logger = create_logger(args.exp_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)

    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    model = apply_flatquant_to_model(args, model)
    logger.info("Finished applying FlatQuant to model.")
    flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
    flat_utils.reparameterize_model(model)
    logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        save_dict = {}
        quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    return model

class MiniCPM_V_flatquant(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-Llama3-V-2_5', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = get_MiniCPM_V_flatquant()
        # self.model = self.model.to(dtype=torch.float16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3
        self.options_system_prompt = ('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt + '\nPlease just indicate your choice.'
            else:
                system_prompt = self.wo_options_system_prompt
            if 'MMMU' in dataset:  # Corner Case
                prompt = system_prompt + '\n' + prompt
                system_prompt = ''
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['MME'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench', 'MMLongBench_DOC'], dataset):
                system_prompt = ''
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def generate_inner(self, message, dataset=None):
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 200
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res

    def chat_inner(self, message, dataset=None):
        max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        msgs = []
        for msg in message:
            content = []
            if len(msg['content']) == 1 and msg['content'][0]['type'] == 'text':
                msg_new = {'role': msg['role'], 'content': msg['content'][0]['value']}
                msgs.append(msg_new)
                continue

            for x in msg['content']:
                if x['type'] == 'text':
                    content.append(x['value'])
                elif x['type'] == 'image':
                    image = Image.open(x['value']).convert('RGB')
                    content.append(image)
            msg_new = {'role': msg['role'], 'content': content}
            msgs.append(msg_new)

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs)

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res

class MiniCPM_AutoGPTQ(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')

        from auto_gptq import AutoGPTQForVIT, BaseQuantizeConfig
        self.model = AutoGPTQForVIT.from_quantized(model_path, device="cuda:0", use_triton=False).model 

        from transformers import AutoTokenizer, TextGenerationPipeline,AutoProcessor
        pretrained_model_dir = "/home/workspace/model/MiniCPM-3o-1B-sft-v1"
        self.model.processor = AutoProcessor.from_pretrained(pretrained_model_dir, trust_remote_code=True)
        # self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()

        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3

        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'HallusionBench', 'OCRBench', 'ChartQA'], dataset):
            return True
        elif listinstr(['MathVista', 'MMVet', 'MMBench', 'MMStar', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMVet', 'MMBench', 'MMStar', 'AI2D', 'OCRBench'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None):
        max_new_tokens = 2048
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        return res

class MiniCPM_V_3o(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        self.model_path = model_path
        print(f'load from path {self.model_path}')
        # self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        from auto_gptq import AutoGPTQForVIT, BaseQuantizeConfig
        quantize_config = BaseQuantizeConfig(
            bits=8,  # quantize model to 4-bit
            group_size=-1,  # it is recommended to set the value to 128
            desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
        )
        self.model = AutoGPTQForVIT.from_pretrained(self.model_path, quantize_config).model
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3

        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'HallusionBench', 'OCRBench', 'ChartQA'], dataset):
            return True
        elif listinstr(['MathVista', 'MMVet', 'MMBench', 'MMStar', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMVet', 'MMBench', 'MMStar', 'AI2D', 'OCRBench'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None):
        max_new_tokens = 2048
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        return res
