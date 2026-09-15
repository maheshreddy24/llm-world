"""The VLM wrapper: builds the one-frame-plus-text-history prompt each step
and calls generate(). See eval_crafter.py's module docstring for why only the
current frame is sent (no image history)."""
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.constants import format_actions
from src.observation import MASTER_PROMPT, NO_PLAN, STEP_PROMPT, format_history


class Agent:
    def __init__(self, args, manual):
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model)
        torch_dtype = getattr(torch, args.dtype)
        kwargs = dict(device_map=args.device)
        if args.attn:
            kwargs['attn_implementation'] = args.attn
        try:  # transformers renamed torch_dtype -> dtype
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch_dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model, torch_dtype=torch_dtype, **kwargs)
        self.model.eval()
        self.model.requires_grad_(False)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.system = MASTER_PROMPT.format(manual=manual, actions=format_actions())
        self.prompt_tokens = 0
        self.output_tokens = 0

    @torch.inference_mode()
    def act(self, image, observation, history, plan, step):
        """One frame, text history. See module docstring for why."""
        user_text = STEP_PROMPT.format(
            step=step, observation=observation,
            history=format_history(history, self.args.detail_steps),
            plan=plan or NO_PLAN)
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                         {'type': 'text', 'text': user_text}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.model.device)
        prompt_length = int(inputs['input_ids'].shape[1])

        started = time.monotonic()
        sample = self.args.temperature > 0
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=sample,
            temperature=self.args.temperature if sample else None,
            top_p=0.95 if sample else None,
            top_k=64 if sample else None)
        latency = time.monotonic() - started

        new_tokens = generated[0][prompt_length:]
        self.prompt_tokens += prompt_length
        self.output_tokens += int(new_tokens.shape[0])
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return text, {'prompt_tokens': prompt_length,
                      'output_tokens': int(new_tokens.shape[0]),
                      'latency': latency,
                      'user_text': user_text}
