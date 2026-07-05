# This work is based in part on code from Show-o (https://github.com/showlab/Show-o).
# Significant modifications for NeoBabel.

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk

class NeoBabel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_huggingface=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        # eager attention: gemma-2 logit soft-capping is silently disabled under sdpa
        if load_from_huggingface:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.neobabel = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        else:
            self.neobabel = AutoModelForCausalLM.from_pretrained(llm_model_path, attn_implementation="eager")
            
        self.neobabel.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):

        if input_embeddings is None:
            logits = self.neobabel(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.neobabel(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            zero_loss = logits.new_zeros(())

            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            if batch_size_t2i > 0:
                loss_t2i = F.cross_entropy(
                    logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                    labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
            else:
                loss_t2i = zero_loss

            # 2. Next token prediction for language modeling
            if batch_size_lm > 0:
                loss_lm = F.cross_entropy(
                    logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                    labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
            else:
                loss_lm = zero_loss

            # 3. Next token prediction for captioning/multimodal understanding
            # (guard batch_size_mmu == 0: logits[-0:] would select the whole batch)
            if batch_size_mmu > 0:
                loss_mmu = F.cross_entropy(
                    logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                    labels[-batch_size_mmu:, 1:].contiguous().view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
            else:
                loss_mmu = zero_loss

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.neobabel.num_vq_tokens
        num_new_special_tokens = config.model.neobabel.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.neobabel.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.neobabel.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.neobabel.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.neobabel.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
