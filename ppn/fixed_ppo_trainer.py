import time
from typing import List, Dict
import warnings
import pandas as pd
import wandb
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import Dataset
from trl import PPOTrainer, PreTrainedModelWrapper, PPOConfig
from trl.core import (
    WANDB_PADDING,
    masked_mean,
    PPODecorators,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
    convert_to_scalar,
    masked_whiten,
    clip_by_value,
    entropy_from_logits,
    masked_var,
    flatten_dict
)

import torch.distributed as dist
from logging import getLogger
from utils.log import set_logger

logger = getLogger(__name__)
set_logger(logger)

@dataclass
class FixedPPOConfig(PPOConfig):
    """
    Changed: Add entropy loss coefficient
    """
    ent_coef: Optional[float] = field(default=0.0, metadata={"help": "Scaling factor for entropy loss"})


class FixedPPOTrainer(PPOTrainer):
    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Changed: Set to train mode (enable dropout) for training
        """
        is_training_mode = self.model.training

        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"], dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=pad_first
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"], dim=1, pad_index=0, pad_first=pad_first
                )

        model_inputs_names = list(model_inputs.keys())

        with torch.no_grad():
            # Set to eval mode (disable dropout) for deterministic logprobs
            self.model.eval()

            all_logprobs, _, values, masks = self.batched_forward_pass(self.model, queries, responses, model_inputs)

            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model, "disable_adapter"
            ):
                with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                    ref_logprobs, _, _, _ = self.batched_forward_pass(self.model, queries, responses, model_inputs)
            elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )

            else:
                ref_logprobs, _, _, _ = self.batched_forward_pass(self.ref_model, queries, responses, model_inputs)

        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
        timing["time/ppo/compute_rewards"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        mini_batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "rewards": rewards,
            "masks": masks,
        }

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["queries", "responses"]:
                    return_dict[key] = [d[key] for d in data]
                else:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(self.current_device)
            return return_dict

        mini_batch_dict.update(model_inputs)
        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        t = time.time()
        all_stats = []
        early_stop = False
        self.model.train()
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            for batch in mini_batch_dataloader:
                with self.accelerator.accumulate(self.model):
                    model_inputs = {k: batch[k] for k in model_inputs_names}
                    logprobs, logits, vpreds, _ = self.batched_forward_pass(
                        self.model, batch["queries"], batch["responses"], model_inputs, return_logits=True
                    )

                train_stats = self.train_minibatch(
                    batch["logprobs"],
                    batch["values"],
                    batch["rewards"],
                    logprobs,
                    logits,
                    vpreds,
                    batch["masks"],
                )

                all_stats.append(train_stats)

                if self.config.early_stopping:
                    policykl = train_stats["policy/policykl"]
                    early_stop = self._early_stop(policykl)
                    if early_stop:
                        break
        # Set back to previous mode
        if not is_training_mode:
            self.model.eval()

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(stats["objective/kl"], self.config.batch_size * self.accelerator.num_processes)

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        """
        Changed: Force forward batch size to 1 to avoid CUDA OOM
        """
        bs = len(queries)
        # fbs = self.config.mini_batch_size
        fbs = 1
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]
            # if self.accelerator.is_main_process:
            #     breakpoint()
            for j in range(fbs):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                if len(logprobs[j, start:end]) < 1: # Apr 28, 2023: changed from 2 to 1
                    print(f"j: {j}, start: {start}, end: {end}")
                    print("input_ids", input_ids[j])
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_cuda_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Changed: Use entropy loss
        """
        loss_p, loss_v, loss_ent, train_stats = self.loss(old_logprobs, values, rewards, logits, vpreds, logprobs, mask)
        loss = loss_p + loss_v + loss_ent
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_grad_norm
            )

        t = time.time()
        self.optimizer.step()
        train_stats["time/ppo/optimizer_step"] = torch.Tensor([time.time() - t]).to(self.current_device)
        return train_stats


    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Changed: Add entropy loss
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds, values - self.config.cliprange_value, values + self.config.cliprange_value
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        entropy = masked_mean(entropy_from_logits(logits), mask)
        ent_loss = -entropy

        loss = pg_loss + self.config.vf_coef * vf_loss + self.config.ent_coef * ent_loss

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), ent_loss=ent_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, self.config.ent_coef * ent_loss, flatten_dict(stats)


    def gather_stats(self, stats):
        """
        Changed: Use `gather()` instead of `all_reduce()` for list-type stats (e.g., kl_dist, logprobs)
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    # concatenate tensors from all processes
                    v = self.accelerator.gather(v)
                else:
                    # mean value from all processes
                    dist.all_reduce(v, dist.ReduceOp.SUM)
                    v /= self.accelerator.num_processes
                stats[k] = v
        return stats

    def record_step_stats(self, kl_coef: float, **data):
        """
        Changed: Use tensor instead of numpy on queries and responses stats for wandb logging
        """
        mask = data.pop("masks")

        kl_list = ((data["logprobs"] - data["ref_logprobs"]) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )  # non_score_reward is size `batch_size`, `response_length`
        mean_scores = torch.stack(data["scores"]).mean()  # scores is size `batch_size`
        std_scores = torch.stack(data["scores"]).std()

        if mean_kl.item() < 0.0:
            # warn users
            warnings.warn(
                f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
                " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
                " that the generation kwargs are set correctly, or review your training hyperparameters."
            )

        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
        }

        # Log text properties
        query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float, device=self.current_device)
        response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float, device=self.current_device)

        stats["tokens/queries_len_mean"] = torch.mean(query_lens)
        stats["tokens/queries_len_std"] = torch.std(query_lens)
        stats["tokens/queries_dist"] = query_lens
        stats["tokens/responses_len_mean"] = torch.mean(response_lens)
        stats["tokens/responses_len_std"] = torch.std(response_lens)
        stats["tokens/responses_dist"] = response_lens

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats

    def log_stats(
        self,
        stats: dict,
        wandb_table_data: Dict[str, wandb.Table],
        rewards: List[torch.FloatTensor],
    ):
        """
        Changed:
        1. Use `gather()` instead of `all_reduce()` for reward logging
        2. Use arbitrary table dataframe for wandb logging
        """
        # Gather all rewards if distributed
        if self.is_distributed:
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.current_device)
            gathered_rewards = self.accelerator.gather(rewards)
        else:
            gathered_rewards = rewards

        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            # Log stats
            logs = {}
            logs.update(stats)

            if self.config.log_with == "wandb":
                logs.update(wandb_table_data)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(gathered_rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(gathered_rewards).cpu().numpy().item()
            logs["env/reward_dist"] = gathered_rewards.cpu().numpy()

            if self.config.log_with == "tensorboard":
                # update the current step
                self.current_step += 1

            self.accelerator.log(logs, step=self.current_step if self.config.log_with == "tensorboard" else None)

