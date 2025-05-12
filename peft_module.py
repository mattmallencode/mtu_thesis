import warnings
import pytorch_lightning as pl
import numpy as np
import torch
import json
from types import MethodType
from typing import Any, Tuple, Dict
from peft import (
    TaskType,
    PeftModel,
    get_peft_model,
)
from peft.peft_model import (
    PeftType,
    PeftModelForSeq2SeqLM,
    shift_tokens_right,
    _get_batch_size,
)
from transformers import BartForConditionalGeneration
from rag_end2end import finetune_rag

GEN_PARAMS = 406291456
DPR_QUESTION_PARAMS = 108891648


delattr(finetune_rag.GenerativeQAModule, "validation_epoch_end")
delattr(finetune_rag.BaseTransformer, "test_epoch_end")
delattr(finetune_rag.GenerativeQAModule, "test_epoch_end")


def forward(
    self,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    decoder_input_ids: torch.Tensor | None = None,
    decoder_attention_mask: torch.Tensor | None = None,
    decoder_inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    task_ids: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Forward pass for PEFT model: handles encoder and encoder-decoder.

    :param self: Instance of the PEFT model
    :param input_ids: Input token IDs
    :param attention_mask: Mask to avoid attention on padding tokens
    :param inputs_embeds: Pre-computed input embeddings
    :param decoder_input_ids: Decoder input token IDs for seq2seq models
    :param decoder_attention_mask: Mask for decoder inputs
    :param decoder_inputs_embeds: Pre-computed decoder input embeddings
    :param labels: Labels for the sequence classification/regression loss
    :param output_attentions: Whether to return attention weights
    :param output_hidden_states: Whether to return hidden states
    :param return_dict: Whether to return a ModelOutput instead of a tuple
    :param task_ids: Task IDs for multi-task learning scenarios
    :param kwargs: Additional keyword arguments passed to the base model
    :return: Model outputs including loss, logits, and/or attention weights
    """
    peft_config = self.active_peft_config

    if not peft_config.is_prompt_learning:
        if peft_config.peft_type == PeftType.POLY:
            kwargs["task_ids"] = task_ids

        with self._enable_peft_forward_hooks(**kwargs):
            return self.base_model(
                **{
                    k: v
                    for k, v in locals().items()
                    if k
                    not in [
                        "self",
                        "kwargs",
                        "peft_config",
                    ]
                    and v is not None
                }
                | {
                    k: v
                    for k, v in kwargs.items()
                    if k not in self.special_peft_forward_args
                }
            )

    is_encoder_only = not hasattr(self.base_model, "decoder") and not any(
        model_type in str(type(self.base_model).__name__).lower()
        for model_type in ["seq2seq", "encoder-decoder", "t5", "bart"]
    )
    batch_size = _get_batch_size(
        decoder_input_ids if not is_encoder_only else input_ids,
        decoder_inputs_embeds if not is_encoder_only else inputs_embeds,
    )

    if peft_config.peft_type != PeftType.PREFIX_TUNING:
        for mask_name in ["attention_mask", "decoder_attention_mask"]:
            if locals()[mask_name] is not None:
                prefix_mask = torch.ones(
                    batch_size,
                    peft_config.num_virtual_tokens,
                ).to(locals()[mask_name].device)
                locals()[mask_name] = torch.cat(
                    (prefix_mask, locals()[mask_name]), dim=1
                )

    for key in ["position_ids", "token_type_ids"]:
        if kwargs.get(key):
            warnings.warn(
                f"{key} are not supported for parameter efficient tuning."
                f"Ignoring {key}."
            )
            kwargs[key] = None

    model_kwargs = {
        "attention_mask": attention_mask,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict,
        **kwargs,
    }

    if not is_encoder_only:
        model_kwargs.update(
            {
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
            }
        )

    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        model_kwargs["past_key_values"] = self.get_prompt(batch_size)
        if "encoder_outputs" in kwargs:
            model_kwargs["encoder_outputs"] = kwargs["encoder_outputs"]

        return self.base_model(
            input_ids=input_ids,
            **(
                {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_inputs_embeds": decoder_inputs_embeds,
                }
                if not is_encoder_only
                else {}
            ),
            **({"inputs_embeds": inputs_embeds} if is_encoder_only else {}),
            **model_kwargs,
        )

    if is_encoder_only:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)
        if inputs_embeds is not None:
            prompts = self.get_prompt(batch_size, task_ids).to(
                inputs_embeds.dtype
                if inputs_embeds is not None
                else self.word_embeddings.weight.dtype
            )
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **model_kwargs)
        return self.base_model(input_ids=input_ids, **model_kwargs)

    if decoder_inputs_embeds is None and decoder_input_ids is None:
        decoder_input_ids = (
            shift_tokens_right(
                labels,
                self.config.pad_token_id,
                self.config.decoder_start_token_id,
            )
            if labels is not None
            else None
        )
        decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

    if decoder_inputs_embeds is not None:
        prompts = self.get_prompt(
            batch_size,
            task_ids,
        ).to(decoder_inputs_embeds.dtype)
        decoder_inputs_embeds = torch.cat(
            (prompts, decoder_inputs_embeds),
            dim=1,
        )

    if "encoder_outputs" in kwargs:
        model_kwargs["encoder_outputs"] = kwargs["encoder_outputs"]

    return self.base_model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        decoder_inputs_embeds=decoder_inputs_embeds,
        **model_kwargs,
    )


class PEFTGenerativeQAModule(finetune_rag.GenerativeQAModule):
    def __init__(self, hparams, **kwargs):
        """
        Initialize PEFT-enhanced Generative QA Module.

        :param self: Instance of PEFTGenerativeQAModule
        :param hparams: Hyperparameters for model configuration
        :param kwargs: Additional keyword arguments passed to parent class
        :return: None
        """
        super().__init__(hparams, **kwargs)

        self.train_steps = 0
        self.metrics_dict = {}
        self.test_metrics_dict = {}

        if not self.is_rag_model:
            return

        from peft import LoraConfig

        generator_lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        qenc = self.model.rag.question_encoder

        qenc_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )

        qenc = self.model.rag.question_encoder
        config = qenc.config
        config.pre_seq_len = 20
        config.prefix_projection = True
        bert_config = qenc.question_encoder.bert_model.config
        config.prefix_hidden_size = bert_config.hidden_size // 5
        config.bert_model = qenc.question_encoder.bert_model

        self.model.rag.question_encoder = get_peft_model(
            self.model.rag.question_encoder, qenc_lora_config
        )
        self.model.rag.generator = self.peftify_generator(
            self.model.rag.generator,
            generator_lora_config,
        )

    def peftify_generator(self, model, _config) -> PeftModel:
        """
        Transform a pretrained model into a PEFT model if not already one.

        :param model: Original model to be transformed
        :param _config: The PEFT config to use
        :return: PEFT-enhanced model with custom forward method
        """
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, _config)
            model.forward = MethodType(forward, model)
        return model

    def _step(self, batch: dict) -> Tuple:
        if not self.is_rag_model:
            raise ValueError("Only RAG models are supported")

        generator = self.model.rag.generator
        source_ids, source_mask = batch["input_ids"], batch["attention_mask"]
        target_ids = batch["decoder_input_ids"]

        if isinstance(
            generator,
            (BartForConditionalGeneration, PeftModelForSeq2SeqLM),
        ):
            if (
                isinstance(generator, PeftModelForSeq2SeqLM)
                and generator.base_model.config.model_type != "bart"
            ):
                raise ValueError(
                    f"Only BART-based PEFT models are supported,"
                    f"got: {generator.base_model.config.model_type}"
                )
            decoder_input_ids = target_ids
        else:
            raise ValueError(f"Unsupported generator type: {type(generator)}")

        if decoder_input_ids is None:
            raise TypeError("decoder_input_ids can't be None!")

        return (
            self(
                source_ids,
                attention_mask=source_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                labels=decoder_input_ids,
                reduce_loss=True,
            )["loss"],
        )

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Execute a test step with the model on a batch of data.

        Updates test metrics dict with new metrics from the gen step.

        :param self: Instance of PEFTGenerativeQAModule
        :param batch: Dictionary containing the input batch data
        :param batch_idx: Index of the current batch
        :return: Dict w/ step metrics from the generative step
        """

        return self._generative_step(batch) | (
            lambda m: self.test_metrics_dict.update(
                {
                    k: self.test_metrics_dict[k]
                    + [v.detach() if isinstance(v, torch.Tensor) else v]
                    for k, v in m.items()
                    if k in self.test_metrics_dict
                }
            )
            or m
        )(self._generative_step(batch))

    def on_validation_epoch_start(self) -> None:
        """
        Initialize metrics dictionary at the start of validation epoch.

        :param self: Instance of GenerativeQAModule.
        :return: None
        """
        for metric_attr in ("progress_bar_metrics", "callback_metrics"):
            if hasattr(self.trainer, metric_attr) and "loss" in getattr(
                self.trainer, metric_attr
            ):
                print(
                    f"\nCurrent training loss before validation: {getattr(self.trainer, metric_attr,)['loss']:.4f}"
                )
                break

        self.metrics_dict = {
            k: []
            for k in [
                "loss",
                "gen_time",
                "gen_len",
                "em",
            ]
        }

    def on_validation_epoch_end(self) -> None:
        """
        Process and log validation metrics at the end of validation epoch.

        Calculates avg metrics: loss, generation time, generation length,
        and exact match score. Logs these metrics and updates the step count.

        :param self: Instance of GenerativeQAModule.
        :return: None
        """
        self.step_count += 1

        metrics = {
            f"val_avg_{key}": (
                float(torch.stack(values).mean())
                if key == "loss"
                and values
                and all(isinstance(v, torch.Tensor) for v in values)
                else (
                    float(
                        np.mean([x for x in values if x is not None]),
                    )
                    if values
                    else 0.0
                )
            )
            for key, values in self.metrics_dict.items()
        }

        metrics["step_count"] = self.step_count

        if any(v != 0.0 for v in metrics.values()):
            self.save_metrics(metrics, "val")
            self.log_dict(
                {
                    **{
                        k: v
                        for _ in ["val_avg_em", "step_count", "val_avg_loss"]
                        for k, v in metrics.items()
                    },
                    "val_loss": metrics["val_avg_loss"],
                    "val_em": torch.tensor(
                        metrics["val_avg_em"],
                        device=self.device,
                    ),
                }
            )

    def on_test_epoch_start(self) -> None:
        """
        Initialize test metrics dictionary at the start of test epoch.

        :param self: Instance of PEFTGenerativeQAModule
        :return: None
        """
        self.test_metrics_dict = {
            k: []
            for k in [
                "loss",
                "gen_time",
                "gen_len",
                "em",
            ]
        }

    def on_test_epoch_end(self) -> None:
        """
        Process and log test metrics at the end of test epoch.

        Calculates avg metrics including loss, gen time, gen length,
        and EM score. Saves and logs these metrics if non-zero values.

        :param self: Instance of PEFTGenerativeQAModule
        :return: None
        """

        metrics = {
            f"test_avg_{key}": (
                float(torch.stack(values).mean())
                if key == "loss"
                and values
                and all(isinstance(v, torch.Tensor) for v in values)
                else (
                    float(np.mean([x for x in values if x is not None]))
                    if values
                    else 0.0
                )
            )
            for key, values in self.test_metrics_dict.items()
        } | {"step_count": self.step_count}

        if not all(v == 0.0 for v in metrics.values()):
            self.save_metrics(metrics, "test")
            self.log_dict(
                {
                    k: metrics[k]
                    for k in (
                        "test_avg_em",
                        "step_count",
                        "test_avg_loss",
                    )
                }
                | {
                    "test_loss": metrics["test_avg_loss"],
                    "test_em": torch.tensor(
                        metrics["test_avg_em"],
                        device=self.device,
                    ),
                }
            )

    def validation_step(self, batch, batch_idx) -> dict:
        """Add metrics to the metrics_dict during validation"""
        metrics = self._generative_step(batch)
        for k, v in metrics.items():
            if k in self.metrics_dict:
                self.metrics_dict[k].append(v)
        return metrics

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Save model checkpoint on rank zero process only.

        :param self: Instance of PEFTGenerativeQAModule
        :param checkpoint: Dictionary containing the checkpoint data
        :return: None
        """
        save_path = self.output_dir.joinpath(f"checkpoint{self.step_count}")
        self.model.config.save_step = self.step_count

        self.tokenizer.save_pretrained(save_path)

        generator_path = save_path.joinpath("generator")
        generator_path.mkdir(exist_ok=True, parents=True)
        self.model.rag.generator.save_pretrained(generator_path)

        qenc_path = save_path.joinpath("question_encoder")
        qenc_path.mkdir(exist_ok=True, parents=True)
        self.model.rag.question_encoder.save_pretrained(qenc_path)

        model_config = {
            "step_count": self.step_count,
            "model_type": "peft_rag_with_lora",
            "components": ["generator", "question_encoder"],
            "is_rag_model": self.is_rag_model,
        }

        with open(save_path.joinpath("model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

    def training_step(self, batch, batch_idx) -> Dict:
        self.train_steps += 1
        loss = super().training_step(batch, batch_idx)
        return loss
