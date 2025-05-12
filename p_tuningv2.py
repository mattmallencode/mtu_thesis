import torch
from torch import nn
from typing import Optional, Tuple, Union, Any
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PrefixEncoder(nn.Module):
    """
    Prefix encoder module implementing P-tuning v2 prefix encoding.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the PrefixEncoder module.

        :param config: Configuration object with the following attributes:
            - pre_seq_len (int): Length of the prefix sequence.
            - hidden_size (int): Hidden size of the model.
            - num_hidden_layers (int): Number of hidden layers in the model.
            - num_attention_heads (int): Number of attention heads.
            - prefix_projection (bool, optional): Apply prefix projection?
            - prefix_hidden_size (int): Hidden size of prefix projection layer.
        """
        super().__init__()
        self.prefix_projection = getattr(config, "prefix_projection", True)
        self.pre_seq_len = config.pre_seq_len
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = self.hidden_size // config.num_attention_heads

        embed_dim = (
            self.hidden_size
            if self.prefix_projection
            else self.num_hidden_layers * 2 * self.hidden_size
        )

        self.embedding = nn.Embedding(self.pre_seq_len, embed_dim)

        if self.prefix_projection:
            self.prefix_projection_layer = nn.Sequential(
                nn.Linear(self.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(
                    config.prefix_hidden_size,
                    self.num_hidden_layers * 2 * self.hidden_size,
                ),
            )

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PrefixEncoder module.

        :param prefix_tokens: Input tensor w/ prefix indices
        :return: Tensor representing the prefix embeddings
        """
        out = self.embedding(prefix_tokens)
        return (
            self.prefix_projection_layer(
                out,
            )
            if self.prefix_projection
            else out
        )


class DPRPrefixEncoder(PreTrainedModel):
    """DPR encoder enhanced with P-tuning v2 prefix-tuning capability."""

    base_model_prefix = "bert_model"

    def __init__(self, config: Any) -> None:
        """
        Initialize DPRPrefixEncoder w/ P-tuning v2.

        :param self: Instance of DPRPrefixEncoder.
        :param config: Model config w/ required parameters such as:
                       - bert_model: The base BERT model.
                       - pre_seq_len: Prefix sequence length.
                       - num_hidden_layers: Number of hidden layers.
                       - num_attention_heads: Number of attention heads.
                       - hidden_size: Hidden size of the model.
                       - hidden_dropout_prob: Dropout probability.
                       - projection_dim (opt): Dimension of projection layer.
        :return: None
        """
        super().__init__(config)
        self.bert_model = config.bert_model
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_encoder = PrefixEncoder(config)
        self.prefix_tokens = torch.arange(
            self.pre_seq_len, device=self.bert_model.device
        )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.projection_dim = getattr(config, "projection_dim", 0)
        if self.projection_dim:
            self.encode_proj = torch.nn.Linear(
                self.bert_model.config.hidden_size, self.projection_dim
            )

        self.init_weights()

    def get_prompt(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Generate prefix prompts for the input batch.

        :param self: Instance of DPRPrefixEncoder.
        :param batch_size: Batch size for which to generate the prefix prompts.
        :return: Tuple of Tensors representing past key values for each layer,
                 formatted as required by the base model.
        """
        past_key_values = self.prefix_encoder(
            self.prefix_tokens.unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.bert_model.device)
        )
        bsz, seqlen, _ = past_key_values.shape
        return (
            self.dropout(
                past_key_values.view(
                    bsz, seqlen, self.n_layer * 2, self.n_head, self.n_embd
                )
            )
            .permute(2, 0, 3, 1, 4)
            .split(2)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.Tensor, ...]]:
        """
        Perform a forward pass through the DPRPrefixEncoder.

        :param self: Instance of DPRPrefixEncoder.
        :param input_ids: Input token IDs.
        :param attention_mask: Attention mask for the input tokens.
        :param token_type_ids: Token type IDs.
        :param inputs_embeds: Optional input embeddings.
        :param output_attentions: Whether to output attention weights.
        :param output_hidden_states: Whether to output hidden states.
        :param return_dict: Whether to return a dict or tuple.
        :return: Model output: a BaseModelOutputWithPooling | tuple of tensors.
        """

        batch_size = (
            input_ids.size(
                0,
            )
            if input_ids is not None
            else None
        )

        if batch_size is None:
            if inputs_embeds is not None:
                batch_size = inputs_embeds.size(0)
            else:
                raise TypeError("Input embeds and ids can't both be None!")

        past_key_values = self.get_prompt(batch_size)

        prefix_attention_mask = torch.ones(
            batch_size, self.pre_seq_len, device=self.bert_model.device
        )
        attention_mask = torch.cat(
            (prefix_attention_mask, attention_mask),
            dim=1,
        )

        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        pooled_output = (
            self.encode_proj(sequence_output[:, 0, :])
            if self.projection_dim
            else sequence_output[:, 0, :]
        )

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        """
        Get the size of the output embeddings.

        :return: Size of embeddings after projection (if applied).
        """
        return (
            self.encode_proj.out_features
            if self.projection_dim
            else self.bert_model.config.hidden_size
        )


class DPRPrefixQuestionEncoder(PreTrainedModel):
    """
    DPR question encoder enhanced with P-tuning v2.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize DPRPrefixQuestionEncoder with P-tuning v2.

        :param config: Configuration object, following optional w/ defaults:
            - pre_seq_len (int, opt): Length of prefix (default: 16).
            - prefix_projection (bool, opt): Apply projection? (default: True).
            - prefix_hidden_size (int, opt): Proj layer size (default: 512).
        """
        super().__init__(config)
        self.config = config

        for attr, default in {
            "pre_seq_len": 16,
            "prefix_projection": True,
            "prefix_hidden_size": 512,
        }.items():
            setattr(config, attr, getattr(config, attr, default))

        self.question_encoder = DPRPrefixEncoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for the DPRPrefixQuestionEncoder.

        :param input_ids: Optional tensor of input token IDs.
        :param attention_mask: Optional tensor; input seq attention mask.
        :param token_type_ids: Optional tensor of token type IDs.
        :param inputs_embeds: Optional tensor of pre-computed input embeddings.
        :param output_attentions: Optional boolean to output attention weights.
        :param output_hidden_states: Optional boolean; output hidden states?
        :param return_dict: Optional boolean to return output as dictionary.
                          If None, uses model config default.
        :return: Either BaseModelOutputWithPooling containing pooled output,
                hidden states, and attentions, or tuple of tensors
                if return_dict is False.
        """

        if return_dict is None:
            return_dict = self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Can't specify both input_ids and inputs_embeds")

        if input_ids is None:
            if inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise TypeError("Must specify input_ids or inputs_embeds")
        else:
            input_shape = input_ids.size()

        if input_ids is None:
            if inputs_embeds is not None:
                device = inputs_embeds.device
            else:
                raise TypeError("Must specify input_ids or inputs_embeds")
        else:
            device = input_ids.device

        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones(input_shape, device=device)
        )
        token_type_ids = (
            token_type_ids
            if token_type_ids is not None
            else torch.zeros(input_shape, dtype=torch.long, device=device)
        )

        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return (
            BaseModelOutputWithPooling(
                pooler_output=outputs.pooler_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            if return_dict
            else outputs[1:]
        )
