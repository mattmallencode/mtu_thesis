import json
import logging
import os
import shutil
import types
from multiprocessing import Process
from pathlib import Path
from typing import Union
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.dpr.modeling_dpr import DPRQuestionEncoderOutput

from p_tuningv2 import DPRPrefixQuestionEncoder
from peft_module import PEFTGenerativeQAModule

logger = logging.getLogger(__name__)


def enhanced_retrieve(
    self, question_hidden_states: torch.Tensor, n_docs: int
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Enhanced retrieval process with re-ranking using adapter.

    First retrieves a larger candidate set, then applies the embedding adapter
    to both query and documents before re-ranking based on similarity.

    :param self: Instance of the retriever
    :param question_hidden_states: Tensor containing query embeddings
    :param n_docs: Number of documents to return after re-ranking
    :return: Tuple of (embeddings, IDs, document dictionaries)
    """
    np_to_torch_dtype = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    if not hasattr(self, "original_embeddings"):
        self.original_embeddings = np.stack(self.index.dataset["embeddings"])

    initial_n_docs = 400

    if isinstance(question_hidden_states, torch.Tensor):

        adapted_query = self.adapter(question_hidden_states)

        adapted_query_np = adapted_query.clone().detach().cpu().numpy()
    else:
        torch_dtype = np_to_torch_dtype.get(
            question_hidden_states.dtype,
            torch.float32,
        )
        question_hidden_states = torch.tensor(
            question_hidden_states, device="cuda", dtype=torch_dtype
        )
        adapted_query = self.adapter(question_hidden_states)

        adapted_query_np = adapted_query.clone().detach().cpu().numpy()

    _, doc_ids, doc_dicts = self._original_retrieve(
        adapted_query_np,
        initial_n_docs,
    )

    if isinstance(doc_ids, torch.Tensor):
        doc_ids_np = doc_ids.cpu().numpy()
    else:
        doc_ids_np = np.array(doc_ids)

    original_embeddings = self.original_embeddings[doc_ids_np.reshape(-1)]
    original_embeddings = original_embeddings.reshape(
        doc_ids_np.shape[0], doc_ids_np.shape[1], -1
    )

    torch_dtype = np_to_torch_dtype.get(
        (
            question_hidden_states.dtype
            if hasattr(question_hidden_states, "dtype")
            else np.float32
        ),
        torch.float32,
    )

    original_embeddings = torch.tensor(
        original_embeddings, device="cuda", dtype=torch_dtype
    )

    batch_size, num_docs, embed_dim = original_embeddings.shape
    adapted_docs = self.adapter(original_embeddings.view(-1, embed_dim)).view(
        batch_size, num_docs, embed_dim
    )

    similarity_scores = torch.bmm(
        adapted_query.unsqueeze(1), adapted_docs.transpose(1, 2)
    ).squeeze(1)

    _, top_indices = torch.topk(similarity_scores, k=n_docs, dim=1)

    reranked_doc_embeds = torch.stack(
        [adapted_docs[i][top_indices[i]] for i in range(batch_size)]
    )

    reranked_doc_ids = torch.stack(
        [
            torch.tensor(
                doc_ids_np[i][top_indices[i].cpu().numpy()],
                device="cuda",
            )
            for i in range(batch_size)
        ]
    )

    reranked_doc_dicts = []
    for i in range(batch_size):
        shard = doc_dicts[i]
        indices = top_indices[i].cpu().numpy()
        new_shard = {k: [v[idx] for idx in indices] for k, v in shard.items()}
        reranked_doc_dicts.append(new_shard)

    return reranked_doc_embeds, reranked_doc_ids, reranked_doc_dicts


class DeepMLPEmbeddingAdapter(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list = [896, 1024, 896],
        output_dim: int = 768,
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual: bool = True,
    ):
        """
        Deep MLP Embedding Adapter w/ multiple hidden layers.

        :param input_dim: Dimension of input embeddings
        :param hidden_dims: List of hidden layer dimensions
        :param output_dim: Dimension of output embeddings
        :param dropout: Dropout probability
        :param layer_norm: Whether to apply layer normalization
        :param residual: Whether to use residual connections
        """
        super().__init__()
        self.residual = residual

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
            )

        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)

        self.activations = nn.ModuleList(
            [nn.GELU() for _ in range(len(hidden_dims))],
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(hidden_dims) + 1)]
        )

        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

        with torch.no_grad():
            nn.init.zeros_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

            for layer in self.hidden_layers:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Deep MLP adapter.

        :param x: Input embeddings tensor
        :return: Transformed embeddings
        """
        hidden = self.input_proj(x)
        hidden = self.activations[0](hidden)
        hidden = self.dropouts[0](hidden)

        for i, layer in enumerate(self.hidden_layers):
            hidden = layer(hidden)
            hidden = self.activations[i + 1](hidden)
            hidden = self.dropouts[i + 1](hidden)

        delta = self.output_proj(hidden)
        delta = self.dropouts[-1](delta)

        if self.layer_norm is not None:
            delta = self.layer_norm(delta)

        if self.residual:
            output = x + delta
        else:
            output = delta

        return output


class ShallowMLPEmbeddingAdapter(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        output_dim: int = 768,
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual: bool = True,
    ):
        """
        Lightweight Shallow MLP Embedding Adapter with a single hidden layer.

        :param input_dim: Dimension of input embeddings
        :param hidden_dim: Dimension of single hidden layer
        :param output_dim: Dimension of output embeddings
        :param dropout: Dropout probability
        :param layer_norm: Whether to apply layer normalization
        :param residual: Whether to use residual connections
        """

        super().__init__()
        self.residual = residual

        hidden_dims = [hidden_dim]

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()

        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)

        self.activations = nn.ModuleList([nn.GELU()])
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout), nn.Dropout(dropout)],
        )

        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

        with torch.no_grad():
            nn.init.zeros_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Shallow MLP adapter.

        :param x: Input embeddings tensor
        :return: Transformed embeddings
        """

        hidden = self.input_proj(x)
        hidden = self.activations[0](hidden)
        hidden = self.dropouts[0](hidden)

        delta = self.output_proj(hidden)
        delta = self.dropouts[-1](delta)

        if self.layer_norm is not None:
            delta = self.layer_norm(delta)

        if self.residual:
            output = x + delta
        else:
            output = delta

        return output


class TransformerEmbeddingAdapter(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        init_with_zeros: bool = True,
    ):
        """
        Transformer-based Embedding Adapter with self-attention mechanism.

        :param input_dim: Dimension of input embeddings
        :param hidden_dim: Dimension of feed-forward hidden layer
        :param num_heads: Number of attention heads
        :param dropout: Dropout probability
        :param residual: Whether to use residual connections
        :param init_with_zeros: Whether to initialize weights with zeros
        """
        super().__init__()
        self.residual = residual

        self.head_dim = input_dim // num_heads
        assert (
            self.head_dim * num_heads == input_dim
        ), "input_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.ff_linear1 = nn.Linear(input_dim, hidden_dim)
        self.ff_linear2 = nn.Linear(hidden_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.activation = nn.GELU()
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        if init_with_zeros:
            with torch.no_grad():
                for layer in [
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.out_proj,
                    self.ff_linear1,
                    self.ff_linear2,
                ]:
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer adapter.

        :param x: Input embeddings tensor
        :return: Transformed embeddings
        """
        orig_shape = x.shape
        is_3d_input = len(orig_shape) == 3
        if not is_3d_input:
            x = x.unsqueeze(1)

        batch_size, seq_len, input_dim = x.shape
        residual = x
        x = self.norm1(x)

        q = (
            self.q_proj(
                x,
            )
            .view(
                batch_size,
                seq_len,
                self.head_dim,
                -1,
            )
            .transpose(1, 2)
        )
        k = (
            self.k_proj(
                x,
            )
            .view(
                batch_size,
                seq_len,
                self.head_dim,
                -1,
            )
            .transpose(1, 2)
        )
        v = (
            self.v_proj(
                x,
            )
            .view(
                batch_size,
                seq_len,
                self.head_dim,
                -1,
            )
            .transpose(1, 2)
        )

        attn_weights = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, input_dim)
        )

        attn_output = self.out_proj(attn_output)

        if self.residual:
            x = residual + attn_output
        else:
            x = attn_output

        residual = x
        x = self.norm2(x)

        x = self.ff_linear1(x)
        x = self.activation(x)
        x = self.ff_dropout(x)
        x = self.ff_linear2(x)

        if self.residual:
            delta = x
            x = residual + delta
        else:
            x = x

        if not is_3d_input:
            x = x.squeeze(1)

        return x

    @property
    def param_count(self) -> int:
        """
        Get the total number of parameters in the adapter.

        :return: Number of parameters
        """
        return sum(p.numel() for p in self.parameters())


class EncoderWithAdapter(nn.Module):

    def __init__(self, original_encoder, embedding_adapter):
        """
        Wrapper that applies an adapter to the output of an encoder.

        :param original_encoder: The original encoder module
        :param embedding_adapter: The adapter module to apply
        """
        super().__init__()
        self.original_encoder = original_encoder
        self.adapter = embedding_adapter

        self.projection_dim = getattr(original_encoder, "projection_dim", 0)

        if hasattr(original_encoder, "question_encoder"):
            self.question_encoder = original_encoder.question_encoder

        if hasattr(original_encoder, "bert_model"):
            self.bert_model = original_encoder.bert_model

        for attr in ["config", "encode_proj", "embeddings_size"]:
            if hasattr(original_encoder, attr):
                setattr(self, attr, getattr(original_encoder, attr))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[DPRQuestionEncoderOutput, Tuple[torch.Tensor, ...]]:
        """
        Forward pass applying the adapter to the encoder output.

        :param input_ids: Input token IDs
        :param attention_mask: Attention mask for padding tokens
        :param token_type_ids: Token type IDs
        :param inputs_embeds: Pre-computed input embeddings
        :param output_attentions: Whether to return attention weights
        :param output_hidden_states: Whether to return hidden states
        :param return_dict: Whether to return a ModelOutput instead of tuple
        :return: Model outputs with adapted pooler output
        """

        outputs = self.original_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if hasattr(outputs, "pooler_output"):
            adapted_pooler = outputs.pooler_output

            if not return_dict:
                return (adapted_pooler,) + outputs[1:]

            if (
                hasattr(
                    outputs,
                    "__class__",
                )
                and "DPR" in outputs.__class__.__name__
            ):
                return type(outputs)(
                    pooler_output=adapted_pooler,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return BaseModelOutputWithPooling(
                    last_hidden_state=None,
                    pooler_output=adapted_pooler,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            if isinstance(outputs, tuple):
                sequence_output = outputs[0] if len(outputs) > 0 else None
                pooled_output = outputs[1] if len(outputs) > 1 else None
                rest_outputs = outputs[2:] if len(outputs) > 2 else tuple()
            else:
                sequence_output = getattr(outputs, "last_hidden_state", None)
                pooled_output = getattr(outputs, "pooler_output", None)
                rest_outputs = tuple()

            if pooled_output is not None:
                adapted_pooler = outputs.pooler_output

                if not return_dict:
                    return (sequence_output, adapted_pooler) + rest_outputs

                return BaseModelOutputWithPooling(
                    last_hidden_state=sequence_output,
                    pooler_output=adapted_pooler,
                    hidden_states=getattr(outputs, "hidden_states", None),
                    attentions=getattr(outputs, "attentions", None),
                )

            return outputs


def transform_embeddings(
    embeddings: np.ndarray,
    adapter: nn.Module,
    batch_size: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """
    Apply adapter transformation to a batch of embeddings.

    :param embeddings: Array of embeddings to transform
    :param adapter: Adapter module to apply
    :param batch_size: Batch size for processing
    :param device: Device to run computation on
    :return: Transformed embeddings array
    """
    adapter.eval()
    transformed = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(
                embeddings[i : i + batch_size],
                device=device,
                dtype=torch.float32,
            )
            transformed_batch = adapter(batch).cpu().numpy()
            transformed.append(transformed_batch)

    return np.vstack(transformed)


def update_index_process(
    shard_dir: str,
    index_path: str,
    adaptr_pth: str,
    _: int = 384,
    batch_size: int = 512,
):
    """
    Update FAISS index w/ adapter-transformed embeddings in a sep process.

    :param shard_dir: Directory containing dataset shards
    :param index_path: Path to save the updated index
    :param adaptr_pth: Path to the adapter model state
    :param _: Hidden dimension for adapter configuration
    :param batch_size: Batch size for processing
    """
    try:
        import faiss
        import torch
        from datasets import load_from_disk

        device = "cuda" if torch.cuda.is_available() else "cpu"
        adapter = TransformerEmbeddingAdapter()
        adapter.load_state_dict(torch.load(adaptr_pth))
        adapter.to(device)
        adapter.eval()

        logger.info(f"Loading dataset from: {shard_dir}/data_0")
        dataset = load_from_disk(f"{shard_dir}/data_0")

        logger.info("Transforming embeddings with adapter")
        original_embeddings = dataset["embeddings"]

        logger.info(f"Original embeddings type: {type(original_embeddings)}")
        if hasattr(original_embeddings, "shape"):
            logger.info(
                f"Original embeddings shape: {original_embeddings.shape}",
            )

        transformed_embeddings = []

        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_embeddings = np.array(
                [emb for emb in original_embeddings[i:batch_end]]
            )

            batch_transformed = transform_embeddings(
                batch_embeddings, adapter, batch_size=batch_size, device=device
            )

            transformed_embeddings.extend([emb for emb in batch_transformed])

        logger.info(f"Transformed {len(transformed_embeddings)} embeddings")

        logger.info("Creating new dataset with transformed embeddings")
        new_dataset = dataset.remove_columns(["embeddings"])
        new_dataset = new_dataset.add_column(
            "embeddings",
            transformed_embeddings,
        )

        logger.info("Building FAISS index from transformed embeddings")
        faiss.omp_set_num_threads(96)
        index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
        new_dataset.add_faiss_index("embeddings", custom_index=index)
        logger.info(f"Saving index to {index_path}")
        new_dataset.get_index("embeddings").save(index_path)

        logger.info(f"Successfully updated index at {index_path}")

    except Exception as e:
        logger.error(f"Error in update_index_process: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise


class AdaptivePEFTGenerativeQAModule(PEFTGenerativeQAModule):

    def __init__(self, hparams, **kwargs):
        """
        Initialize Adaptive PEFT Generative QA Module with document adapter.

        :param hparams: Hyperparameters for model configuration
        :param kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(hparams, **kwargs)
        self.avg_loss = 0

        if self.is_rag_model:
            self.adapter_config = {
                "hidden_dim": getattr(hparams, "adapter_hidden_dim", 384),
                "dropout": getattr(hparams, "adapter_dropout", 0.1),
                "layer_norm": getattr(hparams, "adapter_layer_norm", True),
                "residual": getattr(hparams, "adapter_residual", True),
                "reindex_frequency": getattr(
                    hparams,
                    "reindex_frequency",
                    750,
                ),
                "update_in_progress": False,
                "update_process": None,
            }
            self.embedding_adapter = TransformerEmbeddingAdapter()

            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.embedding_adapter.parameters():
                param.requires_grad = True

            logger.info("Base model frozen.")

            if isinstance(
                self.model.rag.question_encoder,
                DPRPrefixQuestionEncoder,
            ):

                self.model.rag.question_encoder = EncoderWithAdapter(
                    self.model.rag.question_encoder,
                    self.embedding_adapter,
                )

                self.model.question_encoder = EncoderWithAdapter(
                    self.model.question_encoder,
                    self.embedding_adapter,
                )
            else:

                self.model.rag.question_encoder = EncoderWithAdapter(
                    self.model.rag.question_encoder,
                    self.embedding_adapter,
                )

                self.model.question_encoder = EncoderWithAdapter(
                    self.model.question_encoder,
                    self.embedding_adapter,
                )

            self.enhance_retriever()

    def art_training_step(self, batch: dict, _: int) -> torch.Tensor:
        """
        Execute Alignment through Reconstruction Training step.

        :param batch: Dictionary containing the input batch data
        :param _: Index of the current batch
        :return: KL div loss between retrieval and reconstruction dists
        """
        source_ids, source_mask = batch["input_ids"], batch["attention_mask"]

        query_outputs = self.model.question_encoder(
            input_ids=source_ids, attention_mask=source_mask, return_dict=True
        )

        question_hidden_states = query_outputs.pooler_output

        retrieved_doc_embeds, retrieved_doc_ids, retrieved_doc_dicts = (
            self.model.rag.retriever.retrieve(question_hidden_states, n_docs=5)
        )

        batch_size = source_ids.size(0)
        all_recon_probs = []
        all_retrieval_logits = []

        for b_idx in range(batch_size):
            doc_dicts_item = retrieved_doc_dicts[b_idx]

            if "text" in doc_dicts_item and isinstance(
                doc_dicts_item["text"],
                list,
            ):
                passages = doc_dicts_item["text"]
            elif isinstance(doc_dicts_item, list):
                passages = doc_dicts_item
            else:
                print("aghhh!")
                exit(1)

            recon_scores = []
            for passage in passages:
                if isinstance(passage, str):
                    score = self.compute_question_reconstruction_score(
                        source_ids[b_idx : b_idx + 1], passage
                    )
                    recon_scores.append(score)
                else:
                    recon_scores.append(-float("inf"))

            recon_scores = torch.tensor(recon_scores, device=source_ids.device)
            recon_probs = F.softmax(recon_scores / 0.1, dim=0)
            all_recon_probs.append(recon_probs)

            retrieved_doc_embed = retrieved_doc_embeds[b_idx]
            query_emb = self.embedding_adapter(
                question_hidden_states[b_idx : b_idx + 1]
            )
            similarity = torch.matmul(
                query_emb, retrieved_doc_embed.transpose(0, 1)
            ).squeeze(0)
            all_retrieval_logits.append(similarity)

        all_recon_probs = torch.stack(all_recon_probs)
        all_retrieval_logits = torch.stack(all_retrieval_logits)
        retrieval_probs = F.softmax(all_retrieval_logits / 0.1, dim=1)

        kl_loss = F.kl_div(
            (retrieval_probs + 1e-10).log(),
            all_recon_probs,
            reduction="batchmean",
        )

        return kl_loss

    def compute_batch_reconstruction_scores(
        self, question_ids_batch: torch.Tensor, passage_texts_batch: list
    ) -> list:
        """
        Compute reconstruction scores for a batch of questions and passages.

        :param question_ids_batch: Batch of question token IDs
        :param passage_texts_batch: Batch of passage texts
        :return: List of reconstruction scores
        """
        passage_inputs = self.tokenizer(
            passage_texts_batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(question_ids_batch.device)

        batch_size = len(passage_texts_batch)

        decoder_input_ids = question_ids_batch[:, :-1].contiguous()
        question_ids_targets = question_ids_batch[:, 1:].clone()

        with torch.no_grad():
            encoder_outputs = self.model.rag.generator.get_encoder()(
                input_ids=passage_inputs.input_ids,
                attention_mask=passage_inputs.attention_mask,
                return_dict=True,
            )

            outputs = self.model.rag.generator(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )

            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            scores = []
            for i in range(batch_size):
                token_log_probs = []
                seq_len = min(
                    question_ids_targets.size(1) - 1,
                    log_probs.size(1),
                )

                for j in range(seq_len):
                    token_log_prob = log_probs[
                        i,
                        j,
                        question_ids_targets[i, j + 1],
                    ]
                    token_log_probs.append(token_log_prob)

                if token_log_probs:
                    score = sum(token_log_probs) / len(token_log_probs)
                    scores.append(score.item())
                else:
                    scores.append(-float("inf"))

        return scores

    def compute_batch_reconstruction_scores_for_retrieved_docs(
        self, question_ids: torch.Tensor, retrieved_doc_dicts: list
    ) -> list:
        """
        Compute reconstruction scores for retrieved documents.

        :param question_ids: Question token IDs
        :param retrieved_doc_dicts: Dictionary of retrieved documents
        :return: List of reconstruction scores tensors
        """
        batch_size = question_ids.size(0)
        all_passages = []
        n_docs_per_question = []
        question_ids_expanded = []

        for b_idx in range(batch_size):
            doc_dicts_item = retrieved_doc_dicts[b_idx]

            if "text" in doc_dicts_item and isinstance(
                doc_dicts_item["text"],
                list,
            ):
                passages = doc_dicts_item["text"]
            elif isinstance(doc_dicts_item, list):
                passages = doc_dicts_item
            else:
                logger.error(f"Unexpected structure: {type(doc_dicts_item)}")
                passages = ["Empty passage"]

            valid_passages = [p for p in passages if isinstance(p, str)]
            n_docs = len(valid_passages)

            n_docs_per_question.append(n_docs)

            all_passages.extend(valid_passages)

            question_ids_expanded.extend([question_ids[b_idx]] * n_docs)

        question_ids_stacked = torch.stack(question_ids_expanded)

        if all_passages:
            scores = self.compute_batch_reconstruction_scores(
                question_ids_stacked, all_passages
            )
        else:
            scores = []

        result = []
        start_idx = 0
        for n_docs in n_docs_per_question:
            if n_docs > 0:
                result.append(
                    torch.tensor(
                        scores[start_idx : start_idx + n_docs],
                        device=question_ids.device,
                    )
                )
            else:
                result.append(
                    torch.tensor([-float("inf")], device=question_ids.device),
                )
            start_idx += n_docs

        return result

    def training_step(self, batch: dict, batch_idx: int) -> Dict:
        """
        Execute a training step with periodic index updates.

        :param batch: Dictionary containing the input batch data
        :param batch_idx: Index of the current batch
        :return: Dictionary with loss and metrics
        """
        self.train_steps += 1
        s_ratio = self.train_steps % self.adapter_config["reindex_frequency"]
        if (
            self.is_rag_model
            and self.trainer.global_rank == 0
            and self.train_steps > 0
            and s_ratio == 0
            and not self.adapter_config["update_in_progress"]
        ):
            self._save_adapter_for_reindexing()
            self._start_reindex_process()

        if (
            self.adapter_config["update_in_progress"]
            and self.adapter_config["update_process"] is not None
            and not self.adapter_config["update_process"].is_alive()
        ):
            logger.info("Index update process completed")
            self._finalize_index_update()

        kl_loss = self._step(batch)[0]
        if self.train_steps % 4288 == 0:
            print(f"Avg loss: {self.avg_loss / 4288}")
            self.avg_loss = 0

        return kl_loss

    def _get_passages_from_doc_ids(self, doc_ids: list) -> list:
        """
        Retrieve passage texts from document IDs.

        :param doc_ids: List of document IDs
        :return: List of passage texts
        """
        passages = []

        for doc_id in doc_ids:
            doc = self.model.rag.retriever.index.dataset[doc_id]
            if "text" in doc:
                passages.append(doc["text"])
            else:
                passages.append("Empty passage")

        return passages

    def compute_question_reconstruction_score(
        self, question_ids: torch.Tensor, passage_text: str
    ) -> float:
        """
        Compute how well a passage can reconstruct a question.

        :param question_ids: Question token IDs
        :param passage_text: Passage text
        :return: Reconstruction score
        """
        passage_inputs = self.tokenizer(
            passage_text, return_tensors="pt", truncation=True, max_length=512
        ).to(question_ids.device)

        decoder_input_ids = question_ids[:, :-1].contiguous()
        question_ids = question_ids[:, 1:].clone()

        with torch.no_grad():
            encoder_outputs = self.model.rag.generator.get_encoder()(
                input_ids=passage_inputs.input_ids,
                attention_mask=passage_inputs.attention_mask,
                return_dict=True,
            )

            outputs = self.model.rag.generator(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )

            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            token_log_probs = []
            for i in range(question_ids.size(1) - 1):
                token_log_prob = log_probs[0, i, question_ids[0, i + 1]]
                token_log_probs.append(token_log_prob)

            score = (
                sum(token_log_probs) / len(token_log_probs)
                if token_log_probs
                else -float("inf")
            )
            return score.item()

    def _save_adapter_for_reindexing(self):
        """
        Save the current adapter state for reindexing process.
        """
        adapter_dir = self.output_dir / "temp_adapter"
        adapter_dir.mkdir(exist_ok=True, parents=True)

        adaptr_pth = adapter_dir / "adapter_latest.pt"
        torch.save(self.embedding_adapter.state_dict(), adaptr_pth)
        logger.info(f"Saved adapter state to {adaptr_pth}")

        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(
                {
                    "hidden_dim": self.adapter_config["hidden_dim"],
                    "train_steps": self.train_steps,
                },
                f,
            )

    def _start_reindex_process(self, adaptr_pth: Optional[str] = None):
        """
        Start an asynchronous reindexing process.

        :param adaptr_pth: Path to adapter model, defaults to latest
        """
        if adaptr_pth is None:
            adaptr_pth = self.output_dir / "temp_adapter" / "adapter_latest.pt"

        if not hasattr(self.custom_config, "shard_dir") or not hasattr(
            self.config, "index_path"
        ):
            logger.warning(
                "Missing shard_dir or index_path config, skipping reindexing"
            )
            return

        og_idx = Path(self.config.index_path)
        backup_index = og_idx.parent / f"{og_idx.stem}_backup{og_idx.suffix}"

        if not backup_index.exists() and og_idx.exists():
            logger.info(f"Creating backup of original index: {backup_index}")
            shutil.copy(og_idx, backup_index)

        temp_index = og_idx.parent / f"{og_idx.stem}_temp{og_idx.suffix}"

        logger.info(f"Starting index update @ step {self.train_steps}...")

        process = Process(
            target=update_index_process,
            args=(
                "conv_qa/knowledge_base/shards",
                str(temp_index),
                str(adaptr_pth),
                self.adapter_config["hidden_dim"],
            ),
        )
        process.start()

        self.adapter_config["update_in_progress"] = True
        self.adapter_config["update_process"] = process
        self.adapter_config["temp_index"] = temp_index
        self.adapter_config["og_idx"] = og_idx

    def _finalize_index_update(self):
        """
        Finalize the index update by swapping in the new index.
        """
        try:
            temp_index = self.adapter_config["temp_index"]
            og_idx = self.adapter_config["og_idx"]

            if os.path.exists(temp_index):
                if os.path.exists(og_idx):
                    os.remove(og_idx)
                shutil.copy(temp_index, og_idx)
                os.remove(temp_index)

                logger.info(f"Successfully updated index at {og_idx}")

                logger.info("Reloading retriever with updated index...")
                self.model.rag.retriever.re_load()
                self.model.rag.retriever.init_retrieval()
        except Exception as e:
            logger.error(f"Error finalizing index update: {str(e)}")
        finally:
            self.adapter_config["update_in_progress"] = False
            self.adapter_config["update_process"] = None

    def enhance_retriever(self):
        """
        Enhance retriever with adapter capabilities.
        """
        if not hasattr(self, "embedding_adapter"):
            logger.warning("No embedding adapter found.")
            return

        print(self.model.rag.retriever)
        original_retrieve = self.model.rag.retriever.retrieve
        self.model.rag.retriever._original_retrieve = original_retrieve
        self.model.rag.retriever.retrieve = types.MethodType(
            enhanced_retrieve, self.model.rag.retriever
        )
        self.model.rag.retriever.adapter = self.embedding_adapter
        logger.info(
            "Enhanced retriever w/ re-ranking capabilities & query adapter",
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Save model checkpoint with document adapter on rank zero process only.

        Saves the adapter separately from the base model components to enable
        flexible loading and combination of different adapters with models.

        :param self: Instance of AdaptivePEFTGenerativeQAModule
        :param checkpoint: Dictionary containing the checkpoint data
        :return: None
        """
        super().on_save_checkpoint(checkpoint)

        print("saving checkpoint!")

        save_path = self.output_dir.joinpath(f"checkpoint{self.step_count}")
        adapter_dir = save_path.joinpath("document_adapter")
        adapter_dir.mkdir(exist_ok=True, parents=True)

        torch.save(
            self.embedding_adapter.state_dict(),
            adapter_dir / "adapter_model.bin",
        )

        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(
                {
                    "hidden_dim": self.adapter_config["hidden_dim"],
                    "dropout": self.adapter_config["dropout"],
                    "layer_norm": self.adapter_config["layer_norm"],
                    "residual": self.adapter_config["residual"],
                    "train_steps": self.train_steps,
                },
                f,
            )

        model_config_path = save_path.joinpath("model_config.json")
        if model_config_path.exists():
            with open(model_config_path, "r") as f:
                model_config = json.load(f)

            if (
                "components" in model_config
                and "document_adapter" not in model_config["components"]
            ):
                model_config["components"].append("document_adapter")
            else:
                model_config["components"] = [
                    "generator",
                    "question_encoder",
                    "document_adapter",
                ]

            with open(model_config_path, "w") as f:
                json.dump(model_config, f, indent=2)
