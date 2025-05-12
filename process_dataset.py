import logging
import random
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Optional, List
from pathlib import Path
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from rag_end2end.kb_encode_utils import embed_update, add_index
from rag_end2end.utils_rag import Seq2SeqDataset as OriginalSeq2SeqDataset


class RandomSamplingSeq2SeqDataset(OriginalSeq2SeqDataset):
    """
    Class extends OriginalSeq2SeqDataset w/ stratified random sampling.

    :param tokenizer: Tokenizer instance for processing text data
    :param data_dir: Directory containing the dataset files
    :param max_source_length: Maximum length for source sequences
    :param max_target_length: Maximum length for target sequences
    :param type_path: Dataset split type ('train' or 'val')
    :param n_obs: Number of observations to sample (if None, use full dataset)
    :param src_lang: Source language code (optional)
    :param tgt_lang: Target language code (optional)
    :param prefix: Prefix to add to the input (optional)
    :param random_seed: Random seed for sampling validation data (default: 42)
    :param num_bins: Number of bins to use for stratified sampling (default: 5)
    """

    def __init__(
        self,
        tokenizer: Any,
        data_dir: str,
        max_source_length: int,
        max_target_length: int,
        type_path: str = "train",
        n_obs: Optional[int] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        prefix: str = "",
        random_seed: int = 42,
        num_bins: int = 5,
    ) -> None:

        self.src_lens: List[int]

        self.num_bins = num_bins
        self.index_map = None

        super().__init__(
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            type_path=type_path,
            n_obs=None,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            prefix=prefix,
        )

        self.tgt_file = Path(data_dir).joinpath(f"{type_path}.target")
        self.tgt_lens = self.get_char_lens(self.tgt_file)

        if n_obs is None:
            return

        if type_path == "val":
            random.seed(random_seed)
            indices = self.stratified_sample(n_obs)
            self.index_map = dict(enumerate(indices))
            self.src_lens = [self.src_lens[i] for i in indices]
        else:
            self.src_lens = self.src_lens[:n_obs]

    def stratified_sample(self, n_obs: int) -> List[int]:
        """
        Perform stratified sampling based on answer length distribution.

        :param n_obs: Number of observations to sample
        :return: List of sampled indices
        """
        if n_obs >= len(self.tgt_lens):
            return list(range(len(self.tgt_lens)))

        all_lengths = sorted(self.tgt_lens)
        bin_edges = [
            all_lengths[int(i * len(all_lengths) / self.num_bins)]
            for i in range(1, self.num_bins)
        ]

        bins: List[List[int]] = [[] for _ in range(self.num_bins)]
        for idx, length in enumerate(self.tgt_lens):
            bin_idx = self.determine_bin(length, bin_edges)
            bins[bin_idx].append(idx)

        indices_by_bin = {}
        remaining = n_obs

        for bin_idx, bin_indices in enumerate(bins):
            if not bin_indices:
                continue

            expected = max(
                1,
                int(n_obs * len(bin_indices) / len(self.tgt_lens)),
            )

            to_sample = min(len(bin_indices), expected)
            indices_by_bin[bin_idx] = random.sample(bin_indices, to_sample)
            remaining -= to_sample

        vals = indices_by_bin.values()

        if remaining > 0:
            sampled = set(idx for samples in vals for idx in samples)
            unsampled = list(set(range(len(self.tgt_lens))) - sampled)

            if unsampled:
                additional = random.sample(
                    unsampled,
                    min(remaining, len(unsampled)),
                )
                indices_by_bin.setdefault(-1, []).extend(additional)

        vals = indices_by_bin.values()

        return sorted(idx for samples in vals for idx in samples)

    @staticmethod
    def determine_bin(length: int, bin_edges: List[int]) -> int:
        """
        Determine which bin a sequence length belongs to.

        :param length: The sequence length
        :param bin_edges: List of bin edge values
        :return: Bin index
        """
        for i, edge in enumerate(bin_edges):
            if length <= edge:
                return i
        return len(bin_edges)

    def __getitem__(self, index: int) -> dict:
        """
        Get a single item from the dataset using the appropriate index mapping.

        For val data, uses the stored index mapping to maintain consistent
        sampling across epochs. For training data, uses direct indexing.

        :param index: Index of the item to retrieve
        :return: Dictionary containing the processed item data
        """
        return super().__getitem__(
            self.index_map.get(index, index) if self.index_map else index
        )


@dataclass
class KBConfig:
    """
    Configuration dataclass for knowledge base creation settings.

    Handles all configuration parameters needed for creating and managing
    a knowledge base, including paths, model settings, and processing options.

    :param name: Name identifier for the knowledge base
    :param input_path: Path to input data file
    :param output_dir: Directory where outputs will be saved
    :param model_name: Name/path of the DPR context encoder model
    :param device: Computing device for model execution
    :param num_processes: Number of parallel processes for encoding
    :param delimiter: Delimiter character for input file parsing
    :param column_names: Names of columns in input file
    """

    name: str
    input_path: Path
    output_dir: Path
    model_name: str = "facebook/dpr-ctx_encoder-single-nq-base"
    device: str = "cuda"
    num_processes: int = 1
    delimiter: str = "\t"
    column_names: tuple = ("title", "text")
    shard_dir: Path = field(init=False)
    index_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize derived paths and create necessary directories.

        Converts string paths to Path objects and creates output directories
        for shards and final index storage.

        :param self: Instance of KBConfig
        :return: None
        """
        self.input_path, self.output_dir = map(
            Path,
            (self.input_path, self.output_dir),
        )
        self.shard_dir = self.output_dir / "shards"
        self.index_path = self.output_dir / f"{self.name}.faiss"
        Path.mkdir(self.output_dir, parents=True, exist_ok=True)
        Path.mkdir(self.shard_dir, parents=True, exist_ok=True)


class CustomDPRTokenizer(DPRContextEncoderTokenizerFast):
    """
    Custom DPR tokenizer with fixed maximum length settings.

    Extends the base DPR tokenizer to enforce a consistent max length
    of 512 tokens for all inputs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the custom tokenizer with fixed max length.

        :param args: Positional arguments passed to parent class
        :param kwargs: Keyword arguments passed to parent class
        :return: None
        """
        super().__init__(*args, **kwargs)
        self.model_max_length = 512

    def __call__(self, *args: Any, **kwargs: Any) -> BatchEncoding:
        """
        Override call method to enforce max length constraint.

        :param args: Positional arguments for tokenization
        :param kwargs: Keyword arguments for tokenization
        :return: Tokenized output with enforced max length
        """
        kwargs["max_length"] = 512
        return super().__call__(*args, **kwargs)


class KnowledgeBaseCreator:
    """
    Main class for creating and managing knowledge bases using DPR encodings.

    Handles the complete pipeline of loading models, processing documents,
    creating embeddings, and building search indices.
    """

    def __init__(self, config: KBConfig) -> None:
        """
        Initialize the knowledge base creator with given configuration.

        Sets up logging, initializes the tokenizer, and prepares the encoder
        for document processing.

        :param config: Configuration object for knowledge base creation
        :return: None
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.encoder = None
        self.tokenizer = CustomDPRTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

    def process(self) -> None:
        """
        Create KB by encoding documents and building a FAISS index.

        :param self: Instance of KnowledgeBaseCreator
        :return: None
        :raises: Exception if any processing step fails
        """
        try:
            self.logger.info(
                f"Starting knowledge base creation: {self.config.name}",
            )

            if not self.encoder:
                self.logger.info(
                    f"Loading DPR context encoder: {self.config.model_name}"
                )
                self.encoder = DPRContextEncoder.from_pretrained(
                    self.config.model_name,
                )

            import types

            def patched_tokenizer(*_: Any, **__: Any) -> CustomDPRTokenizer:
                return self.tokenizer

            import rag_end2end.kb_encode_utils as utils

            tokenizer = utils.DPRContextEncoderTokenizerFast
            tokenizer.from_pretrained = types.MethodType(
                patched_tokenizer, utils.DPRContextEncoderTokenizerFast
            )

            num_processes = self.config.num_processes
            for process_num in range(num_processes):
                self.logger.info(f"Shard {process_num + 1}/{num_processes}")
                embed_update(
                    ctx_encoder=self.encoder,
                    total_processes=self.config.num_processes,
                    device=self.config.device,
                    process_num=process_num,
                    shard_dir=str(self.config.shard_dir),
                    csv_path=str(self.config.input_path),
                )

            self.logger.info("Creating FAISS index")
            add_index(str(self.config.shard_dir), str(self.config.index_path))
            self.logger.info(f"KB complete. Saved to {self.config.output_dir}")

        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            raise


if __name__ == "__main__":
    KnowledgeBaseCreator(
        KBConfig(
            name="covid_qa",
            input_path=Path("covid_qa/splitted_covid_dump.csv"),
            output_dir=Path("covid_qa/knowledge_base"),
        )
    ).process()
