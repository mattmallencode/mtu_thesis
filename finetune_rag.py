import inspect
import os
import sys
import random
import numpy as np
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from multiprocessing import set_start_method
from pathlib import Path
from typing import Any, Union


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across different libraries.

    :param seed: Integer value to use as the random seed
    :return: None
    """
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _clean_kwargs(kwargs: dict) -> dict:
    """
    Helper function to remove deprecated arguments from kwargs dictionary.

    :param kwargs: Dictionary containing keyword arguments to be cleaned
    :return: Dictionary with deprecated arguments removed
    """
    return {
        k: kwargs[k]
        for k in set(kwargs)
        - {
            "weights_summary",
            "filepath",
        }
    }


def from_argparse_args(
    cls: Any,
    args: Union[Namespace, dict],
    **kwargs: Any,
) -> Any:
    """
    Recreate the functionality of `from_argparse_args` for PyTorch Lightning.

    :param cls: Class to instantiate with the parsed arguments
    :param args: Args from ArgumentParser, either as Namespace or dict
    :param kwargs: Additional keyword arguments to override parsed arguments
    :return: Instance of cls initialized with the processed arguments
    """
    args_dict = vars(args) if isinstance(args, Namespace) else args
    valid_kwargs = {
        name: args_dict[name]
        for name in inspect.signature(cls.__init__).parameters
        if name in args_dict
    }

    merged_kwargs = {
        **_clean_kwargs(valid_kwargs),
        **_clean_kwargs(kwargs),
        "val_check_interval": 0.33,
    }

    return cls(**merged_kwargs)


def setup_paths() -> None:
    """
    Extends system path: include CWD and rag_end2end subdir.

    :return: None
    """
    sys.path.extend(
        str(p) for p in [Path.cwd(), Path.cwd() / "rag_end2end"] if p.exists()
    )
    sys.path.extend(
        str(
            p,
        )
        for p in [Path.cwd(), Path.cwd() / "doc_adaptation"]
        if p.exists()
    )


setup_paths()

import rag_end2end.utils_rag as utils_rag  # noqa: E402

from process_dataset import RandomSamplingSeq2SeqDataset  # noqa: E402
from peft_module import PEFTGenerativeQAModule  # noqa: E402

original_dataset = utils_rag.Seq2SeqDataset
utils_rag.Seq2SeqDataset = RandomSamplingSeq2SeqDataset

import rag_end2end.finetune_rag as finetune_rag  # noqa: E402

finetune_rag.Seq2SeqDataset = utils_rag.Seq2SeqDataset


def main(*arguments: Any) -> None:
    """
    Main entry point for the RAG fine-tuning process.

    :param arguments: Variable length argument list containing CLI args.
    :return: None
    """
    set_seed(arguments[0].seed)
    pl.Trainer.add_argparse_args = lambda arg_parser: arg_parser
    pl.Trainer.from_argparse_args = classmethod(from_argparse_args)
    finetune_rag.GenerativeQAModule = PEFTGenerativeQAModule
    # finetune_rag.GenerativeQAModule = AdaptivePEFTGenerativeQAModule
    finetune_rag.main(*arguments)


if __name__ == "__main__":

    set_start_method("spawn")
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use pytorch_lightning.profiler.AdvancedProfiler",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the saved checkpoint for testing.",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=None,
        help="How many model checkpoints to save.",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--doc_adapt_only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--prompt_only",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--shards_dir",
        type=str,
        default="conv_qa/knowledge_base/shards/data_0",
    )
    # QAModule = AdaptivePEFTGenerativeQAModule
    QAModule = PEFTGenerativeQAModule
    parser = QAModule.add_model_specific_args(parser, os.getcwd())
    parser = QAModule.add_retriever_specific_args(parser)
    parser = QAModule.add_ray_specific_args(parser)

    main(parser.parse_args())
