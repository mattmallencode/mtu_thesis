import argparse
import os
import sys
import json
import logging
import random
import torch
import re
import numpy as np
from typing import Dict, Any
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process
from transformers import RagTokenForGeneration, RagRetriever
from peft import PeftConfig, get_peft_model
from types import MethodType
from scipy import stats

sys.path.append(os.path.join(os.getcwd(), "rag_end2end"))
import rag_end2end.eval_rag as eval_rag  # noqa: E402
from rag_end2end.utils_rag import normalize_answer, f1_score  # noqa: E402

from p_tuningv2 import DPRPrefixQuestionEncoder  # noqa: E402
from peft_module import forward  # noqa: E402

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across different libraries.

    :param seed: Integer value to use as the random seed
    :return: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_peft_rag_model(model_path: str, **kwargs) -> RagTokenForGeneration:
    """
    Load a PEFT-enhanced RAG model from a checkpoint directory.

    :param model_path: Path to dir w/ model checkpoints & configuration
    :param kwargs: Additional kwargs to pass to the model constructor
    :return: loaded RagTokenForGeneration model w/ PEFT adaptations
    :raises ValueError: If model config not found at the specified path
    """

    if model_path == "facebook/rag-token-nq":
        logger.info("Loading baseline model (not PEFT-enhanced)")
        retriever = RagRetriever.from_pretrained(model_path, **kwargs)
        model = RagTokenForGeneration.from_pretrained(
            model_path,
            retriever=retriever,
        )
        model.config.n_docs = args.n_docs
        model.retriever.init_retrieval()
        return model

    config_path = os.path.join(model_path, "model_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Model config not found at {config_path}")

    base_model_path = kwargs.pop("base_model_path", "facebook/rag-token-nq")

    retriever = RagRetriever.from_pretrained(base_model_path, **kwargs)
    base_model = RagTokenForGeneration.from_pretrained(
        base_model_path, retriever=retriever
    )
    base_model.config.n_docs = args.n_docs
    base_model.retriever.init_retrieval()

    generator_path = os.path.join(model_path, "generator")
    if os.path.exists(generator_path):
        generator_config = PeftConfig.from_pretrained(generator_path)

        if os.path.exists(
            os.path.join(
                generator_path,
                "adapter_model.safetensors",
            )
        ):
            logger.info("Using generator adapter from safetensors file")

        base_model.rag.generator = get_peft_model(
            base_model.rag.generator, generator_config
        )

        base_model.rag.generator.load_adapter(generator_path, "default")

        base_model.rag.generator.forward = MethodType(
            forward,
            base_model.rag.generator,
        )

    qenc_path = os.path.join(model_path, "question_encoder")
    if os.path.exists(qenc_path):
        with open(os.path.join(qenc_path, "adapter_config.json"), "r") as f:
            prefix_config = json.load(f)

        config = base_model.rag.question_encoder.config

        if "lora_alpha" in prefix_config:
            logger.info("Loading LoRA adapter for question encoder")
            qenc_config = PeftConfig.from_pretrained(qenc_path)
            base_model.rag.question_encoder = get_peft_model(
                base_model.rag.question_encoder, qenc_config
            )
            base_model.rag.question_encoder.load_adapter(qenc_path, "default")
        else:
            logger.info("Loading prefix tuning adapter for question encoder")
            config.pre_seq_len = prefix_config.get("pre_seq_len")
            config.prefix_projection = prefix_config.get("prefix_projection")
            config.prefix_hidden_size = prefix_config.get("prefix_hidden_size")
            question_encoder = base_model.rag_question_encoder.question_encoder
            config.bert_model = question_encoder.bert_model
            custom_qenc = DPRPrefixQuestionEncoder(config)

            if os.path.exists(
                os.path.join(
                    qenc_path,
                    "adapter_model.safetensors",
                )
            ):
                logger.info("Loading question encoder adapter")
                from safetensors.torch import load_file

                qenc_state = load_file(
                    os.path.join(qenc_path, "adapter_model.safetensors")
                )
            elif os.path.exists(os.path.join(qenc_path, "adapter_model.bin")):
                logger.info("Loading question encoder adapter from bin file")
                qenc_state = torch.load(
                    os.path.join(
                        qenc_path,
                        "adapter_model.bin",
                    )
                )
            else:
                raise ValueError(f"No adapter weights found in {qenc_path}")

            custom_qenc.load_state_dict(qenc_state, strict=False)
            base_model.rag.question_encoder = custom_qenc

    return base_model


def efficient_fuzzy_match(
    passage: str,
    answer: str,
    threshold: float = 75.0,
) -> bool:
    """
    Efficiently check if answer appears in passage using fuzzy matching.

    :param passage: Normalized passage text to search in
    :param answer: Normalized answer text to search for
    :param threshold: Minimum similarity ratio (0-100) for fuzzy matching
    :return: True if answer is found in passage above threshold
    """
    if answer in passage:
        return True

    threshold = (
        max(
            threshold + 10,
            85.0,
        )
        if len(answer.split()) <= 2
        else threshold
    )

    return bool(
        (_ := default_process(answer))
        and fuzz.token_set_ratio(answer, passage, processor=default_process)
        >= threshold
    )


def patched_get_precision_at_k(
    _args: argparse.Namespace, preds_path: str, gold_data_path: str
) -> tuple[float, list[float]]:
    """
    Calculate precision@k w/ fuzzy matching against document content.

    :param _args: Namespace w/ eval params including k and kb_path
    :param preds_path: Path to file containing model predictions (document IDs)
    :param gold_data_path: Path to file containing gold standard answers
    :return: Tuple of (Precision@k, individual example results)
    """
    if not hasattr(_args, "kb_path") or not _args.kb_path:
        raise RuntimeError("kb_path is required for content-based precision@k")

    document_db: dict = {}
    logger.info(f"Loading knowledge base from {_args.kb_path}")

    try:
        with open(_args.kb_path) as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    doc_id, content = parts[0].strip(), parts[1].strip()

                    if doc_id in document_db:
                        document_db[doc_id] += " " + content
                    else:
                        document_db[doc_id] = content

        logger.info(f"Loaded {len(document_db)} docs into KB")
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        return 0.0, []

    with open(preds_path) as f, open(gold_data_path) as g:
        predictions = [p.strip() for p in f]
        gold_answers = [normalize_answer(g.strip()) for g in g]

    individual_results = []
    correct_retrievals = 0

    for pred, gold in zip(predictions, gold_answers):
        doc_ids = pred.split("\t")[: _args.k]

        is_correct = False
        for doc_id in doc_ids:
            if doc_id in document_db:

                doc_content = normalize_answer(document_db[doc_id])
                if efficient_fuzzy_match(doc_content, gold):
                    is_correct = True
                    break
            else:
                pass

        individual_results.append(1.0 if is_correct else 0.0)
        if is_correct:
            correct_retrievals += 1

    precision = 100.0 * correct_retrievals / len(predictions)

    logger.info(f"Number of questions evaluated: {len(predictions)}")
    logger.info(f"Number of correct retrievals: {correct_retrievals}")
    logger.info(f"Precision@{_args.k}: {precision:.2f}")

    return precision, individual_results


def bootstrap_significance_test(
    baseline_results: np.ndarray,
    comparison_seed_results: list[list[float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[np.floating[Any], tuple[np.floating[Any], np.floating[Any]], bool]:
    """
    Performs bootstrapped significance test comparing runs to baseline.

    :param baseline_results: Array of per-example metrics for baseline model
    :param comparison_seed_results: List of arrays; example metrics for run
    :param n_bootstrap: Number of bootstrap samples
    :param alpha: Significance level (default: 0.05 for 95% confidence)
    :return: Mean difference, conf interval, and if difference is significant
    """

    baseline_results = np.array(baseline_results)
    comparison_seed_results = np.array(comparison_seed_results)

    n_seeds = len(comparison_seed_results)
    n_examples = len(baseline_results)

    all_differences = comparison_seed_results - baseline_results.reshape(1, -1)

    direct_diff = np.mean(all_differences)

    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        example_indices = np.random.choice(
            n_examples,
            size=n_examples,
            replace=True,
        )

        seed_indices = np.random.choice(n_seeds, size=n_examples, replace=True)

        _ = np.arange(n_examples)

        bootstrap_sample = all_differences[seed_indices, example_indices]

        bootstrap_diff = np.mean(bootstrap_sample)
        bootstrap_diffs.append(bootstrap_diff)

    prop_less = np.mean(
        [1 if bd < direct_diff else 0 for bd in bootstrap_diffs],
    )
    z0 = stats.norm.ppf(prop_less)

    jackknife_diffs: list = []

    for i in range(n_examples):
        jack_sample = np.delete(all_differences, i, axis=1)
        jack_mean = np.mean(jack_sample)
        jackknife_diffs.append(jack_mean)

    jack_mean = np.mean(jackknife_diffs)
    numerator: np.number = np.sum((jack_mean - jackknife_diffs) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_diffs) ** 2) ** 1.5)

    if denominator == 0:
        a = 0
    else:
        a = numerator / denominator

    alpha1 = alpha / 2
    alpha2 = 1 - alpha / 2

    z1 = stats.norm.ppf(alpha1)
    z2 = stats.norm.ppf(alpha2)

    adjusted_alpha1 = stats.norm.cdf(z0 + (z0 + z1) / (1 - a * (z0 + z1)))
    adjusted_alpha2 = stats.norm.cdf(z0 + (z0 + z2) / (1 - a * (z0 + z2)))

    adjusted_pct1 = adjusted_alpha1 * 100
    adjusted_pct2 = adjusted_alpha2 * 100

    ci_lower = np.percentile(bootstrap_diffs, adjusted_pct1)
    ci_upper = np.percentile(bootstrap_diffs, adjusted_pct2)

    significant = not (ci_lower <= 0 <= ci_upper)

    return direct_diff, (ci_lower, ci_upper), significant


def patch_get_args() -> None:
    """
    This function replaces the original get_args function w/ extra args.
    Also adds support for PEFT-specific arguments.

    :return: None
    """

    def patched_get_args() -> argparse.Namespace:
        """
        Create and configure the argument parser with all necessary parameters.

        :return: Parsed command line arguments
        """
        parser = eval_rag.argparse.ArgumentParser(
            description="Evaluation script for RAG models."
        )

        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "bart"],
            type=str,
            help=(
                "RAG model type: rag_sequence, rag_token or bart, if none "
                "specified, the type is"
                "inferred from the"
                " model_name_or_path"
            ),
        )
        parser.add_argument(
            "--metric",
            type=str,
        )
        parser.add_argument(
            "--kb_path",
            type=str,
        )
        parser.add_argument(
            "--index_name",
            default=None,
            choices=[
                "exact",
                "compressed",
                "legacy",
                "custom",
            ],
            type=str,
            help="RAG model retriever type",
        )

        parser.add_argument(
            "--compare_models",
            action="store_true",
            help="Perform bootstrap significance testing?",
        )
        parser.add_argument(
            "--baseline_model_path",
            type=str,
            default="facebook/rag-token-nq",
            help="Path to the baseline model for statistical comparison",
        )
        parser.add_argument(
            "--comparison_model_paths",
            type=str,
            nargs="+",
            help="Paths to the comparison models with different seeds",
        )
        parser.add_argument(
            "--bootstrap_samples",
            type=int,
            default=1000,
            help="Number of bootstrap samples for significance testing",
        )

        parser.add_argument(
            "--passages_path",
            type=str,
            default=None,
            help="Path to the dataset of passages for custom index.",
        )
        parser.add_argument(
            "--index_path",
            default=None,
            type=str,
            help="Path to the retrieval index",
        )
        parser.add_argument(
            "--n_docs", default=5, type=int, help="Number of retrieved docs"
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=False,
            help="Path to pretrained checkpoints or model identifier",
        )

        parser.add_argument(
            "--is_peft_model",
            action="store_true",
            default=True,
            help="Whether the model is a PEFT-enhanced model",
        )
        parser.add_argument(
            "--base_model_path",
            default="facebook/rag-token-nq",
            type=str,
            help="Path to the base model (before PEFT)",
        )

        parser.add_argument(
            "--eval_mode",
            choices=["e2e", "retrieval"],
            default="e2e",
            type=str,
            help=(
                "e2e calculates exact match and F1 of the downstream task, "
                "retrieval calculates"
                " precision@k."
            ),
        )
        parser.add_argument(
            "--k", default=1, type=int, help="k for precision@k calculation"
        )
        parser.add_argument(
            "--evaluation_set",
            default=None,
            type=str,
            required=True,
            help="Path to a file containing evaluation samples",
        )
        parser.add_argument(
            "--gold_data_path",
            default=None,
            type=str,
            required=True,
            help="Path to a tab-separated file with gold samples",
        )
        parser.add_argument(
            "--gold_data_mode",
            default="qa",
            type=str,
            choices=["qa", "ans"],
            help="Format of the gold data file",
        )
        parser.add_argument(
            "--predictions_path",
            type=str,
            default="predictions.txt",
            help="Name of the predictions file",
        )

        parser.add_argument(
            "--eval_all_checkpoints",
            action="store_true",
            help="Evaluate all checkpoints starting w/ "
            "same prefix as model_name ending"
            "and ending with step number",
        )
        parser.add_argument(
            "--eval_batch_size",
            default=2,
            type=int,
            help="Batch size per GPU/CPU for evaluation.",
        )
        parser.add_argument(
            "--recalculate",
            help="Recalculate predictions even if the prediction file exists",
            action="store_true",
        )

        parser.add_argument(
            "--num_beams",
            default=4,
            type=int,
            help="Number of beams to be used when generating answers",
        )
        parser.add_argument(
            "--min_length",
            default=1,
            type=int,
            help="Min length of the generated answers",
        )
        parser.add_argument(
            "--max_length",
            default=50,
            type=int,
            help="Max length of the generated answers",
        )

        parser.add_argument(
            "--print_predictions",
            action="store_true",
            help="If True, prints predictions while evaluating.",
        )
        parser.add_argument(
            "--print_docs",
            action="store_true",
            help="If True, prints docs retried while generating.",
        )

        parsed_args = parser.parse_args()
        parsed_args.device = eval_rag.torch.device(
            "cuda" if eval_rag.torch.cuda.is_available() else "cpu"
        )
        return parsed_args

    eval_rag.get_args = patched_get_args


def patched_get_scores(
    _args: argparse.Namespace, preds_path: str, gold_data_path: str
) -> tuple:
    """
    Wrapper for eval_rag.get_scores that returns individual results.

    :param _args: Namespace with evaluation parameters including metric choice
    :param preds_path: Path to file containing model predictions (answers)
    :param gold_data_path: Path to file containing gold standard answers
    :return: Tuple of (overall score, list of individual example scores)
    """

    with open(preds_path) as f, open(gold_data_path) as g:
        predictions = [p.strip() for p in f]
        gold_answers = [g.strip() for g in g]

    metric = (
        getattr(
            _args,
            "metric",
            "em",
        ).lower()
        if hasattr(_args, "metric")
        else "em"
    )

    if metric == "f1":
        individual_results = []
        total_f1 = 0
        for pred, gold in zip(predictions, gold_answers):
            f1 = f1_score(pred, gold)
            individual_results.append(f1)
            total_f1 += f1

        score = 100.0 * total_f1 / len(predictions) if predictions else 0
        logger.info(f"Manually calculated F1: {score:.2f}")
    else:
        individual_results = []
        correct = 0.0
        for pred, gold in zip(predictions, gold_answers):
            norm_pred = normalize_answer(pred)
            norm_gold = normalize_answer(gold)
            is_correct = 1.0 if norm_pred == norm_gold else 0.0
            individual_results.append(is_correct)
            correct += is_correct

        score = 100.0 * correct / len(predictions) if predictions else 0
        logger.info(f"Manually calculated EM: {score:.2f}")
    with open(preds_path) as f, open(gold_data_path) as g:
        predictions = [p.strip() for p in f]
        gold_answers = [g.strip() for g in g]

    metric = (
        getattr(
            _args,
            "metric",
            "em",
        ).lower()
        if hasattr(_args, "metric")
        else "em"
    )

    if metric == "f1":
        individual_results = []
        total_f1 = 0
        for pred, gold in zip(predictions, gold_answers):
            f1 = f1_score(pred, gold)
            individual_results.append(f1)
            total_f1 += f1

        score = 100.0 * total_f1 / len(predictions) if predictions else 0
        logger.info(f"Manually calculated F1: {score:.2f}")
    else:
        individual_results = []
        correct = 0
        for pred, gold in zip(predictions, gold_answers):
            norm_pred = normalize_answer(pred)
            norm_gold = normalize_answer(gold)
            is_correct = 1.0 if norm_pred == norm_gold else 0.0
            individual_results.append(is_correct)
            correct += is_correct

        score = 100.0 * correct / len(predictions) if predictions else 0
        logger.info(f"Manually calculated EM: {score:.2f}")
    return score, individual_results


def patch_main() -> None:
    """
    Patch main function to handle custom passages and PEFT models.

    :return: None
    """

    def patched_main(main_args: argparse.Namespace) -> None:
        """
        Execute the main evaluation loop with the provided arguments.

        :param main_args: CLI args for model config and evaluation
        :return: None
        """
        model_kwargs: Dict[str, Any] = {}

        if (
            not hasattr(
                main_args,
                "is_peft_model",
            )
            or not main_args.is_peft_model
        ):
            config_path = os.path.join(
                main_args.model_name_or_path, "model_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    try:
                        config = json.load(f)
                        if (
                            config.get("model_type")
                            == "peft_rag_with_custom_prefix_tuning"
                        ):
                            main_args.is_peft_model = True
                            eval_rag.logger.info(
                                "Detected PEFT-enhanced RAG model",
                            )
                    except json.JSONDecodeError:
                        pass

        if main_args.model_type is None:
            main_args.model_type = eval_rag.infer_model_type(
                main_args.model_name_or_path
            )
            assert main_args.model_type is not None

        if main_args.model_type.startswith("rag"):
            model_class = (
                eval_rag.RagTokenForGeneration
                if main_args.model_type == "rag_token"
                else eval_rag.RagSequenceForGeneration
            )

            model_kwargs["n_docs"] = main_args.n_docs
            if main_args.index_name is not None:
                model_kwargs["index_name"] = main_args.index_name
            if main_args.index_path is not None:
                model_kwargs["index_path"] = main_args.index_path
            if main_args.passages_path is not None:
                model_kwargs["passages_path"] = main_args.passages_path
            if (
                hasattr(
                    main_args,
                    "base_model_path",
                )
                and main_args.base_model_path
            ):
                model_kwargs["base_model_path"] = main_args.base_model_path
        else:
            model_class = eval_rag.BartForConditionalGeneration
        checkpoints = (
            [
                f.path
                for f in os.scandir(
                    main_args.model_name_or_path,
                )
                if f.is_dir() and not f.name.startswith(".")
            ]
            if main_args.eval_all_checkpoints
            else [main_args.model_name_or_path]
        )

        eval_rag.logger.info("Evaluate following chkpnts: %s", checkpoints)
        eval_rag.get_precision_at_k = patched_get_precision_at_k

        score_fn = (
            patched_get_scores
            if main_args.eval_mode == "e2e"
            else patched_get_precision_at_k
        )
        evaluate_batch_fn = (
            eval_rag.evaluate_batch_e2e
            if main_args.eval_mode == "e2e"
            else eval_rag.evaluate_batch_retrieval
        )

        if not main_args.compare_models:
            for checkpoint in checkpoints:
                if os.path.exists(main_args.predictions_path) and (
                    not main_args.recalculate
                ):
                    eval_rag.logger.info(
                        "Calc metrics w/ existing predictions file: {}".format(
                            main_args.predictions_path
                        )
                    )
                    score_fn(
                        main_args,
                        main_args.predictions_path,
                        main_args.gold_data_path,
                    )
                    continue

                eval_rag.logger.info(
                    "***** Running evaluation for {} *****".format(checkpoint)
                )
                eval_rag.logger.info(
                    "  Batch size = %d",
                    main_args.eval_batch_size,
                )
                eval_rag.logger.info(
                    "  Predictions will be stored under {}".format(
                        main_args.predictions_path
                    )
                )

                if main_args.is_peft_model:
                    eval_rag.logger.info("Loading PEFT-enhanced RAG model...")
                    model = load_peft_rag_model(checkpoint, **model_kwargs)
                elif main_args.model_type.startswith("rag"):
                    retriever = eval_rag.RagRetriever.from_pretrained(
                        checkpoint, **model_kwargs
                    )
                    model = model_class.from_pretrained(
                        checkpoint, retriever=retriever, **model_kwargs
                    )
                    model.retriever.init_retrieval()
                else:
                    model = model_class.from_pretrained(
                        checkpoint,
                        **model_kwargs,
                    )
                model.to(main_args.device)

                with open(main_args.evaluation_set, "r") as eval_file, open(
                    main_args.predictions_path, "w"
                ) as preds_file:
                    lines = eval_file.read().splitlines()
                    eval_file.seek(0)

                    questions = []
                    for line in eval_rag.tqdm(
                        lines,
                        total=len(lines),
                        desc="Evaluating",
                    ):
                        questions.append(line.strip())
                        if len(questions) == main_args.eval_batch_size:
                            answers = evaluate_batch_fn(
                                main_args,
                                model,
                                questions,
                            )
                            preds_file.write("\n".join(answers) + "\n")
                            preds_file.flush()
                            questions = []

                    if questions:
                        answers = evaluate_batch_fn(
                            main_args,
                            model,
                            questions,
                        )
                        preds_file.write("\n".join(answers))
                        preds_file.flush()

                    score_fn(
                        main_args,
                        main_args.predictions_path,
                        main_args.gold_data_path,
                    )
        else:
            logger.info("Performing bootstrap significance testing...")

            if main_args.eval_mode == "e2e":
                base_preds_pth = f"preds_baseline_{main_args.eval_mode}.txt"
            else:
                base_preds_pth = (
                    f"preds_baseline_{main_args.eval_mode}_k{main_args.k}.txt"
                )
            original_predictions_path = main_args.predictions_path
            original_model_path = main_args.model_name_or_path

            main_args.predictions_path = base_preds_pth
            main_args.model_name_or_path = main_args.baseline_model_path

            if not os.path.exists(base_preds_pth) or main_args.recalculate:

                logger.info(
                    f"Evaluating baseline: {main_args.baseline_model_path}",
                )

                if main_args.baseline_model_path != "facebook/rag-token-nq":
                    model = load_peft_rag_model(
                        main_args.baseline_model_path, **model_kwargs
                    )
                elif main_args.model_type.startswith("rag"):
                    retriever = eval_rag.RagRetriever.from_pretrained(
                        main_args.baseline_model_path, **model_kwargs
                    )
                    model = model_class.from_pretrained(
                        main_args.baseline_model_path,
                        retriever=retriever,
                        **model_kwargs,
                    )
                    model.retriever.init_retrieval()
                else:
                    model = model_class.from_pretrained(
                        main_args.baseline_model_path, **model_kwargs
                    )

                model.to(main_args.device)

                with open(main_args.evaluation_set, "r") as eval_file, open(
                    base_preds_pth, "w"
                ) as preds_file:
                    lines = eval_file.read().splitlines()
                    eval_file.seek(0)

                    questions = []
                    for line in eval_rag.tqdm(
                        lines,
                        total=len(lines),
                        desc="Evaluating baseline model",
                    ):
                        questions.append(line.strip())
                        if len(questions) == main_args.eval_batch_size:
                            answers = evaluate_batch_fn(
                                main_args,
                                model,
                                questions,
                            )
                            preds_file.write("\n".join(answers) + "\n")
                            preds_file.flush()
                            questions = []

                    if questions:
                        answers = evaluate_batch_fn(
                            main_args,
                            model,
                            questions,
                        )
                        preds_file.write("\n".join(answers))
                        preds_file.flush()

            _, baseline_results = score_fn(
                main_args,
                base_preds_pth,
                main_args.gold_data_path,
            )

            logger.info("Baseline evaluation complete.")

            comparison_results = {}

            for model_path in main_args.comparison_model_paths:
                model_name = os.path.basename(model_path)
                logger.info(f"Evaluating comparison model: {model_name}")

                seed_results = []

                all_files = os.listdir(model_path)
                seed_pattern = re.compile(r"seed_(\d+)_([^.]+)\.txt")
                seed_files: dict = {}

                for file in all_files:
                    match = seed_pattern.match(file)
                    if match:
                        seed_num = match.group(1)
                        eval_type = match.group(2)

                        if seed_num not in seed_files:
                            seed_files[seed_num] = {}

                        seed_files[seed_num][eval_type] = file

                if not seed_files:
                    logger.warning(
                        f"No seed files found for {model_name}.",
                    )
                    continue

                logger.info(
                    f"Found {len(seed_files)} seeds for {model_name}.",
                )

                for seed_num, files in seed_files.items():

                    if main_args.eval_mode == "e2e" and "e2e" in files:
                        pred_file = files["e2e"]
                    elif (
                        main_args.eval_mode == "retrieval"
                        and f"retrieval_k{main_args.k}" in files
                    ):
                        pred_file = files[f"retrieval_k{main_args.k}"]
                    else:
                        logger.warning(
                            f"No {main_args.eval_mode} file for {seed_num}",
                        )
                        continue

                    predictions_path = os.path.join(model_path, pred_file)

                    if not os.path.exists(predictions_path):
                        logger.warning(
                            f"Predictions file not found: {predictions_path}"
                        )
                        continue

                    logger.info(f"Using predictions from: {predictions_path}")

                    _, individual_results = score_fn(
                        main_args,
                        predictions_path,
                        main_args.gold_data_path,
                    )

                    seed_results.append(individual_results)

                if len(seed_results) > 0:
                    diff, ci, significant = bootstrap_significance_test(
                        baseline_results,
                        seed_results,
                        n_bootstrap=main_args.bootstrap_samples,
                    )

                    comparison_results[model_name] = {
                        "mean_difference": float(diff),
                        "confidence_interval": (float(ci[0]), float(ci[1])),
                        "significant": bool(significant),
                    }

                    logger.info(
                        f"\nStat comparison: {model_name} vs baseline:",
                    )
                    logger.info(f"Mean difference: {diff:.4f}")
                    logger.info(
                        f"95% confidence interval: ({ci[0]:.4f}, {ci[1]:.4f})",
                    )
                    logger.info(f"Statistically significant: {significant}")

            main_args.model_name_or_path = original_model_path
            main_args.predictions_path = original_predictions_path

            if main_args.eval_mode == "e2e":
                res_pth = f"boot_{main_args.eval_mode}_{main_args.metric}.json"
            else:
                res_pth = f"boot_{main_args.eval_mode}_{main_args.k}.json"
            with open(res_pth, "w") as f:
                json.dump(comparison_results, f, indent=2)

            logger.info(f"Bootstrap testing done. Results saved to {res_pth}")

    eval_rag.main = patched_main


def apply_patches() -> None:
    """
    Apply all patches to the eval_rag module.

    :return: None
    """
    patch_get_args()
    patch_main()


if __name__ == "__main__":
    set_seed(42)
    apply_patches()
    args = eval_rag.get_args()
    eval_rag.main(args)
