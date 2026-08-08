"""Pinned UD acquisition and leakage-safe CoNLL-U preparation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from .artifacts import (
    ArtifactIntegrityError,
    atomic_output_path,
    atomic_write_json,
    canonical_json_bytes,
    copy_stream_and_hash,
    sha256_file,
    strict_resume,
    write_manifest,
)
from .config import UPOS_TAGS

FORM_HASH_ALGORITHM = "sha256-canonical-json-ordered-forms-v1"
DATASET_MANIFEST_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class DataError(RuntimeError):
    """Base class for acquisition and preparation errors."""


class DownloadIntegrityError(DataError):
    """Downloaded bytes do not match their pinned SHA-256."""


class ConlluFormatError(DataError):
    """A CoNLL-U source is malformed for this task."""


class DataLeakageError(DataError):
    """A split or serialized artifact violates an evaluation boundary."""


@dataclass(frozen=True, slots=True)
class SentenceProvenance:
    """Stable location of a sentence in one source file."""

    source_file: str
    split: str
    sentence_index: int
    sent_id: str | None


@dataclass(frozen=True, slots=True)
class TokenProvenance:
    """Stable location of a syntactic token in one source file."""

    conllu_id: int
    line_number: int


@dataclass(frozen=True, slots=True)
class Token:
    """An unlabeled syntactic token; deliberately has no UPOS field."""

    form: str
    provenance: TokenProvenance


@dataclass(frozen=True, slots=True)
class LabeledToken(Token):
    """A syntactic token carrying a gold UPOS label."""

    upos: str


@dataclass(frozen=True, slots=True)
class Sentence:
    """One parsed sentence with exact comments and source provenance."""

    tokens: tuple[Token, ...]
    provenance: SentenceProvenance
    comments: tuple[str, ...]

    @property
    def forms(self) -> tuple[str, ...]:
        return tuple(token.form for token in self.tokens)

    @property
    def form_hash(self) -> str:
        return ordered_form_hash(self.forms)


@dataclass(frozen=True, slots=True)
class PinnedDownload:
    """One immutable remote file and its expected local bytes."""

    url: str
    destination: Path
    sha256: str


def _validate_sha256(value: str) -> str:
    value = value.lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"expected a 64-character SHA-256, got {value!r}")
    return value


def download_pinned(
    url: str,
    destination: str | Path,
    sha256: str,
    *,
    timeout: float = 120.0,
) -> Path:
    """Download immutable bytes atomically and verify their pinned digest.

    A valid existing destination is reused. An invalid existing destination is
    never overwritten automatically because that would hide cache corruption.
    """
    destination = Path(destination)
    expected = _validate_sha256(sha256)
    if destination.exists():
        if not destination.is_file():
            raise DownloadIntegrityError(f"download destination is not a file: {destination}")
        actual = sha256_file(destination)
        if actual != expected:
            raise DownloadIntegrityError(
                f"cached SHA-256 mismatch for {destination}: expected {expected}, got {actual}"
            )
        return destination

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sanna-tagging-data/2.0"},
    )
    try:
        with atomic_output_path(destination) as temporary:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as output:
                    actual, _ = copy_stream_and_hash(response, output)
                    output.flush()
            if actual != expected:
                raise DownloadIntegrityError(
                    f"downloaded SHA-256 mismatch for {url}: expected {expected}, got {actual}"
                )
    except DownloadIntegrityError:
        raise
    except Exception as error:
        raise DataError(f"failed to download pinned file {url}") from error
    return destination


def _ud_url(repository: str, revision: str, filename: str) -> str:
    if not repository.startswith("UD_") or "/" in repository:
        raise ValueError(f"invalid Universal Dependencies repository: {repository!r}")
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError(f"UD revision must be a full 40-character commit: {revision!r}")
    return (
        "https://raw.githubusercontent.com/UniversalDependencies/"
        f"{quote(repository)}/{revision}/{quote(filename)}"
    )


def pinned_corpus_files(
    config: Mapping[str, Any], raw_root: str | Path
) -> dict[str, dict[str, tuple[PinnedDownload, ...]]]:
    """Expand pinned corpus config into deterministic download specifications."""
    raw_root = Path(raw_root)
    result: dict[str, dict[str, tuple[PinnedDownload, ...]]] = {}
    corpora = config["data"]
    for language in ("czech", "slovak"):
        corpus = corpora[language]
        repository = str(corpus["repository"])
        revision = str(corpus["revision"])
        language_splits: dict[str, tuple[PinnedDownload, ...]] = {}
        for split in ("train", "dev", "test"):
            entries = corpus.get(split, ())
            downloads = []
            for entry in entries:
                filename = str(entry["name"])
                if Path(filename).name != filename:
                    raise ValueError(f"corpus filename must be a basename: {filename!r}")
                digest = _validate_sha256(str(entry["sha256"]))
                destination = raw_root / language / repository / revision / filename
                downloads.append(
                    PinnedDownload(
                        url=_ud_url(repository, revision, filename),
                        destination=destination,
                        sha256=digest,
                    )
                )
            language_splits[split] = tuple(downloads)
        result[language] = language_splits
    return result


def download_corpora(
    config: Mapping[str, Any], raw_root: str | Path, *, timeout: float = 120.0
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Acquire all configured UD files with atomic, pinned downloads."""
    downloaded: dict[str, dict[str, tuple[Path, ...]]] = {}
    for language, splits in pinned_corpus_files(config, raw_root).items():
        downloaded[language] = {}
        for split, specifications in splits.items():
            downloaded[language][split] = tuple(
                download_pinned(
                    specification.url,
                    specification.destination,
                    specification.sha256,
                    timeout=timeout,
                )
                for specification in specifications
            )
    return downloaded


def _sent_id(comments: Sequence[str]) -> str | None:
    for comment in comments:
        match = re.match(r"^#\s*sent_id\s*=\s*(.*?)\s*$", comment)
        if match:
            return match.group(1)
    return None


def _parse_conllu(
    path: str | Path,
    *,
    split: str,
    source_file: str | None,
    labeled: bool,
) -> tuple[Sentence, ...]:
    path = Path(path)
    stable_source = source_file or path.name
    sentences: list[Sentence] = []
    comments: list[str] = []
    tokens: list[Token] = []
    sentence_index = 0
    block_has_content = False

    def finish_sentence() -> None:
        nonlocal comments, tokens, sentence_index, block_has_content
        if not block_has_content:
            return
        if tokens:
            sentences.append(
                Sentence(
                    tokens=tuple(tokens),
                    provenance=SentenceProvenance(
                        source_file=stable_source,
                        split=split,
                        sentence_index=sentence_index,
                        sent_id=_sent_id(comments),
                    ),
                    comments=tuple(comments),
                )
            )
        sentence_index += 1
        comments = []
        tokens = []
        block_has_content = False

    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise DataError(f"cannot open CoNLL-U source: {path}") from error

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                finish_sentence()
                continue
            block_has_content = True
            if line.startswith("#"):
                comments.append(line)
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ConlluFormatError(
                    f"{stable_source}:{line_number}: expected 10 tab-separated columns, "
                    f"got {len(fields)}"
                )
            raw_id = fields[0]
            if "-" in raw_id or "." in raw_id:
                continue
            try:
                token_id = int(raw_id)
            except ValueError as error:
                raise ConlluFormatError(
                    f"{stable_source}:{line_number}: invalid token ID {raw_id!r}"
                ) from error
            if token_id <= 0:
                raise ConlluFormatError(
                    f"{stable_source}:{line_number}: token ID must be positive"
                )
            token_provenance = TokenProvenance(
                conllu_id=token_id,
                line_number=line_number,
            )
            form = fields[1]
            if labeled:
                upos = fields[3]
                if upos not in UPOS_TAGS:
                    raise ConlluFormatError(
                        f"{stable_source}:{line_number}: invalid gold UPOS {upos!r}"
                    )
                tokens.append(
                    LabeledToken(form=form, provenance=token_provenance, upos=upos)
                )
            else:
                # Do not read or retain fields[3]. Test sources use this path.
                tokens.append(Token(form=form, provenance=token_provenance))
        finish_sentence()
    return tuple(sentences)


def parse_labeled_conllu(
    path: str | Path, *, split: str, source_file: str | None = None
) -> tuple[Sentence, ...]:
    """Parse gold train/dev data while refusing the protected test split."""
    if split.casefold() == "test":
        raise DataLeakageError(
            "ordinary data preparation may not parse test UPOS; use the final-evaluation path"
        )
    return _parse_conllu(
        path,
        split=split,
        source_file=source_file,
        labeled=True,
    )


def parse_unlabeled_conllu(
    path: str | Path, *, split: str, source_file: str | None = None
) -> tuple[Sentence, ...]:
    """Parse only IDs and exact FORM values, never exposing UPOS."""
    return _parse_conllu(
        path,
        split=split,
        source_file=source_file,
        labeled=False,
    )


def parse_conllu(
    path: str | Path,
    *,
    split: str,
    source_file: str | None = None,
    labeled: bool = False,
) -> tuple[Sentence, ...]:
    """Parse CoNLL-U safely; labels are opt-in and forbidden for test."""
    if labeled:
        return parse_labeled_conllu(path, split=split, source_file=source_file)
    return parse_unlabeled_conllu(path, split=split, source_file=source_file)


def ordered_form_hash(forms: Iterable[str]) -> str:
    """Hash the exact ordered FORM sequence with unambiguous UTF-8 framing."""
    exact_forms = list(forms)
    if not all(isinstance(form, str) for form in exact_forms):
        raise TypeError("all FORM values must be strings")
    return hashlib.sha256(canonical_json_bytes(exact_forms)).hexdigest()


def _selection_identity(sentence: Sentence, occurrence: int) -> str:
    provenance = sentence.provenance
    identity = {
        "form_hash": sentence.form_hash,
        "source_file": provenance.source_file,
        "sentence_index": provenance.sentence_index,
        "sent_id": provenance.sent_id,
        "occurrence": occurrence,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _ranked_indices(sentences: Sequence[Sentence], seed: int) -> list[int]:
    """Create a deterministic order using FORM/provenance only, never labels."""
    ranked = []
    occurrences: dict[str, int] = {}
    for index, sentence in enumerate(sentences):
        base = sentence.form_hash
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        identity = _selection_identity(sentence, occurrence)
        rank = hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).digest()
        ranked.append((rank, identity, index))
    ranked.sort()
    return [index for _, _, index in ranked]


def deterministic_sample(
    sentences: Sequence[Sentence], size: int, *, seed: int
) -> tuple[Sentence, ...]:
    """Select an exact-size deterministic sample and retain corpus order."""
    if size < 0 or size > len(sentences):
        raise ValueError(f"sample size {size} is outside 0..{len(sentences)}")
    selected = set(_ranked_indices(sentences, seed)[:size])
    return tuple(sentence for index, sentence in enumerate(sentences) if index in selected)


def _partition_counts(size: int, fractions: Mapping[str, float]) -> dict[str, int]:
    if not fractions:
        raise ValueError("partition fractions must not be empty")
    numeric = {name: float(value) for name, value in fractions.items()}
    if any(value < 0 for value in numeric.values()):
        raise ValueError("partition fractions must be non-negative")
    if not math.isclose(sum(numeric.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("partition fractions must sum to one")
    exact = {name: size * value for name, value in numeric.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = size - sum(counts.values())
    order = sorted(
        numeric,
        key=lambda name: (-(exact[name] - counts[name]), list(numeric).index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def deterministic_partitions(
    sentences: Sequence[Sentence],
    fractions: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, tuple[Sentence, ...]]:
    """Partition sentences exactly once using no label-derived information."""
    counts = _partition_counts(len(sentences), fractions)
    ranked = _ranked_indices(sentences, seed)
    result: dict[str, tuple[Sentence, ...]] = {}
    offset = 0
    for name, count in counts.items():
        selected = set(ranked[offset : offset + count])
        result[name] = tuple(
            sentence for index, sentence in enumerate(sentences) if index in selected
        )
        offset += count
    return result


def nested_gold_samples(
    sentences: Sequence[Sentence], sizes: Sequence[int], *, seed: int
) -> dict[int, tuple[Sentence, ...]]:
    """Create nested, label-independent samples from one deterministic ranking."""
    normalized_sizes = list(sizes)
    if (
        not normalized_sizes
        or normalized_sizes != sorted(set(normalized_sizes))
        or normalized_sizes[0] <= 0
        or normalized_sizes[-1] > len(sentences)
    ):
        raise ValueError("sizes must be unique, positive, increasing, and available")
    ranking = _ranked_indices(sentences, seed)
    samples: dict[int, tuple[Sentence, ...]] = {}
    for size in normalized_sizes:
        selected = set(ranking[:size])
        samples[size] = tuple(
            sentence for index, sentence in enumerate(sentences) if index in selected
        )
    return samples


def assert_disjoint_form_hashes(
    splits: Mapping[str, Sequence[Sentence]],
) -> None:
    """Reject exact ordered-FORM sentence overlap between named splits."""
    owners: dict[str, str] = {}
    for split, sentences in splits.items():
        for sentence in sentences:
            digest = sentence.form_hash
            previous = owners.get(digest)
            if previous is not None and previous != split:
                raise DataLeakageError(
                    f"exact ordered-FORM overlap between {previous!r} and {split!r}: "
                    f"{digest}"
                )
            owners[digest] = split


def _hash_counts(sentences: Sequence[Sentence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        counts[sentence.form_hash] = counts.get(sentence.form_hash, 0) + 1
    return counts


def _deduplicate(
    sentences: Sequence[Sentence], *, excluded_hashes: set[str] | None = None
) -> tuple[Sentence, ...]:
    """Keep the first occurrence in corpus order and remove excluded text."""
    excluded = excluded_hashes or set()
    seen: set[str] = set()
    kept = []
    for sentence in sentences:
        digest = sentence.form_hash
        if digest in excluded or digest in seen:
            continue
        seen.add(digest)
        kept.append(sentence)
    return tuple(kept)


def _overlap_summary(
    left: Sequence[Sentence], right: Sequence[Sentence]
) -> dict[str, int]:
    left_counts = _hash_counts(left)
    right_counts = _hash_counts(right)
    shared = left_counts.keys() & right_counts.keys()
    return {
        "unique_hashes": len(shared),
        "left_sentences": sum(left_counts[digest] for digest in shared),
        "right_sentences": sum(right_counts[digest] for digest in shared),
    }


def _duplicate_summary(sentences: Sequence[Sentence]) -> dict[str, int]:
    counts = _hash_counts(sentences)
    return {
        "sentences": len(sentences),
        "unique_hashes": len(counts),
        "duplicate_sentences": len(sentences) - len(counts),
        "duplicated_hashes": sum(count > 1 for count in counts.values()),
    }


def _is_labeled(sentence: Sentence) -> bool:
    return all(isinstance(token, LabeledToken) for token in sentence.tokens)


def sentence_record(sentence: Sentence, *, include_labels: bool) -> dict[str, Any]:
    """Convert one sentence to a provenance-preserving JSON record."""
    provenance = sentence.provenance
    tokens = []
    for token in sentence.tokens:
        token_record: dict[str, Any] = {
            "id": token.provenance.conllu_id,
            "form": token.form,
            "line_number": token.provenance.line_number,
        }
        if include_labels:
            if not isinstance(token, LabeledToken):
                raise DataLeakageError("labeled output requested for an unlabeled sentence")
            token_record["upos"] = token.upos
        tokens.append(token_record)
    return {
        "provenance": {
            "source_file": provenance.source_file,
            "split": provenance.split,
            "sentence_index": provenance.sentence_index,
            "sent_id": provenance.sent_id,
        },
        "comments": list(sentence.comments),
        "form_hash": sentence.form_hash,
        "tokens": tokens,
    }


def write_prepared_jsonl(
    path: str | Path,
    sentences: Sequence[Sentence],
    *,
    split: str,
    include_labels: bool,
) -> None:
    """Atomically write prepared JSONL while enforcing the test-label barrier."""
    if split.casefold() == "test" and include_labels:
        raise DataLeakageError("ordinary prepared test artifacts may not contain UPOS")
    if split.casefold() == "test" and any(_is_labeled(sentence) for sentence in sentences):
        raise DataLeakageError("labeled sentence objects may not enter prepared test artifacts")
    if include_labels and any(not _is_labeled(sentence) for sentence in sentences):
        raise DataLeakageError("all sentences in a labeled artifact must carry UPOS")

    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as handle:
            for sentence in sentences:
                record = sentence_record(sentence, include_labels=include_labels)
                handle.write(canonical_json_bytes(record))
            handle.flush()


def _source_name(language: str, split: str, path: Path) -> str:
    return f"{language}/{split}/{path.name}"


def _flatten(
    files: Mapping[str, Mapping[str, Sequence[str | Path]]]
) -> dict[str, Path]:
    flattened: dict[str, Path] = {}
    for language, splits in files.items():
        for split, paths in splits.items():
            for value in paths:
                path = Path(value)
                flattened[_source_name(language, split, path)] = path
    return flattened


def _expected_outputs(
    prepared_root: Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    outputs = {
        "czech/train": prepared_root / "czech" / "train.jsonl",
        "czech/lapt_train_forms": prepared_root / "czech" / "lapt_train_forms.jsonl",
        "czech/dev_sample": prepared_root / "czech" / "dev_sample.jsonl",
        "czech/test": prepared_root / "czech" / "test.jsonl",
        "slovak/train": prepared_root / "slovak" / "train.jsonl",
        "slovak/dev": prepared_root / "slovak" / "dev.jsonl",
        "data_audit": prepared_root / "data_audit.json",
    }
    for name in config["data"]["dev_partitions"]:
        outputs[f"czech/dev/{name}"] = prepared_root / "czech" / "dev" / f"{name}.jsonl"
    for seed in config["random_seeds"]:
        for size in config["seed_sizes"]:
            outputs[f"czech/gold/seed_{seed}/n_{size}"] = (
                prepared_root / "czech" / "gold" / f"seed_{seed}" / f"n_{size}.jsonl"
            )
    outputs["dataset_manifest"] = prepared_root / "dataset_manifest.json"
    return outputs


def _parse_many(
    paths: Sequence[str | Path], *, language: str, split: str, labeled: bool
) -> tuple[Sentence, ...]:
    result: list[Sentence] = []
    for value in paths:
        path = Path(value)
        source = _source_name(language, split, path)
        parser = parse_labeled_conllu if labeled else parse_unlabeled_conllu
        result.extend(parser(path, split=split, source_file=source))
    return tuple(result)


def _split_metadata(sentences: Sequence[Sentence]) -> dict[str, Any]:
    return {
        "sentences": len(sentences),
        "tokens": sum(len(sentence.tokens) for sentence in sentences),
        "ordered_form_hashes": [sentence.form_hash for sentence in sentences],
    }


def prepare_data(
    config: Mapping[str, Any],
    raw_files: Mapping[str, Mapping[str, Sequence[str | Path]]],
    prepared_root: str | Path,
    *,
    run_fingerprint: str,
) -> dict[str, Any]:
    """Prepare pinned corpora and atomically commit a leakage-safe dataset.

    Czech test is parsed as unlabeled and written without UPOS. Its raw gold
    source must be held in a protected cache outside the portable run tree by
    the caller. A second Czech-train artifact is serialized FORM-only for LAPT.
    """
    prepared_root = Path(prepared_root)
    artifact_manifest = prepared_root / "artifact-manifest.json"
    outputs = _expected_outputs(prepared_root, config)
    inputs = _flatten(raw_files)
    parameters = {
        "release": config["data"]["release"],
        "dev_cap": config["data"]["dev_cap"],
        "dev_sample_seed": config["data"]["dev_sample_seed"],
        "dev_partition_seed": config["data"]["dev_partition_seed"],
        "dev_partitions": dict(config["data"]["dev_partitions"]),
        "seed_sizes": list(config["seed_sizes"]),
        "random_seeds": list(config["random_seeds"]),
        "form_hash_algorithm": FORM_HASH_ALGORITHM,
        "lapt_input_policy": "dedicated-clean-czech-train-forms-only-v1",
    }
    if strict_resume(
        artifact_manifest,
        stage="data-preparation",
        run_fingerprint=run_fingerprint,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        relative_to=prepared_root,
    ):
        value = json.loads(outputs["dataset_manifest"].read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ArtifactIntegrityError("dataset manifest must be a JSON object")
        return value

    required = {
        ("czech", "train"),
        ("czech", "dev"),
        ("czech", "test"),
        ("slovak", "train"),
        ("slovak", "dev"),
    }
    missing = [
        f"{language}/{split}"
        for language, split in sorted(required)
        if not raw_files.get(language, {}).get(split)
    ]
    if missing:
        raise DataError(f"missing configured raw splits: {', '.join(missing)}")

    czech_train_raw = _parse_many(
        raw_files["czech"]["train"], language="czech", split="train", labeled=True
    )
    czech_dev_raw = _parse_many(
        raw_files["czech"]["dev"], language="czech", split="dev", labeled=True
    )
    czech_test = _parse_many(
        raw_files["czech"]["test"], language="czech", split="test", labeled=False
    )
    slovak_train_raw = _parse_many(
        raw_files["slovak"]["train"], language="slovak", split="train", labeled=True
    )
    slovak_dev_raw = _parse_many(
        raw_files["slovak"]["dev"], language="slovak", split="dev", labeled=True
    )

    # Audit the raw official splits before applying the documented precedence:
    # protected test > dev > train. Test stays in official order; duplicate train
    # and dev text is removed so no model-selection text reaches training/LAPT.
    czech_test_hashes = {sentence.form_hash for sentence in czech_test}
    czech_dev = _deduplicate(czech_dev_raw, excluded_hashes=czech_test_hashes)
    czech_dev_hashes = {sentence.form_hash for sentence in czech_dev}
    czech_train = _deduplicate(
        czech_train_raw, excluded_hashes=czech_test_hashes | czech_dev_hashes
    )
    slovak_dev = _deduplicate(slovak_dev_raw)
    slovak_train = _deduplicate(
        slovak_train_raw,
        excluded_hashes={sentence.form_hash for sentence in slovak_dev},
    )
    assert_disjoint_form_hashes(
        {"czech_train": czech_train, "czech_dev": czech_dev, "czech_test": czech_test}
    )

    data_audit = {
        "schema_version": 1,
        "hash_algorithm": FORM_HASH_ALGORITHM,
        "policy": "deduplicate within splits; exclude test hashes from dev/train; "
        "exclude retained dev hashes from train; preserve official test order",
        "raw": {
            "czech/train": _duplicate_summary(czech_train_raw),
            "czech/dev": _duplicate_summary(czech_dev_raw),
            "czech/test": _duplicate_summary(czech_test),
            "slovak/train": _duplicate_summary(slovak_train_raw),
            "slovak/dev": _duplicate_summary(slovak_dev_raw),
        },
        "raw_cross_split": {
            "czech/train-dev": _overlap_summary(czech_train_raw, czech_dev_raw),
            "czech/train-test": _overlap_summary(czech_train_raw, czech_test),
            "czech/dev-test": _overlap_summary(czech_dev_raw, czech_test),
            "slovak/train-dev": _overlap_summary(slovak_train_raw, slovak_dev_raw),
        },
        "retained": {
            "czech/train": _duplicate_summary(czech_train),
            "czech/dev": _duplicate_summary(czech_dev),
            "czech/test_official": _duplicate_summary(czech_test),
            "slovak/train": _duplicate_summary(slovak_train),
            "slovak/dev": _duplicate_summary(slovak_dev),
        },
        "removed": {
            "czech/train": len(czech_train_raw) - len(czech_train),
            "czech/dev": len(czech_dev_raw) - len(czech_dev),
            "slovak/train": len(slovak_train_raw) - len(slovak_train),
            "slovak/dev": len(slovak_dev_raw) - len(slovak_dev),
        },
        "test_labels_exposed": False,
    }
    atomic_write_json(outputs["data_audit"], data_audit)

    dev_cap = min(int(config["data"]["dev_cap"]), len(czech_dev))
    czech_dev_sample = deterministic_sample(
        czech_dev,
        dev_cap,
        seed=int(config["data"]["dev_sample_seed"]),
    )
    dev_partitions = deterministic_partitions(
        czech_dev_sample,
        config["data"]["dev_partitions"],
        seed=int(config["data"]["dev_partition_seed"]),
    )
    gold_samples = {
        int(seed): nested_gold_samples(
            czech_train,
            config["seed_sizes"],
            seed=int(seed),
        )
        for seed in config["random_seeds"]
    }

    write_prepared_jsonl(
        outputs["czech/train"], czech_train, split="train", include_labels=True
    )
    write_prepared_jsonl(
        outputs["czech/lapt_train_forms"],
        czech_train,
        split="train",
        include_labels=False,
    )
    write_prepared_jsonl(
        outputs["czech/dev_sample"], czech_dev_sample, split="dev", include_labels=True
    )
    write_prepared_jsonl(
        outputs["czech/test"], czech_test, split="test", include_labels=False
    )
    write_prepared_jsonl(
        outputs["slovak/train"], slovak_train, split="train", include_labels=True
    )
    write_prepared_jsonl(
        outputs["slovak/dev"], slovak_dev, split="dev", include_labels=True
    )
    for name, sentences in dev_partitions.items():
        write_prepared_jsonl(
            outputs[f"czech/dev/{name}"], sentences, split="dev", include_labels=True
        )
    for seed, samples in gold_samples.items():
        for size, sentences in samples.items():
            write_prepared_jsonl(
                outputs[f"czech/gold/seed_{seed}/n_{size}"],
                sentences,
                split="train",
                include_labels=True,
            )

    dataset_manifest: dict[str, Any] = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "form_hash_algorithm": FORM_HASH_ALGORITHM,
        "sources": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(inputs.items())
        },
        "splits": {
            "czech/train": _split_metadata(czech_train),
            "czech/lapt_train_forms": {
                **_split_metadata(czech_train),
                "labels_included": False,
                "purpose": "czech-train-form-only-lapt",
            },
            "czech/dev_clean": _split_metadata(czech_dev),
            "czech/dev_sample": _split_metadata(czech_dev_sample),
            "czech/test_unlabeled": _split_metadata(czech_test),
            "slovak/train": _split_metadata(slovak_train),
            "slovak/dev": _split_metadata(slovak_dev),
            **{
                f"czech/dev/{name}": _split_metadata(sentences)
                for name, sentences in dev_partitions.items()
            },
            **{
                f"czech/gold/seed_{seed}/n_{size}": _split_metadata(sentences)
                for seed, samples in gold_samples.items()
                for size, sentences in samples.items()
            },
        },
        "test_labels_exposed": False,
    }
    atomic_write_json(outputs["dataset_manifest"], dataset_manifest)
    write_manifest(
        artifact_manifest,
        stage="data-preparation",
        run_fingerprint=run_fingerprint,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        relative_to=prepared_root,
    )
    return dataset_manifest
