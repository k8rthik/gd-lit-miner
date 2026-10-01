"""Exercise inference wrappers with a tiny randomly initialised BERT (no downloads)."""

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertModel, BertTokenizerFast

from bioscrolls import config
from bioscrolls.ner.predict import NerTagger
from bioscrolls.relation.model import MarkerRelationClassifier
from bioscrolls.relation.predict import RelationClassifier

WORDS = ["APOE", "AD", "raises", "risk", "of", "in", "the", "levodopa", "Parkinson", "disease", "."]


@pytest.fixture
def tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
    return tok


def tiny_config(vocab_size, **kwargs):
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        **kwargs,
    )


def test_ner_tagger_roundtrip(tokenizer, tmp_path):
    torch.manual_seed(0)
    labels = dict(enumerate(config.NER_LABELS))
    model = BertForTokenClassification(
        tiny_config(
            tokenizer.vocab_size, num_labels=len(labels), id2label=labels, label2id={v: k for k, v in labels.items()}
        )
    )
    model.save_pretrained(tmp_path / "ner")
    tokenizer.save_pretrained(tmp_path / "ner")
    tagger = NerTagger.from_dir(tmp_path / "ner", torch.device("cpu"))
    out = tagger.tag(["APOE raises AD risk .", "levodopa in Parkinson disease"], batch_size=1)
    assert len(out) == 2
    for text, spans in zip(["APOE raises AD risk .", "levodopa in Parkinson disease"], out, strict=True):
        for span in spans:
            assert text[span.start : span.end] == span.text


def test_ner_tagger_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        NerTagger.from_dir(tmp_path / "nope", torch.device("cpu"))


def test_relation_classifier_save_load_classify(tokenizer, tmp_path):
    torch.manual_seed(0)
    tokenizer.add_special_tokens({"additional_special_tokens": list(config.ENTITY_MARKERS)})
    encoder = BertModel(tiny_config(len(tokenizer)))
    head_id, tail_id = tokenizer.convert_tokens_to_ids([config.HEAD_START, config.TAIL_START])
    model = MarkerRelationClassifier(encoder, len(config.RELATION_LABELS), head_id, tail_id)
    model.save(tmp_path / "re", tokenizer, config.RELATION_LABELS)
    clf = RelationClassifier.from_dir(tmp_path / "re", torch.device("cpu"))
    preds = clf.classify(["[E1] APOE [/E1] raises [E2] AD [/E2] risk .", "no markers here"], batch_size=1)
    assert len(preds) == 2
    for label, confidence, probs in preds:
        assert label in config.RELATION_LABELS
        assert sum(p for _, p in probs) == pytest.approx(1.0, abs=1e-5)
        assert confidence == max(p for _, p in probs)


def test_relation_model_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        MarkerRelationClassifier.load(tmp_path / "nope", torch.device("cpu"))
