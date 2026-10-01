"""Load the trained models and vocabularies into a ready-to-run pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from bioscrolls import config
from bioscrolls.extraction import Classifier, NormalizerFactory, Tagger
from bioscrolls.normalize.lexicon import lexicon_sources
from bioscrolls.normalize.normalizer import Normalizer


@dataclass(frozen=True)
class Pipeline:
    tagger: Tagger
    classifier: Classifier
    normalizer_factory: NormalizerFactory
    device: str


def _normalizer_factory(requests) -> Normalizer:
    return Normalizer.build(lexicon_sources(), requests)


def load_pipeline() -> Pipeline:  # pragma: no cover - needs trained weights
    from bioscrolls.ner.predict import NerTagger
    from bioscrolls.relation.predict import RelationClassifier
    from bioscrolls.training_utils import pick_device

    lexicon_sources()  # fail fast if vocabularies are missing
    device = pick_device()
    return Pipeline(
        tagger=NerTagger.from_dir(config.NER_MODEL_DIR, device),
        classifier=RelationClassifier.from_dir(config.RE_MODEL_DIR, device),
        normalizer_factory=_normalizer_factory,
        device=str(device),
    )
