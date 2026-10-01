from bioscrolls.models import Span
from bioscrolls.ner.evaluate import span_metrics


def test_span_metrics_micro_and_per_type():
    gold = [[Span(0, 4, "Gene"), Span(10, 12, "Disease")], [Span(0, 3, "Chemical")]]
    pred = [[Span(0, 4, "Gene"), Span(10, 13, "Disease")], [Span(0, 3, "Chemical"), Span(5, 6, "Gene")]]
    m = span_metrics(gold, pred)
    assert (m["micro"].tp, m["micro"].fp, m["micro"].fn) == (2, 2, 1)
    assert m["per_type"]["Gene"].tp == 1 and m["per_type"]["Gene"].fp == 1
    assert m["per_type"]["Disease"].f1 == 0.0
    assert m["per_type"]["Chemical"].f1 == 1.0
