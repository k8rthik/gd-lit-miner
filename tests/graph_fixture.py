"""Builds a small populated knowledge-graph database for query/trend/export tests."""

from bioscrolls.graph.build import build_graph
from bioscrolls.graph.store import open_store
from bioscrolls.models import Document

ENTITIES = [
    ("NCBIGene:348", "Gene", "APOE", True),
    ("MESH:D000544", "Disease", "Alzheimer Disease", True),
    ("MESH:D000077265", "Chemical", "Donepezil", True),
    ("gene-text:foo", "Gene", "foo", False),
]


def populate(db_path):
    docs = [Document(str(i), "t", "a", 2010 + i) for i in range(10)]
    with open_store(db_path) as store:
        store.upsert_documents(docs)
        for doc in docs:
            i = int(doc.pmid)
            mentions = [
                (doc.pmid, 0, 0, 4, "APOE", "Gene", "NCBIGene:348"),
                (doc.pmid, 0, 10, 12, "AD", "Disease", "MESH:D000544"),
            ]
            relations = [
                (doc.pmid, 0, "NCBIGene:348", "MESH:D000544", "Association", 0.9, "[E1] APOE [/E1] in [E2] AD [/E2]")
            ]
            if i >= 7:  # donepezil only appears recently -> emerging
                mentions.append((doc.pmid, 1, 20, 29, "donepezil", "Chemical", "MESH:D000077265"))
                relations.append((doc.pmid, 1, "MESH:D000077265", "MESH:D000544", "Negative_Correlation", 0.8, "s"))
            if i == 0:
                relations.append((doc.pmid, 0, "gene-text:foo", "MESH:D000544", "None", 0.1, "s"))
            store.save_extraction(doc.pmid, "title+abstract", mentions, relations, ENTITIES)
        build_graph(store, min_confidence=0.5)
    return db_path
