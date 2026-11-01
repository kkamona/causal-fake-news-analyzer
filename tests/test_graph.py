from causalnews.extractor import extract_claims
from causalnews.graph import CausalGraph, normalize_concept


def graph_of(text):
    return CausalGraph.from_claims(extract_claims(text))


def test_normalize_concept():
    assert normalize_concept("The heavy Rains,") == "heavy rain"
    assert normalize_concept("") == ""


def test_chain_length():
    g = graph_of("Inflation causes unemployment, which leads to crime.")
    assert g.longest_chain() == 2
    assert not g.has_cycle()


def test_cycle_detection():
    g = graph_of("Poverty causes crime. Crime causes poverty.")
    assert g.has_cycle()


def test_hub_detection():
    g = graph_of("Sugar causes cancer. Sugar causes madness. Sugar causes poverty.")
    assert g.hubs() == ["Sugar"]
    assert g.max_out_degree() == 3


def test_same_concept_is_merged():
    g = graph_of("Heavy rain causes floods. The heavy rains cause landslides.")
    assert len(g.nodes) == 3


def test_to_dict_shape():
    d = graph_of("Rain causes floods.").to_dict()
    assert set(d) == {"nodes", "edges", "longest_chain", "has_cycle", "hubs"}
    assert d["edges"][0]["claims"] == 1


def test_empty_graph():
    g = CausalGraph()
    assert g.longest_chain() == 0 and not g.has_cycle() and g.nodes == []
