from causalnews.preprocess import clean_clause, split_sentences, tokenize


def test_split_sentences_basic():
    assert split_sentences("It rained. The river rose! Did it flood?") == [
        "It rained.", "The river rose!", "Did it flood?"]


def test_split_sentences_keeps_abbreviations_and_decimals():
    text = "Dr. Smith met U.S. officials. Inflation reached 6.1 percent."
    assert len(split_sentences(text)) == 2


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_tokenize_lowercases_and_keeps_hyphens():
    assert tokenize("COVID-19 isn't Wi-Fi!") == ["covid-19", "isn't", "wi-fi"]


def test_clean_clause_removes_reporting_frame_and_attribution():
    assert clean_clause("Researchers reported that sea temperatures rise") == \
        "sea temperatures rise"
    assert clean_clause("flooding in towns, the agency reported.") == \
        "flooding in towns"
