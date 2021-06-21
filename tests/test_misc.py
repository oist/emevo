from emevo.environments import _levenshtein_distance


def test_levenhtein() -> None:
    a = "ready to go"
    b = "readily to do"
    assert _levenshtein_distance(a, b) == 3
