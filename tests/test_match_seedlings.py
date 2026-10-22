

def test_duplicate_series_finds_byte_identical_frames_and_lets_a_seedling_match_its_twin(tmp_path):
    from multi.match_seedlings import duplicate_series, match_seedlings
    for name, content in (("A", b"same"), ("B", b"same"), ("C", b"other")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "f1.tif").write_bytes(content)
    aliases = duplicate_series(tmp_path)
    assert aliases == {"A": "A", "B": "A"}                                   # C has no twin
    new = {("B", 0): (0, 0, 10, 10)}
    ref = [("", {("A", 4): (0, 0, 10, 10)})]
    assert match_seedlings(new, ref) == {}                                   # different series names: not seen
    assert list(match_seedlings(new, ref, aliases=aliases)) == [("B", 0)]
