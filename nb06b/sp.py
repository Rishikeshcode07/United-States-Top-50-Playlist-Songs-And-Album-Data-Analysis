def patch(path, pairs):
    t = open(path, encoding="utf-8").read()
    for old, new in pairs:
        n = t.count(old)
        assert n == 1, f"found {n} times (need exactly 1): {old[:70]!r}"
        t = t.replace(old, new)
    open(path, "w", encoding="utf-8").write(t)
