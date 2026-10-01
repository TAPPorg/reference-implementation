def lookup(mapping, key, description):
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(f"unknown {description} {key!r}")
