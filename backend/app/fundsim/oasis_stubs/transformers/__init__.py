"""Substitut minimal de transformers (voir stubs/torch)."""


class AutoModel:
    @classmethod
    def from_pretrained(cls, *a, **k):
        raise RuntimeError("transformers indisponible : substitut léger")


class AutoTokenizer(AutoModel):
    pass
