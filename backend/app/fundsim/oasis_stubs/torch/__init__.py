"""Substitut minimal de torch : seulement ce qu'OASIS touche à l'import. Le recommandeur 'reddit'
(score de popularité) n'en a pas besoin ; les recommandeurs neuronaux sont désactivés."""


class Tensor:
    """Classe inerte : les bibliothèques qui testent isinstance(x, torch.Tensor) obtiennent False."""


class _Cuda:
    @staticmethod
    def is_available():
        return False


cuda = _Cuda()


class device:
    def __init__(self, name="cpu"):
        self.type = name


def _unavailable(name):
    def fn(*args, **kwargs):
        raise RuntimeError(f"torch.{name} indisponible : substitut léger, recommandeurs neuronaux désactivés")
    return fn


class no_grad:
    """Utilisable comme décorateur ou gestionnaire de contexte ; ne fait rien."""
    def __call__(self, fn):
        return fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


for _n in ("tensor", "topk", "norm", "cat", "stack", "rand", "matmul"):
    globals()[_n] = _unavailable(_n)
