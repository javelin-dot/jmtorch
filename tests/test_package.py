"""骨架冒烟：可导入，版本与 pyproject 一致。"""

import tinytorch


def test_import_tinytorch():
    assert tinytorch.__version__ == "0.1.0"


def test_subpackages_importable():
    import tinytorch.autograd
    import tinytorch.data
    import tinytorch.export
    import tinytorch.nn
    import tinytorch.optim
    import tinytorch.tensor
    import tinytorch.training
