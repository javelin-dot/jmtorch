"""骨架冒烟：可导入，版本与 pyproject 一致。"""

import jmtorch


def test_import_jmtorch():
    assert jmtorch.__version__ == "0.1.0"


def test_subpackages_importable():
    import jmtorch.autograd
    import jmtorch.data
    import jmtorch.export
    import jmtorch.nn
    import jmtorch.optim
    import jmtorch.tensor
    import jmtorch.training
