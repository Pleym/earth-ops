import json

import numpy as np
import pytest

from earth_ops.synthetic import make_product, to_dn, write_product


@pytest.fixture
def product(tmp_path):
    """Nominal simulated product: 128x128 px, 25 % cloudy rows."""
    return make_product(tmp_path / "products", size=128, cloud_fraction=0.25)


@pytest.fixture
def make_custom(tmp_path):
    """Build a product from explicit reflectances / SCL classes."""

    def _make(red, nir, scl, product_id="CUSTOM", date="2025-06-01"):
        return write_product(
            tmp_path / "products", product_id, to_dn(red), to_dn(nir),
            np.asarray(scl, np.uint8), date=date,
        )

    return _make


def edit_metadata(product_dir, **changes):
    path = product_dir / "metadata.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({**meta, **changes}))
