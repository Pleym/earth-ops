"""Validation, NDVI and quality control (REQ-VAL, REQ-PRO, REQ-QUA)."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from earth_ops import mosaic, quality
from earth_ops.ingest import ingest_item, write_band
from earth_ops.ndvi import (NDVI_NODATA, NDVI_SCALE, compute_ndvi, open_ndvi_series,
                              open_product)
from earth_ops.validate import ValidationError, load_metadata, validate_product
from tests.conftest import edit_metadata

# --- REQ-VAL: input validation ------------------------------------------------


def test_req_val_01_nominal_product_is_accepted(product):
    meta = validate_product(product)
    assert meta["product_id"] == product.name


def test_req_val_02_missing_band_is_rejected(product):
    (product / "B08.tif").unlink()
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_MISSING_FILE"


@pytest.mark.parametrize("bad", ['{"product_id": "x"}', "not json", "5", "null"])
def test_req_val_03_incomplete_metadata_is_rejected(product, bad):
    (product / "metadata.json").write_text(bad)
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_METADATA"


@pytest.mark.parametrize("change", [{"scale": "abc"}, {"scale": float("nan")},
                                    {"datetime": "garbage"}])
def test_req_val_03_invalid_metadata_values_are_rejected(product, change):
    edit_metadata(product, **change)
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_METADATA"


@pytest.mark.parametrize("corrupt", [lambda b: b"\x00garbage", lambda b: b[: len(b) // 2]])
def test_req_val_04_corrupted_file_is_rejected(product, corrupt):
    band = product / "B04.tif"
    band.write_bytes(corrupt(band.read_bytes()))  # garbage header, or valid header + cut blocks
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_CORRUPT"


def _rewrite(product, band, data, crs="EPSG:32631"):
    write_band(product / f"{band}.tif", data, crs, from_origin(360000, 4840000, 10, 10), 0)


def test_req_val_05_band_shape_mismatch_is_rejected(product):
    _rewrite(product, "B08", np.ones((64, 64), np.uint16))
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_GEOMETRY"


def test_req_val_05_band_crs_mismatch_is_rejected(product):
    _rewrite(product, "B08", np.ones((128, 128), np.uint16), crs="EPSG:32630")
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_GEOMETRY"


def test_req_val_05_geographic_crs_is_rejected(product):
    for band, dtype in (("B04", np.uint16), ("B08", np.uint16), ("SCL", np.uint8)):
        _rewrite(product, band, np.ones((128, 128), dtype), crs="EPSG:4326")
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_CRS"


def test_req_val_06_wrong_dtype_is_rejected(product):
    _rewrite(product, "B04", np.ones((128, 128), np.float32))
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_DTYPE"


def test_req_val_07_empty_product_is_rejected(product):
    _rewrite(product, "B04", np.zeros((128, 128), np.uint16))
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_EMPTY"


def test_req_val_08_offset_inconsistent_with_data_is_rejected(product):
    # Real case: the catalogue advertised offset -0.1 on DNs from which it was already removed
    edit_metadata(product, offset=-0.2)
    with pytest.raises(ValidationError) as err:
        validate_product(product)
    assert err.value.code == "E_RADIOMETRY"


# --- REQ-PRO: NDVI processing -------------------------------------------------


def test_req_pro_01_ndvi_matches_reference_formula(make_custom):
    p = make_custom(red=np.full((4, 4), 0.1), nir=np.full((4, 4), 0.5), scl=np.full((4, 4), 4))
    meta = load_metadata(p)
    ds = open_product(p)
    ndvi = compute_ndvi(ds.red, ds.nir, meta["scale"], meta["offset"])
    np.testing.assert_allclose(ndvi.values, (0.5 - 0.1) / (0.5 + 0.1), rtol=1e-5)
    assert ndvi.dtype == np.float32


def test_req_pro_02_nodata_and_zero_denominator_give_nan(make_custom):
    red = np.array([[-0.1, 0.0, -0.002]])  # 1st pixel: DN 0 = nodata
    nir = np.array([[0.5, 0.0, -0.001]])  # 2nd: red + nir ~ 0 (float32: ~1e-9); 3rd: sum < 0
    p = make_custom(red=red, nir=nir, scl=[[4, 4, 6]])
    ds = open_product(p)
    assert int(ds.red[0, 0]) == 0
    ndvi = compute_ndvi(ds.red, ds.nir, 1e-4, -0.1)
    assert np.isnan(ndvi.values).all()


def test_req_pro_03_chunked_result_equals_in_memory_result(product):
    meta = load_metadata(product)
    eager = open_product(product)
    lazy = open_product(product, chunks=32)
    assert lazy.red.chunks is not None
    a = compute_ndvi(eager.red, eager.nir, meta["scale"], meta["offset"])
    b = compute_ndvi(lazy.red, lazy.nir, meta["scale"], meta["offset"]).compute()
    xr.testing.assert_identical(a, b)


def test_req_pro_04_series_is_stacked_by_date(make_custom):
    ones = np.ones((8, 8))
    later = make_custom(0.1 * ones, 0.5 * ones, 4 * ones, "B", date="2025-07-01")
    earlier = make_custom(0.1 * ones, 0.3 * ones, 4 * ones, "A", date="2025-06-01")
    cube = open_ndvi_series([later, earlier], chunks=4)
    assert cube.dims == ("time", "y", "x")
    assert list(cube.time.dt.month.values) == [6, 7]
    np.testing.assert_allclose(cube.mean(["y", "x"]).values, [0.5, 2 / 3], rtol=1e-5)


def test_req_pro_04_series_rejects_misaligned_grids(make_custom):
    a = make_custom(np.full((8, 8), 0.1), np.full((8, 8), 0.5), np.full((8, 8), 4), "A")
    b = make_custom(np.full((4, 4), 0.1), np.full((4, 4), 0.5), np.full((4, 4), 4), "B")
    with pytest.raises(ValueError):
        open_ndvi_series([a, b])


# --- REQ-QUA: quality control -------------------------------------------------


def test_req_qua_01_clouds_and_out_of_range_values_are_masked():
    ndvi = xr.DataArray(np.array([[0.5, 0.5, 1.5]], np.float32), dims=("y", "x"))
    scl = xr.DataArray(np.array([[4, 9, 4]], np.uint8), dims=("y", "x"))
    masked = quality.apply_mask(ndvi, scl)
    assert masked.values[0, 0] == pytest.approx(0.5)
    assert np.isnan(masked.values[0, 1:]).all()


def test_req_qua_02_statistics_on_known_product(product):
    meta = load_metadata(product)
    ds = open_product(product, chunks=64)
    masked = quality.apply_mask(compute_ndvi(ds.red, ds.nir, meta["scale"], meta["offset"]), ds.scl)
    stats = quality.as_dict(quality.quality_stats(masked, ds.scl).compute())
    assert stats["valid_ratio"] == pytest.approx(0.75)
    assert stats["cloud_ratio"] == pytest.approx(0.25)
    assert -1 <= stats["ndvi_min"] <= stats["ndvi_mean"] <= stats["ndvi_max"] <= 1


def test_req_qua_03_status_depends_on_valid_ratio():
    assert quality.assess({"valid_ratio": 0.8}, min_valid_ratio=0.5) == "NOMINAL"
    assert quality.assess({"valid_ratio": 0.2}, min_valid_ratio=0.5) == "DEGRADED"
    assert quality.assess({"valid_ratio": None}, min_valid_ratio=0.5) == "DEGRADED"


def test_req_qua_04_all_masked_product_gives_null_stats():
    ndvi = xr.DataArray(np.full((2, 2), np.nan, np.float32), dims=("y", "x"))
    scl = xr.DataArray(np.full((2, 2), 9, np.uint8), dims=("y", "x"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # numpy: "All-NaN slice", expected here
        stats = quality.as_dict(quality.quality_stats(ndvi, scl))
    assert stats["valid_ratio"] == 0
    assert stats["ndvi_mean"] is None


# --- REQ-ING / mosaic -----------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["../evil", "/abs/path", ".."])
def test_req_ing_05_unsafe_product_id_is_refused(tmp_path, bad_id):
    with pytest.raises(OSError):
        ingest_item(SimpleNamespace(id=bad_id), (0, 0, 1, 1), tmp_path / "archive")
    assert not (tmp_path / "archive").exists()


def test_req_pro_06_mosaic_keeps_max_ndvi_and_survives_bad_scenes(monkeypatch):
    whole = Window(0, 0, 2, 1)
    scenes = {"a": (np.array([[0.2, np.nan]], np.float32), whole),
              "b": (np.array([[0.5, 0.1]], np.float32), whole), "rejected": None}

    def fake_scene(item, _transform, _shape):
        if item.id == "down":
            raise OSError("network down")
        return scenes[item.id]

    monkeypatch.setattr(mosaic, "scene_ndvi", fake_scene)
    items = [SimpleNamespace(id=i) for i in ("a", "b", "rejected", "down")]
    result = mosaic.build_mosaic(items, None, (2, 2))
    np.testing.assert_allclose(result[0], [0.5, 0.1])
    assert np.isnan(result[1]).all()


def test_req_pro_06_mosaic_end_to_end_on_local_scenes(product, tmp_path, monkeypatch):
    def local_item(name):  # STAC-like item pointing at a simulated product
        band = {"raster:bands": [{"scale": 1e-4, "offset": -0.1}]}
        assets = {key: SimpleNamespace(href=str(product / f), extra_fields=band)
                  for key, f in (("red", "B04.tif"), ("nir", "B08.tif"), ("scl", "SCL.tif"))}
        return SimpleNamespace(id=name, assets=assets, properties={"grid:code": "MGRS-31TCJ"})

    monkeypatch.setattr(mosaic, "best_scenes", lambda *_args: [local_item("A"), local_item("B")])
    monkeypatch.setattr(mosaic, "_href", lambda item, key: item.assets[key].href)
    with rasterio.open(product / "B04.tif") as src:
        bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    tif = mosaic.run_mosaic(bbox, "2025-06-01", "2025-06-30", 30, 100, tmp_path / "mosaic")
    with rasterio.open(tif) as src:
        data, epsg = src.read(1), src.crs.to_epsg()
    ndvi = data[data != NDVI_NODATA] / NDVI_SCALE
    assert epsg == 2154 and ndvi.size > 0
    assert ((ndvi > 0) & (ndvi <= 1)).all()  # simulated fields are vegetated
    assert (tmp_path / "mosaic" / "ndvi_mosaic.png").exists()
