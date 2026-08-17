"""P1: decoded matrices are cached on disk, so re-opening one skips the
several-second decode."""

import os

import numpy as np
import pytest

import matrix_cache


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    """Redirect the cache somewhere disposable. Both variables are set
    because cache_dir() prefers LOCALAPPDATA on Windows and XDG_CACHE_HOME
    elsewhere, and the tests must not depend on which platform they run
    on -- or write into the developer's real cache."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "matrix.mtx"
    path.write_bytes(b"pretend this is an encoded matrix")
    return path


def test_a_stored_matrix_comes_back_identical(cache_home, source):
    matrix = np.arange(60, dtype=np.int64).reshape(6, 10)
    assert matrix_cache.store(str(source), matrix) is True

    loaded = matrix_cache.load(str(source))
    assert loaded is not None
    assert loaded.dtype == matrix.dtype
    assert np.array_equal(loaded, matrix)


def test_a_miss_returns_none_rather_than_raising(cache_home, source):
    assert matrix_cache.load(str(source)) is None


def test_editing_the_source_invalidates_the_cache(cache_home, source):
    """The failure this guards against: re-sorting a matrix from the same
    run writes a NEW file at the SAME path, and a path-only key would
    serve the old array back -- which looks exactly like the program
    ignoring the user's new data."""
    matrix_cache.store(str(source), np.zeros((4, 4), dtype=np.int64))
    assert matrix_cache.load(str(source)) is not None

    # Same path, different contents and mtime.
    os.utime(source, (0, 0))
    source.write_bytes(b"different contents entirely, of a different length")

    assert matrix_cache.load(str(source)) is None


def test_a_truncated_cache_file_is_a_miss_not_a_crash(cache_home, source):
    """A crash mid-write, or a half-copied cache directory, must not take
    the next run down with it."""
    matrix_cache.store(str(source), np.ones((5, 5), dtype=np.int64))
    target = matrix_cache.cached_path(str(source))
    with open(target, "r+b") as handle:
        handle.truncate(12)

    assert matrix_cache.load(str(source)) is None


def test_a_deleted_source_is_a_miss_not_a_crash(cache_home, source):
    matrix_cache.store(str(source), np.ones((3, 3), dtype=np.int64))
    source.unlink()
    assert matrix_cache.load(str(source)) is None


def test_no_temporary_files_are_left_behind(cache_home, source):
    matrix_cache.store(str(source), np.ones((8, 8), dtype=np.int64))
    leftovers = [
        name for name in os.listdir(matrix_cache.cache_dir())
        if name.endswith(".tmp")
    ]
    assert leftovers == []


def test_pruning_evicts_least_recently_used_until_it_fits(cache_home, tmp_path):
    """Without this the cache grows without bound -- every matrix a user
    ever opens, at hundreds of megabytes each."""
    sources = []
    for i in range(4):
        path = tmp_path / f"m{i}.mtx"
        path.write_bytes(bytes([i]) * (i + 1))
        sources.append(path)
        matrix_cache.store(str(path), np.zeros((50, 50), dtype=np.int64))

    directory = matrix_cache.cache_dir()
    assert len(os.listdir(directory)) == 4

    # Make the first two look old, the last two recently used.
    for i, path in enumerate(sources):
        target = matrix_cache.cached_path(str(path))
        stamp = 1000.0 + i * 1000.0
        os.utime(target, (stamp, stamp))

    sizes = [os.path.getsize(matrix_cache.cached_path(str(p))) for p in sources]
    matrix_cache.prune(directory, limit=int(sum(sizes) * 0.6))

    survivors = set(os.listdir(directory))
    # The two most recently used survive; the two oldest are gone.
    assert os.path.basename(matrix_cache.cached_path(str(sources[3]))) in survivors
    assert os.path.basename(matrix_cache.cached_path(str(sources[0]))) not in survivors


def test_the_cache_lives_outside_the_users_data_directory(cache_home, source):
    """Deliberately not beside the matrix, unlike HDTV's .prx/.pry
    sidecars: writing into the directory someone keeps measurements in
    enlarges their backups for a reason they did not choose."""
    matrix_cache.store(str(source), np.ones((3, 3), dtype=np.int64))
    assert os.path.dirname(matrix_cache.cached_path(str(source))) != str(source.parent)
    assert not any(p.suffix == ".npy" for p in source.parent.iterdir())
