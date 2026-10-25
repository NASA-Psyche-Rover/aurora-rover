"""Channel reader counting: producers pause when nobody reads a stream."""
import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip("rclpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _channel():
    spec = importlib.util.spec_from_file_location("dash_server", os.path.join(os.path.dirname(__file__), "..", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Channel(b"placeholder")


def test_readers_counts_open_streams_and_releases_on_close():
    ch = _channel()
    a, b = ch.mjpeg(), ch.mjpeg()
    next(a), next(b)
    assert ch.readers == 2
    a.close()
    assert ch.readers == 1
    b.close()
    assert ch.readers == 0


def test_watched_follows_polling_and_readers():
    ch = _channel()
    assert not ch.watched()
    ch.next_frame(-1)           # a page asking for a frame counts as watching
    assert ch.watched()
    ch.last_poll -= 5
    assert not ch.watched()
    g = ch.mjpeg()
    next(g)
    assert ch.watched()
    g.close()
    assert not ch.watched()
