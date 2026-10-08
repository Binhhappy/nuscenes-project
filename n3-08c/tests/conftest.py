import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def nusc():
    from utils.nuscenes_loader import load_nuscenes
    return load_nuscenes()
