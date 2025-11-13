from pathlib import Path

from chap_pymc.configs.chap_config import ChapConfig

config_dir = Path(__file__).parent.parent / 'configurations'


def test_read_config():
    ChapConfig.model_validate_json(open(config_dir / 'ar_config.json').read())
