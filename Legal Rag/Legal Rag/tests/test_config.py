"""
Tests for configuration module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, get_config


class TestConfig:
    """Test configuration management"""

    def test_default_config_creation(self):
        """Test creating config with defaults"""
        config = Config()

        assert config.llm.model_name == "LiquidAI/LFM2.5-1.2B-Instruct"
        assert config.embeddings.model_name == "aubmindlab/bert-base-arabertv2"
        assert config.vector_store.collection_name == "egyptian_labor_laws"
        assert config.rag.section_top_k == 10
        assert config.rag.clause_top_k == 5

    def test_config_from_yaml(self):
        """Test loading config from YAML file"""
        config = Config.from_yaml("config.yaml")

        assert config.llm.model_name is not None
        assert config.vector_store.persist_directory is not None
        assert isinstance(config.rag.section_top_k, int)

    def test_device_detection(self):
        """Test device detection (cuda/cpu)"""
        config = Config()
        device = config.get_device()

        assert device in ["cuda", "cpu"]

    def test_ensure_directories(self):
        """Test directory creation"""
        config = Config()
        config.ensure_directories()

        # Check that directories are created
        assert Path(config.vector_store.persist_directory).parent.exists()

    def test_singleton_pattern(self):
        """Test that get_config returns same instance"""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
