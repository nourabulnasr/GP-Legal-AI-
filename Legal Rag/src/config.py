"""
Configuration management module for Legal RAG System
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class EmbeddingsConfig(BaseModel):
    """Configuration for embedding models"""
    model_config = {"protected_namespaces": ()}
    model_name: str = Field(default="aubmindlab/bert-base-arabertv2")
    fallback_model: str = Field(default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    device: str = Field(default="auto")
    batch_size: int = Field(default=32)


class LLMConfig(BaseModel):
    """Configuration for LLM"""
    model_config = {"protected_namespaces": ()}
    model_name: str = Field(default="LiquidAI/LFM2.5-1.2B-Instruct")
    device: str = Field(default="auto")
    load_in_8bit: bool = Field(default=False)
    max_new_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)
    do_sample: bool = Field(default=True)
    repetition_penalty: float = Field(default=1.1)


class VectorStoreConfig(BaseModel):
    """Configuration for ChromaDB vector store"""
    persist_directory: str = Field(default="./chroma_db")
    collection_name: str = Field(default="egyptian_labor_laws")
    distance_metric: str = Field(default="cosine")
    embedding_function: str = Field(default="arabic")


class RAGConfig(BaseModel):
    """Configuration for RAG retrieval"""
    section_top_k: int = Field(default=10)
    clause_top_k: int = Field(default=5)
    min_similarity_score: float = Field(default=0.5)
    use_mmr: bool = Field(default=True)
    mmr_diversity: float = Field(default=0.3)


class ParsingConfig(BaseModel):
    """Configuration for contract parsing"""
    supported_formats: list = Field(default=["pdf", "docx", "txt"])
    max_file_size_mb: int = Field(default=10)
    preserve_structure: bool = Field(default=True)
    detect_sections: bool = Field(default=True)
    min_clause_length: int = Field(default=20)


class OutputConfig(BaseModel):
    """Configuration for output generation"""
    formats: list = Field(default=["json", "html", "pdf"])
    default_format: str = Field(default="json")
    language: str = Field(default="ar")
    include_law_references: bool = Field(default=True)
    reports_directory: str = Field(default="./reports")


class DataConfig(BaseModel):
    """Configuration for data paths"""
    labor_law_file: str = Field(default="./data/labor14_2025_chunks.cleaned.jsonl")
    sample_contracts_dir: str = Field(default="./data/sample_contracts")


class APIConfig(BaseModel):
    """Configuration for API server"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    enable_cors: bool = Field(default=True)
    allowed_origins: list = Field(default=["*"])
    max_upload_size_mb: int = Field(default=10)


class LoggingConfig(BaseModel):
    """Configuration for logging"""
    level: str = Field(default="INFO")
    file: str = Field(default="legal_rag.log")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_output: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class"""
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "Config":
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object
        """
        if not os.path.exists(yaml_path):
            print(f"Warning: Config file {yaml_path} not found. Using default configuration.")
            return cls()

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def save_yaml(self, yaml_path: str = "config.yaml"):
        """
        Save configuration to YAML file

        Args:
            yaml_path: Path to save YAML configuration file
        """
        config_dict = self.model_dump()

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def get_device(self) -> str:
        """
        Get the device to use: auto | cpu | cuda.
        DEVICE env overrides config. If auto, use cuda only when available.
        """
        device = os.getenv("DEVICE", getattr(self.llm, "device", "auto")).strip().lower()
        if not device:
            device = "auto"

        try:
            import torch
            if device == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                return "cpu"
            return device if device in ("cpu", "cuda") else "cpu"
        except ImportError:
            return "cpu" if device == "auto" else (device if device in ("cpu", "cuda") else "cpu")

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.vector_store.persist_directory,
            self.output.reports_directory,
            self.data.sample_contracts_dir,
            os.path.dirname(self.data.labor_law_file)
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance (singleton pattern)

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    global _config
    if _config is None:
        _config = Config.from_yaml(config_path)
        _config.ensure_directories()
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Reload configuration from file

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    global _config
    _config = Config.from_yaml(config_path)
    _config.ensure_directories()
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully:")
    print(f"LLM Model: {config.llm.model_name}")
    print(f"Embedding Model: {config.embeddings.model_name}")
    print(f"Device: {config.get_device()}")
    print(f"Vector Store: {config.vector_store.collection_name}")
    print(f"Data File: {config.data.labor_law_file}")
