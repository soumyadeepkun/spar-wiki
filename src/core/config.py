from pathlib import Path

from pydantic import Field, FilePath, field_validator
from pydantic_settings import BaseSettings
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""

    TERM_OFFSETS_PATH: FilePath  # must link to a .npz file
    TERM_WEIGHTS_PATH: FilePath  # must link to a .npy file
    DOC_IDS_PATH: FilePath  # must link to a .npy file

    OFFSETS_KEY: str = Field(default="offsets", description="Key for term offsets in .npz file")
    W_MAX: float = Field(default=8.0, description="Upper bound used during weight quantization")
    QMAX: int = Field(default=255, description="Quantization max value")
    DEFAULT_TOP_K: int = Field(default=10, description="Default number of hits to return")
    MAX_QUERY_BATCH_SIZE: int = Field(default=16, ge=1, description="Maximum query batch size")
    HF_TOKEN: str = Field(..., description="Huggingface access token")
    SPARSE_MODEL_NAME: str = Field(default="naver/splade-v3", description="Sparse encoder model")
    SPARSE_MODEL_DEVICE: str = Field(default="cpu", description="Device for sparse encoder")
    DATABASE_URL: str = Field(
        default="sqlite:///./docs.db",
        description="SQLAlchemy database URL for document metadata",
    )
    DOC_FETCH_CHUNK_SIZE: int = Field(
        default=900,
        ge=1,
        description="Maximum number of doc IDs per SQL IN-clause chunk",
    )

    @field_validator("TERM_OFFSETS_PATH", mode="after")
    @classmethod
    def must_be_npz(cls, v: Path) -> Path:
        if v.suffix != ".npz":
            raise ValueError("TERM_OFFSETS_PATH must point to a .npz file")
        return v

    @field_validator("TERM_WEIGHTS_PATH", "DOC_IDS_PATH", mode="after")
    @classmethod
    def must_be_npy(cls, v: Path) -> Path:
        if v.suffix != ".npy":
            raise ValueError("must point to a .npy file")
        return v

    @field_validator("HF_TOKEN", mode="after")
    @classmethod
    def validate_hf_token(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("HF_TOKEN must be non-empty")
        return v

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        v = v.strip().strip('"').strip("'")
        if (
            v.startswith("sqlite:///")
            and not v.startswith("sqlite:////")
            and v != "sqlite:///:memory:"
        ):
            database = make_url(v).database or ""
            if database:
                absolute_candidate = Path("/") / database
                if absolute_candidate.exists():
                    raise ValueError(
                        "DATABASE_URL looks like an absolute SQLite path but is missing a slash. "
                        f"Use sqlite:////{database}"
                    )
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables in .env but ignore them


# Global settings instance
settings = Settings()  # type: ignore
