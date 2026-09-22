"""Central configuration for all pipeline tunables (docs 03–09)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class DetectionConfig:
    """Kraken line detection settings (doc 03)."""

    max_dimension: int = 2048
    detector_name: str = "kraken-blla"
    model_path: Path = Path("blla.mlmodel")


@dataclass(frozen=True)
class ReadingOrderConfig:
    """Row-grouping tolerance for reading-order reconstruction (doc 04)."""

    row_grouping_tolerance_fraction: float = 0.5


@dataclass(frozen=True)
class CropConfig:
    """Line cropping settings (doc 05)."""

    padding_px: int = 4
    min_crop_size_px: int = 8
    process_pool_workers: int = 2


@dataclass(frozen=True)
class RecognitionConfig:
    """TrOCR inference settings (doc 06)."""

    model_id: str = "microsoft/trocr-base-handwritten"
    batch_size: int = 4
    min_batch_size: int = 1
    beam_width: int = 4
    use_fp16_on_cuda: bool = True


@dataclass(frozen=True)
class SpellConfig:
    """SymSpell correction settings (doc 07)."""

    enabled: bool = True
    dictionary_path: Optional[Path] = None
    protected_vocab_path: Optional[Path] = None
    max_edit_distance: int = 2
    min_word_length: int = 3


@dataclass(frozen=True)
class ConfidenceConfig:
    """Word confidence aggregation settings (doc 08)."""

    aggregation: str = "geometric_mean"


@dataclass(frozen=True)
class QueueConfig:
    """GPU inference queue bounds (doc 09)."""

    max_depth: int = 8
    worker_count: int = 1


@dataclass(frozen=True)
class AppConfig:
    """Top-level config bundle passed through the pipeline."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reading_order: ReadingOrderConfig = field(default_factory=ReadingOrderConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    spell: SpellConfig = field(default_factory=SpellConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)


class Settings(BaseSettings):
    """Environment-variable overrides for deployment knobs."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    detection_max_dimension: int = 2048
    kraken_model_path: str = "blla.mlmodel"
    crop_padding_px: int = 4
    min_crop_size_px: int = 8
    process_pool_workers: int = 2
    row_grouping_tolerance_fraction: float = 0.5

    trocr_model_id: str = "microsoft/trocr-base-handwritten"
    trocr_batch_size: int = 4
    trocr_beam_width: int = 4
    trocr_use_fp16_on_cuda: bool = True

    spell_correction_enabled: bool = True
    spell_dictionary_path: Optional[str] = None
    protected_vocab_path: Optional[str] = None
    spell_max_edit_distance: int = 2
    spell_min_word_length: int = 3

    queue_max_depth: int = 8
    gpu_worker_count: int = 1

    def to_app_config(self) -> AppConfig:
        spell_dict = (
            Path(self.spell_dictionary_path) if self.spell_dictionary_path else None
        )
        protected = (
            Path(self.protected_vocab_path) if self.protected_vocab_path else None
        )
        return AppConfig(
            detection=DetectionConfig(
                max_dimension=self.detection_max_dimension,
                model_path=Path(self.kraken_model_path),
            ),
            reading_order=ReadingOrderConfig(
                row_grouping_tolerance_fraction=self.row_grouping_tolerance_fraction,
            ),
            crop=CropConfig(
                padding_px=self.crop_padding_px,
                min_crop_size_px=self.min_crop_size_px,
                process_pool_workers=self.process_pool_workers,
            ),
            recognition=RecognitionConfig(
                model_id=self.trocr_model_id,
                batch_size=self.trocr_batch_size,
                beam_width=self.trocr_beam_width,
                use_fp16_on_cuda=self.trocr_use_fp16_on_cuda,
            ),
            spell=SpellConfig(
                enabled=self.spell_correction_enabled,
                dictionary_path=spell_dict,
                protected_vocab_path=protected,
                max_edit_distance=self.spell_max_edit_distance,
                min_word_length=self.spell_min_word_length,
            ),
            queue=QueueConfig(
                max_depth=self.queue_max_depth,
                worker_count=self.gpu_worker_count,
            ),
        )


def load_config() -> AppConfig:
    """Load config from environment with sane 4 GB GPU defaults."""
    return Settings().to_app_config()
