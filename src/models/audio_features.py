from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import JSON, Column, Field, Index, Relationship, SQLModel
from track import Track


class AudioFeatureBase(SQLModel, table=False):
    track_id: int = Field(foreign_key="track.id")

    __table_args__ = (
        Index("ix_audio_feature_tempo", "tempo"),
        Index("ix_audio_feature_rms_mean", "rms_mean"),
        Index("ix_audio_feature_spectral_centroid_mean", "spectral_centroid_mean"),
    )

    # Rhythmic features
    tempo: Optional[float] = None
    num_beats: Optional[int] = None
    num_onsets: Optional[int] = None
    onset_rate: Optional[float] = None

    # Spectral features
    spectral_centroid_mean: Optional[float] = None
    spectral_centroid_std: Optional[float] = None
    spectral_bandwidth_mean: Optional[float] = None
    spectral_bandwidth_std: Optional[float] = None
    spectral_contrast_mean: Optional[float] = None
    spectral_contrast_std: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    spectral_rolloff_std: Optional[float] = None

    # Energy features
    rms_mean: Optional[float] = None
    rms_std: Optional[float] = None

    # Tonal features
    zero_crossing_rate_mean: Optional[float] = None
    zero_crossing_rate_std: Optional[float] = None

    # Complex features stored as JSON
    chroma_mean: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    chroma_std: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    mfcc_mean: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    mfcc_std: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tonnetz_mean: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tonnetz_std: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tempogram_mean_vector: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )

    # Metadata
    analysis_version: Optional[str] = None
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class AudioFeature(AudioFeatureBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    track: "Track" = Relationship(back_populates="audio_feature")

    # Add methods for feature comparison and normalization
    def get_feature_vector(self, selected_features=None):
        """
        Returns a normalized vector of features for similarity comparison
        """
        if selected_features is None:
            # Default set of features for comparison
            selected_features = [
                "tempo",
                "spectral_centroid_mean",
                "spectral_bandwidth_mean",
                "rms_mean",
                "zero_crossing_rate_mean",
            ]

        return {feature: getattr(self, feature, None) for feature in selected_features}
