from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from src.models.audio_features import AudioFeature


class TagBase(SQLModel, table=False):
    name: str = Field(...)


class Tag(TagBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tracks: List["TrackTag"] = Relationship(back_populates="tag")


class TagOut(TagBase, table=False):
    id: int


class GenreBase(SQLModel, table=False):
    name: str = Field(...)


class Genre(GenreBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tracks: List["Track"] = Relationship(back_populates="genre")


class GenreOut(GenreBase, table=False):
    id: int


class TrackTag(SQLModel, table=True):
    track_id: Optional[int] = Field(
        default=None, foreign_key="track.id", primary_key=True
    )
    tag_id: Optional[int] = Field(default=None, foreign_key="tag.id", primary_key=True)
    track: "Track" = Relationship(back_populates="track_tags")
    tag: Tag = Relationship(back_populates="tracks")


class TrackBase(SQLModel, table=False):
    title: str = Field(...)
    artist: str = Field(index=True)
    duration: Optional[int] = Field(default=None)
    genre_id: Optional[int] = Field(default=None, foreign_key="genre.id")


class Track(TrackBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    genre: Optional[Genre] = Relationship(back_populates="tracks")
    track_tags: List[TrackTag] = Relationship(back_populates="track")
    playlist_tracks: List["TrackPlayList"] = Relationship(back_populates="track")

    audio_feature: Optional["AudioFeature"] = Relationship(back_populates="track")

    file_path: Optional[str] = Field(default=None)
    features: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    @property
    def tags(self) -> List[Tag]:
        return [track_tag.tag for track_tag in self.track_tags]


class FeatureGroup(SQLModel, table=True):
    """Model for grouping tracks by similar features for visualization purposes"""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None

    tempo_min: Optional[float] = None
    tempo_max: Optional[float] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    brightness_min: Optional[float] = None
    brightness_max: Optional[float] = None

    tracks: List["FeatureGroupTrack"] = Relationship(back_populates="feature_group")

    track_count: int = Field(default=0)
    avg_tempo: Optional[float] = None
    avg_energy: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureGroupTrack(SQLModel, table=True):
    feature_group_id: int = Field(foreign_key="featuregroup.id", primary_key=True)
    track_id: int = Field(foreign_key="track.id", primary_key=True)

    feature_group: FeatureGroup = Relationship(back_populates="tracks")
    track: Track = Relationship()

    similarity_score: float = Field(default=1.0)


class WaveForm(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: int = Field(foreign_key="track.id")

    waveform_data: Dict[str, Any] = Field(sa_column=Column(JSON))

    spectrogram_data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )
    chromagram_data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )

    # Metadata
    sample_rate: int
    duration: float
    resolution: int = Field(default=1000)

    track: Track = Relationship(back_populates="waveform")


class FeatureComparison(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    track_a_id: int = Field(foreign_key="track.id")
    track_b_id: int = Field(foreign_key="track.id")

    similarity_score: float

    tempo_similarity: Optional[float] = None
    spectral_similarity: Optional[float] = None
    tonal_similarity: Optional[float] = None
    rhythmic_similarity: Optional[float] = None

    # Metadata
    comparison_method: str = Field(default="euclidean")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    track_a: Track = Relationship(foreign_keys=[track_a_id])
    track_b: Track = Relationship(foreign_keys=[track_b_id])


class TrackOut(TrackBase, table=False):
    id: int
    genre: Optional[Genre] = None
    tags: List["TagOut"] = Field(default_factory=list)


class TrackPlayList(SQLModel, table=True):
    track_id: Optional[int] = Field(
        default=None, foreign_key="track.id", primary_key=True
    )
    playlist_id: Optional[int] = Field(
        default=None, foreign_key="playlist.id", primary_key=True
    )
    position: int = Field(default=0)
    track: Track = Relationship(back_populates="playlist_tracks")
    playlist: "PlayList" = Relationship(back_populates="playlist_tracks")


class PlayListBase(SQLModel, table=False):
    name: str = Field(index=True)


class PlayList(PlayListBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    playlist_tracks: List[TrackPlayList] = Relationship(back_populates="playlist")

    @property
    def tracks(self) -> List[Track]:
        return [
            pt.track for pt in sorted(self.playlist_tracks, key=lambda x: x.position)
        ]


class PlayListOut(PlayListBase, table=False):
    id: int
    tracks: List[TrackOut] = Field(default_factory=list)
