import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Path,
    Request,
    UploadFile,
)
from httpx import AsyncClient
from liblistenbrainz import ListenBrainz
from sqlmodel import Session, SQLModel, select

from audio_features import AudioFeatureExtractor
from config import Config
from db.engine import engine
from db.session import get_session
from deps.clients import get_httpx_client, get_listenbrainz_client
from models.track import AudioFeature, Genre, PlayList, Track, TrackPlayList, TrackTag
from visualization_router import router as visualization_router

load_dotenv()

config = Config()

AUTH_HEADER = {"Authorization": f"Token {config.LISTEN_BRAINZ_TOKEN}"}
MUSIC_ROOT_DIR = "./data"


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(visualization_router)



@app.post("analyze/audio")
async def analyze_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Upload and analyze a single audio file to extract features"""
    os.makedirs("./temp_uploads", exist_ok=True)

    file_path = f"./temp_uploads{file.filename}"

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(
        process_audio_file_enhanced, file_path=file_path, session=session
    )

    return {"message": f"File {file.filename} scheduled for enhanced analysis"}


def process_audio_file_enhanced(
    file_path: str, session: Session, track_id: Optional[int] = None
):
    try:
        extractor = AudioFeatureExtractor()
        features_dict, duration = extractor.extract_features(file_path)

        if not features_dict:
            return

        if track_id:
            track = session.get(Track, track_id)
        else:
            filename = os.path.basename(file_path)
            track_name = os.path.splitext(filename)[0]

            parts = track_name.split(" - ", 1)
            if len(parts) == 2:
                artist, title = parts
            else:
                artist, title = "Unknown", track_name

            track = Track(
                title=title, artist=artist, duration=duration, file_path=file_path
            )
            session.add(track)
            session.commit()
            session.refresh(track)

        track.features = features_dict

        feature_data = {
            "track_id": track.id,
            "analyzed_at": datetime.now(),
            "analysis_version": "1.0",
        }

        scalar_features = [
            "tempo",
            "num_beats",
            "num_onsets",
            "onset_rate",
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_std",
            "spectral_contrast_mean",
            "spectral_contrast_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "rms_mean",
            "rms_std",
            "zero_crossing_rate_mean",
            "zero_crossing_rate_std",
        ]

        for feature in scalar_features:
            if feature in features_dict:
                feature_data[feature] = features_dict[feature]

        audio_feature = AudioFeature.model_dump(**feature_data)
        session.add(audio_feature)
        session.commit()

        if file_path.startswith("./temp_uploads/"):
            os.remove(file_path)

    except Exception as e:
        print(f"Error in enhanced processing of {file_path}: {str(e)}")


@app.post("/analyze/directory")
async def analyze_directory(
    background_tasks: BackgroundTasks,
    directory_path: str,
    output_csv: Optional[str] = "audio_features.csv",
    session: Session = Depends(get_session),
):
    if not os.path.isdir(directory_path):
        raise HTTPException(
            status_code=400, detail=f"Invalid directory path: {directory_path}"
        )

    background_tasks.add_task(
        process_directory,
        directory_path=directory_path,
        output_csv=output_csv,
        session=session,
    )
    return {"message": f"Directory {directory_path} scheduled for processing"}


@app.get("/tracks/features/{track_id}")
async def get_track_features(
    track_id: int,
    session: Session = Depends(get_session),
):
    track = session.get(Track, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if not track.features:
        return {"message": "No features extracted for this track yet"}

    return {"track": track, "features": track.features}


@app.get("/listen/analyze/{username}")
async def analyze_current_listen(
    background_tasks: BackgroundTasks,
    username: str,
    session: Session = Depends(get_session),
    client: ListenBrainz = Depends(get_listenbrainz_client),
):
    response = client.get_playing_now(username)
    if response is None:
        raise HTTPException(status_code=404, detail="No current listen found")

    track_info = {
        "title": response.track_name,
        "artist": response.artist_name,
        "duration": response.additional_info.get("duration", 0),
    }

    matching_file = find_matching_file(track_info["title"], track_info["artist"])

    track = Track.model_dump(**track_info)

    session.add(track)
    session.commit()
    session.refresh(track)

    if matching_file:
        background_tasks.add_task(
            process_audio_file,
            file_path=matching_file,
            session=session,
            track_id=track.id,
        )
        return {
            "track": track,
            "message": f"Found matcing file and scheduled for analysis: {matching_file}",
        }
    else:
        # Get local file from nictone+ api
        return {"track": track, "message": "No matching local found for analysis."}


def find_matching_file(title: str, artist: str) -> Optional[str]:
    import re

    from unidecode import unidecode

    def normalize(s):
        return re.sub(r"[\w\s]", "", unidecode(s.lower()))

    normalized_title = normalize(title)
    normalized_artist = normalize(artist)

    for root, _, files in os.walk(MUSIC_ROOT_DIR):
        for file in files:
            if file.endswith((".mp3", ".wav", ".flac")):
                filename = os.path.splitext(file)[0]
                normalized_filename = normalize(filename)

                if (
                    normalized_title in normalized_filename
                    and normalized_artist in normalized_filename
                ):
                    return os.path.join(root, file)

    return None


def process_audio_file(
    file_path: str, session: Session, track_id: Optional[int] = None
):
    try:
        extractor = AudioFeatureExtractor()
        features, duration = extractor.extract_features(file_path)

        if features:
            if track_id:
                track = session.get(Track, track_id)
                if track:
                    track.features = features
                    track.duration = duration
                    session.add(track)
                    session.commit()
            else:
                filename = os.path.basename(file_path)
                track_name = os.path.splitext(filename)[0]

                parts = track_name.split(" - ", 1)
                if len(parts) == 2:
                    artist, title = parts
                else:
                    artist, title = "Unknown", track_name

                track = Track(
                    title=title,
                    artist=artist,
                    duration=duration,
                    features=features,
                    file_path=file_path,
                )
                session.add(track)
                session.commit()

            if file_path.startswith("./temp_uploads/"):
                os.remove(file_path)
    except Exception as e:
        print(f"Error processing audio file {file_path}: {str(e)}")


def process_directory(
    directory_path: str,
    output_csv: str,
    session: Session,
):
    try:
        extractor = AudioFeatureExtractor()
        df = extractor.analyze_directory(directory_path, output_csv)

        if df is not None:
            for _, row in df.iterrows():
                filename = os.path.basename(row["file_path"])
                track_name = os.path.splitext(filename)[0]

                parts = track_name.split(" - ", 1)
                if len(parts) == 2:
                    artits, title = parts
                else:
                    artist, title = "Uknown", track_name

                features = row.to_dict()

                track = Track(
                    title=title,
                    artist=artist,
                    duration=row.get("duration", 0),
                    features=features,
                    file_path=row["file_path"],
                )
                session.add(track)
            session.commit()
    except Exception as e:
        print(f"Error processing directory {directory_path}: {str(e)}")


@app.get("/{username}/{count}")
async def listens(
    username: str = Path(...),
    count: str = Path(...),
    client: ListenBrainz = Depends(get_listenbrainz_client),
):
    return client.get_listens(username=username, count=count)


@app.get("/{username}")
async def current_listen(
    username: str = Path(...),
    session: Session = Depends(get_session),
    client: ListenBrainz = Depends(get_listenbrainz_client),
):
    response = client.get_playing_now(username)

    if response is None:
        return None

    track = Track(
        title=response.track_name,
        artist=response.artist_name,
        duration=response.additional_info["duration"],
    )
    session.add(track)
    session.commit()
    session.refresh(track)
    return track


@app.get("/tracks/unassigned")
async def get_unassigned_tracks(
    session: Session = Depends(get_session),
):
    statement = select(Track).where(Track.genre_id == None)
    results = session.exec(statement).all()
    return list(results)


@app.get("/top_artists")
async def get_top_artists():
    statement = select(Track.artist)


# @app.post("/{name}")
# async def create_playlist(name: str = Path(...)):
#     listens = []
#     playlist = Playlist(name=name, listens=listens)

#     return playlist


@app.post("/genre")
async def create_genre(
    genre: Genre,
    session: Session = Depends(get_session),
) -> Genre:
    session.add(genre)
    session.commit()
    session.refresh(genre)

    return genre


@app.get("/genre")
async def get_genres(
    session: Session = Depends(get_session),
) -> List[Genre]:
    statement = select(Genre)
    results = session.exec(statement).all()
    return list(results)


@app.post("/tag/{track}")
async def tag_track(
    track_id: int = Path(...),
    tags: List[int] = [],
    session: Session = Depends(get_session),
):
    statement = select(Track).where(id == track_id)
    track = session.exec(statement).one()

    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")

    for tag_id in tags:
        track_tag = TrackTag(track_id=track_id, tag_id=tag_id)
        session.add(track_tag)

    session.commit()
    session.refresh(track)

    return track


async def fetch_track(track_title: str, artist: str):
    pass


async def process_track(track: Track):
    pass


@app.post("/track/classify")
async def classify_track(
    track_id: int,
    session: Session = Depends(get_session),
):
    statement = select(Track).where(id == track_id)
    track = session.exec(statement).one()
    if not track:
        return None

    await process_track(track)


@app.post("/tag/generate")
async def generate_tag():
    pass


@app.post("/track")
async def create_track(
    track: Track,
    session: Session = Depends(get_session),
):
    session.add(track)
    session.commit()
    session.refresh(track)
    return track


@app.get("/track")
async def get_tracks(
    session: Session = Depends(get_session),
) -> List[Track]:
    statement = select(Track)
    tracks = session.exec(statement).all()
    return list(tracks)


@app.post("/playlist")
async def create_playlist(
    playlist: PlayList,
    session: Session = Depends(get_session),
):
    session.add(playlist)
    session.commit()
    session.refresh(playlist)
    return playlist


@app.post("/playlist/add")
async def add_to_playlist(
    track_playlist: TrackPlayList,
    session: Session = Depends(get_session),
):
    session.add(track_playlist)
    session.commit()
    session.refresh(track_playlist)
    return track_playlist


@app.get("/get_listens")
async def get_listens(
    username: str,
    filename: str,
    request: Request,
    min_ts: int | None = None,
    max_ts: int | None = None,
    count: int | None = None,
    client: AsyncClient = Depends(get_httpx_client),
):
    url = f"{config.LISTEN_BRAINZ_BASE_URL}/1/user/{username}/listens"

    params = {}
    if min_ts is not None:
        params["min_ts"] = min_ts
    if max_ts is not None:
        params["max_ts"] = max_ts
    if count is not None:
        params["count"] = min(count, 100)

    try:
        response = await client.get(url, headers=AUTH_HEADER, params=params)
        response.raise_for_status()

        data = response.json()["payload"]["listens"]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail=f"Request to external API timed out: {url}"
        )
    finally:
        await client.aclose()
