import httpx
import liblistenbrainz
import spotipy
from qbittorrent import Client
from spotipy.oauth2 import SpotifyClientCredentials

from src.config import Config

config = Config()


def get_listenbrainz_client():
    return liblistenbrainz.ListenBrainz()


def get_spotify_client():
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=config.CLIENT_ID, client_secret=config.CLIENT_SECRET
        )
    )


def get_qb_client():
    return Client("http://localhost:8080/")


def get_httpx_client():
    return httpx.AsyncClient(timeout=10.0)
