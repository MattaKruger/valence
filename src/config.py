from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    LISTEN_BRAINZ_TOKEN: str = Field(default="")
    LISTEN_BRAINZ_USERNAME: str = Field(default="")
    LISTEN_BRAINZ_BASE_URL: str = Field(default="")

    MONGODB_URL: str = Field(default="")

    GOOGLE_AI_API_KEY: str = Field(default="")

    # Spotify
    CLIENT_ID: str = Field(default="")
    CLIENT_SECRET: str = Field(default="")

    SQLITE_URL: str = Field(default="")
    SQLITE_FILENAME: str = Field(default="")
