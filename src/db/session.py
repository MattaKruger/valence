from sqlmodel import Session, create_engine

from src.config import Config

config = Config()


def get_session() -> Session:
    engine = create_engine(config.SQLITE_URL)
    session = Session(engine)
    return session
