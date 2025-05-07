from sqlmodel import create_engine

from config import Config

config = Config()

connect_args = {"check_same_thread": False}

engine = create_engine(config.SQLITE_URL, connect_args=connect_args)
