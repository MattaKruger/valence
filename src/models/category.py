from sqlmodel import Field, SQLModel


class Base(SQLModel, table=False):
    id: int = Field(primary_key=True)


class CategoryBase(Base, table=False):
    pass


class Category(CategoryBase, table=True):
    pass


class TrackBase(SQLModel, table=False):
    pass


class Track(SQLModel, table=True):
    pass
