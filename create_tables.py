from sqlalchemy import create_engine
from assets.models import Base
from assets.const import DB_PARAMS

def create_tables():
    # Create the database engine
    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )
    
    # Create all tables in the database
    Base.metadata.create_all(engine)
    print("Tables created successfully!")

if __name__ == "__main__":
    create_tables()
