import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()  # reads the .env file at the repo root


def get_engine():
    """Build a SQLAlchemy engine from environment config.

    Same env-var idea as your dbt profiles.yml — localhost from the host,
    the DB's Docker service name when run inside the compose network.
    """
    return create_engine(
        "postgresql+psycopg://"
        f"{os.environ['DBT_USER']}:{os.environ['DBT_PASSWORD']}"
        f"@{os.environ.get('DBT_HOST', 'localhost')}:{os.environ.get('DBT_PORT', '5432')}"
        f"/{os.environ['DBT_DBNAME']}"
    )


def load_features(query: str) -> pd.DataFrame:
    """Run a SQL query and return the result as a DataFrame."""
    return pd.read_sql(query, get_engine())