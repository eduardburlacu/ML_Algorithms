import os
from db import Database

paths_query = ["SQL/query_all.sql","SQL/query.sql","SQL/query_name.sql", "SQL/query_agg.sql", "SQL/query_groupby.sql"]

def run_query(db: Database, query_file: str) -> None:
    """Run a SQL query from a file and print the results."""
    query = db.query_from_file(query_file)
    results = db.fetchall(query)
    for row in results:
        print(dict(row))


if __name__ == "__main__":
    db = Database("SQL/students.db")
    db.connect()
    for query_file in paths_query:
        print(f"Executing query from file: {query_file}")
        print("-" * 50)
        run_query(db, query_file)
        print("\n")
    db.close()
