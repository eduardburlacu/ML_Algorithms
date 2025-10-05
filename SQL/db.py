import os
import sqlite3 as sqlite
from sqlite3 import Connection, Cursor
from typing import Any, List, Optional, Tuple, Union

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: Optional[Connection] = None

    def connect(self) -> None:
        """Establish a connection to the SQLite database."""
        if self.connection is None:
            self.connection = sqlite.connect(self.db_path)
            self.connection.row_factory = sqlite.Row

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(self, query: str, params: Optional[Union[Tuple[Any, ...], List[Any]]] = None) -> Cursor:
        """Execute a SQL query with optional parameters."""
        if self.connection is None:
            raise RuntimeError("Database connection is not established.")
        
        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor

    def executemany(self, query: str, params_list: Optional[List[Union[Tuple[Any, ...], List[Any]]]] = None) -> None:
        if self.connection is None:
            raise RuntimeError("Database connection is not established.")
        
        cursor = self.connection.cursor()
        if params_list:
            cursor.executemany(query, params_list)
        else:
            cursor.execute(query)

    def fetchall(self, query: str, params: Optional[Union[Tuple[Any, ...], List[Any]]] = None) -> List[sqlite.Row]:
        """Fetch all rows from a SQL query."""
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def fetchone(self, query: str, params: Optional[Union[Tuple[Any, ...], List[Any]]] = None) -> Optional[sqlite.Row]:
        """Fetch a single row from a SQL query."""
        cursor = self.execute(query, params)
        return cursor.fetchone()

    def commit(self) -> None:
        """Commit the current transaction."""
        if self.connection:
            self.connection.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.connection:
            self.connection.rollback()

    @staticmethod
    def query_from_file(file_path: str) -> str:
        """Read a SQL query from a file and return it as a string."""
        with open(file_path, 'r') as file:
            return file.read()
    
    def insert(self, table: str, data: dict) -> int:
        """Insert a new record into the specified table."""
        keys = ', '.join(data.keys())
        placeholders = ', '.join('?' for _ in data)
        query = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
        cursor = self.execute(query, tuple(data.values()))
        self.commit()
        return cursor.lastrowid
    
    def insert_many(self, table: str, data_list: List[Tuple]) -> None:
        """Insert multiple records into the specified table."""
        if not data_list:
            return
        placeholders = ", ".join("?" for _ in data_list[0])
        query = f"INSERT INTO {table} VALUES ({placeholders})"
        self.executemany(query, data_list)
        self.commit()

if __name__ == "__main__":
    db = Database("SQL/students.db")
    db.connect()
    paths = {
        "create": "SQL/create.sql",
        "insert": "SQL/insert.sql",
        "update": [f"SQL/update_{i}.sql" for i in range(1, 3)],
        "delete": [f"SQL/delete_{i}.sql" for i in range(1, 3)],
        "query": ["SQL/query.sql","SQL/query_name.sql"],
        "db": "SQL/students.db"
    }
    create = db.query_from_file(paths["create"])
    db.execute(create)
    db.commit()

    insert = db.query_from_file(paths["insert"])
    db.execute(insert)
    db.commit()

    db.insert_many("students", [
        ('Hannah', 23, 128, 3.4),
        ('Ed', 15, 138, 3.7),
    ])

    for u in paths["update"]:
        update = db.query_from_file(u)
        db.execute(update)
        db.commit()
    
    for d in paths["delete"]:
        delete = db.query_from_file(d)
        db.execute(delete)
        db.commit()

    for q in paths["query"]:
        print(f"Executing query from file: {q}")
        print("-" * 50)
        query = db.query_from_file(q)
        results = db.fetchall(query)
        for row in results:
            print(dict(row))

    db.close()
