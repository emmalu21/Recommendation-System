

import sqlite3

def create_database(db_path='database.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create movies table
    c.execute('''CREATE TABLE movies
                 (id INTEGER PRIMARY KEY, title TEXT, overview TEXT)''')

    # Create ratings table
    c.execute('''CREATE TABLE ratings
                 (user_id INTEGER, movie_id INTEGER, rating FLOAT,
                  PRIMARY KEY (user_id, movie_id))''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_database()
