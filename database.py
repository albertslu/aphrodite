import sqlite3

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def main():
    database = "profile_matching.db"

    sql_create_profiles_table = """ CREATE TABLE IF NOT EXISTS profiles (
                                        id integer PRIMARY KEY,
                                        name text NOT NULL,
                                        features text NOT NULL
                                    ); """

    conn = create_connection(database)

    if conn is not None:
        create_table(conn, sql_create_profiles_table)
    else:
        print("Error! Cannot create the database connection.")

if __name__ == '__main__':
    main()
