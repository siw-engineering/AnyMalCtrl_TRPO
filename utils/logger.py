# Copyright 2020 EcoSys Group Sdn Bhd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import sqlite3
from sqlite3 import Error
import numpy as np
from typing import Union
import json
from datetime import datetime
from UnderdogEnvs.tasks import CheetaState


def adapt_np_array(arr: np.ndarray):
    return json.dumps(arr.tolist())


def convert_np_array(text):
    return np.array(json.loads(text))


def adapt_state(state: Union[CheetaState]):
    return str(state.name)


def convert_state(state):
    key = state.decode('utf-8')
    if key in CheetaState.__members__:
        return CheetaState[key]
    else:
        return "UnknownType"


class Logger:
    def __init__(self, table_name=None, buffer_size: int = 100, db_file: str = 'env_log3.db'):
        self.table_name = table_name
        """ create a database connection to a SQLite database """
        self.conn = None
        self.db_file = db_file
        try:
            self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, timeout=60.0)
            sqlite3.register_adapter(np.ndarray, adapt_np_array)
            sqlite3.register_converter("np_array", convert_np_array)
            sqlite3.register_adapter(CheetaState, adapt_state)
            sqlite3.register_converter("state", convert_state)
            print(f'Using sqlite3, version: {sqlite3.sqlite_version}')
        except Error as e:
            print(e)

        self.buffer_size = buffer_size
        self.table_created = False
        self.insert_sql_str = None
        self.buffer = []

    def __del__(self):
        self.__store_buffer_into_db()

    def __getstate__(self):
        print("Logger3 getstate")
        return self.buffer, self.buffer_size, self.db_file, self.table_name, self.insert_sql_str

    def __setstate__(self, tup):
        print("Logger3 setstate")
        self.buffer = tup[0]
        self.buffer_size = tup[1]
        self.db_file = tup[2]
        self.table_name = tup[3]
        self.insert_sql_str = tup[4]
        try:
            self.conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES, timeout=60.0)
            sqlite3.register_adapter(np.ndarray, adapt_np_array)
            sqlite3.register_converter("np_array", convert_np_array)
            sqlite3.register_adapter(CheetaState, adapt_state)
            sqlite3.register_converter("state", convert_state)
            print(f'Using sqlite3, version: {sqlite3.sqlite_version}')
            cur = self.conn.cursor()

            get_tables_sql = \
                '''
                SELECT name FROM sqlite_master WHERE type='table'
                ORDER BY name;
                '''
            cur.execute(get_tables_sql)
            table_names = cur.fetchall()
            table_names = [a for a, in table_names]  # Make it a list of string instead of list of tuple
            if self.table_name in table_names:
                self.table_created = True
            else:
                self.table_created = False
        except Error as e:
            print(e)

    def store(self, **kwargs):
        if self.conn is None:
            raise Exception("Connection cannot be none when storing")
        if not self.table_created:
            self.__create_table(**kwargs)

        data_tup = tuple([*kwargs.values()])
        self.buffer.append(data_tup)
        if len(self.buffer) >= self.buffer_size:
            self.__store_buffer_into_db()

    def load(self, sql_statement: str = None):
        if self.table_name is None:
            raise RuntimeError("Table name cannot be None when loading")
        if len(self.buffer) > 0:
            self.__store_buffer_into_db()

        cur = self.conn.cursor()
        if sql_statement is None:
            cur.execute(f"select * from {self.table_name}")
            data = cur.fetchall()
            return data
        else:
            cur.execute(sql_statement)
            data = cur.fetchall()
            return data

    def __create_table(self, **kwargs):
        """
        This method creates the sql table with name = self.table_name.
        If self.table_name=None, set it to be based on date time
        If successful, sets self.table_created = True and creates a corresponding
        sql insert stored in self.insert_sql_str.

        :param kwargs: dictionary of items to store into table
        """
        if self.table_name is None:
            now = datetime.now()
            self.table_name = f'run_{now.strftime("%d_%m_%Y__%H_%M_%S")}'

        # We drop the table if it exists (it shouldn't)
        drop_table_sql = """DROP TABLE %s""" % self.table_name
        try:
            self.conn.execute(drop_table_sql)
        except Error as e:
            print(e)

        print(f'Creating table: {self.table_name}')
        # Generate the sql string to create the table
        create_table_sql = f"CREATE TABLE {self.table_name} (id integer PRIMARY KEY"
        var_list = []
        for k, v in kwargs.items():
            val_type = self.__determine_types(v)
            var_sql = f', {k} {val_type} NOT NULL'
            var_list.append(k)
            create_table_sql += var_sql
        create_table_sql += ", date_time text NOT NULL"
        var_list.append('date_time')
        create_table_sql += ");"

        # Generate the insertion sql string
        insert_sql = f"insert into {self.table_name} ({', '.join(var_list)}) VALUES "
        question_mark_list = ['?' for _ in var_list]
        question_mark_list[-1] = "datetime('now', 'localtime')"
        insert_sql += f"( {', '.join(question_mark_list)})"
        self.insert_sql_str = insert_sql

        try:
            self.conn.execute(create_table_sql)
            self.table_created = True
            print(f'Table {self.table_name} created')
        except Error as e:
            print(e)
            self.table_created = False

    def __store_buffer_into_db(self):
        if len(self.buffer) == 0:
            return
        cur = self.conn.cursor()
        try:
            cur.executemany(self.insert_sql_str, self.buffer)
            self.conn.commit()
            self.buffer = []
        except Error as e:
            print(e)

    @staticmethod
    def __determine_types(val) -> str:
        if isinstance(val, int):
            return "integer"
        elif isinstance(val, float):
            return "real"
        elif isinstance(val, np.ndarray):
            return "np_array"
        elif isinstance(val, CheetaState):
            return "state"
        else:
            raise Exception("Unknown type passed to logger")
