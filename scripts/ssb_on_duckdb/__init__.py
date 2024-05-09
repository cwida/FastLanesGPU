import duckdb

from .load import *
from .query_11 import *
from .query_12 import *
from .query_13 import *
from .query_21 import *


class Query:
    name = " "
    run = ""

    def __init__(self, query, run):
        self.name = query
        self.run = run


Query11 = Query("query_11", run_modified_query_11)
Query12 = Query("query_12", run_query_12)
Query13 = Query("query_13", run_query_13)
Query21 = Query("query_21", run_query_21)


def run_query(query):
    sf = [1]
    print("-- name : {0}".format(query.name))

    for sf in SFs:
        print("---- sf : {0}".format(SFs[sf]))
        con = duckdb.connect()
        load_table(con, SFs[sf])
        query.run(con)
