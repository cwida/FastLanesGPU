QUERY_11 = """
  select sum(lo_extendedprice * lo_discount) as revenue
  from lineorder
  where lo_orderdate >= 19930101 and lo_orderdate <= 19940101 and lo_discount>=1
  and lo_discount<=3
  and lo_quantity<25;
  """

# the result of this query match fls_q11
MODIFIED_QUERY_11 = """
  select sum(lo_extendedprice * lo_discount) as revenue
  from lineorder
  where lo_orderdate >= 19930000 and lo_orderdate <= 19940000 and lo_discount>=1
  and lo_discount<=3
  and lo_quantity<25;
  """


def run_query_11(con):
    tbl = con.sql(QUERY_11).fetch_arrow_table()
    print(tbl.to_pandas())


def run_modified_query_11(con):
    tbl = con.sql(MODIFIED_QUERY_11).fetch_arrow_table()
    print(tbl.to_pandas())
