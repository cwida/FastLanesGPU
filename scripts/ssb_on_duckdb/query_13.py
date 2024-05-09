QUERY_13 = """
select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19940204
and lo_orderdate <= 19940210
and lo_discount>=5
and lo_discount<=7
and lo_quantity>=26
and lo_quantity<=35;
  """


def run_query_13(con):
    tbl = con.sql(QUERY_13).fetch_arrow_table()
    print(tbl.to_pandas())
