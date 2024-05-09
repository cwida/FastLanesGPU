QUERY_21 = """
select sum(lo_revenue),d_year,p_brand1
from lineorder,part,supplier,ddate
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 1
and s_region = 1
group by d_year,p_brand1;
  """


def run_query_21(con):
    tbl = con.sql(QUERY_21).fetch_arrow_table()
    print(tbl.to_pandas())
