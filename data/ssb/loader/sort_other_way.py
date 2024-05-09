import duckdb
import argparse


CREATE_LINEORDER_TABLE = '''
        CREATE TABLE lineorder (
        LO_ORDERKEY UINT32, 
        LO_LINENUMBER  UINT8,  
        LO_CUSTKEY  UINT32,  
        LO_PARTKEY  UINT32,
        LO_SUPPKEY  UINT32, 
        LO_ORDERDATE   INT32,  
        LO_ORDERPRIORITY   string, 
        LO_SHIPPRIORITY   UINT8,  
        LO_QUANTITY   INT32,  
        LO_EXTENDEDPRICE   INT32, 
        LO_ORDTOTALPRICE   UINT32,  
        LO_DISCOUNT   INT32,  
        LO_REVENUE   UINT32, 
        LO_SUPPLYCOST   UINT32, 
        LO_TAX   UINT8,  
        LO_COMMITDATE   UINT64, 
        LO_SHIPMODE   string);
        COPY lineorder FROM '{0}' WITH (HEADER false, DELIMITER '|');
        '''

SAVE_SORTED_ORDRERDATE_CUSTKEY = '''
COPY (SELECT * FROM lineorder ORDER BY LO_CUSTKEY, LO_ORDERDATE  ASC) TO '{0}' (HEADER false, DELIMITER '|');
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert')
    parser.add_argument('data_directory', type=str, help='Data Directory')
    args = parser.parse_args()

    data_dir = args.data_directory
    # process lineorder
    input = data_dir + 'lineorder.tbl'
    output = data_dir + 'lineorder.tbl.s'
    con = duckdb.connect()
    con.sql(CREATE_LINEORDER_TABLE.format(input))
    con.sql(SAVE_SORTED_ORDRERDATE_CUSTKEY.format(output))
    con.close()
