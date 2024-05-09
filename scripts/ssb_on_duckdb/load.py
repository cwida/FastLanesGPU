SFs = {
    #
    1: "1"
}

CREATE_CUSTOMER_QUERY = """
  CREATE TABLE customer AS SELECT *
  FROM
  read_csv('{0}/customer.tbl'
     ,AUTO_DETECT=TRUE
     ,delim = '|')
      """

CREATE_DATE_TABLE = '''
        CREATE TABLE ddate (
        d_datekey integer not null,
       d_date char(18) not null,
       d_dayofweek char(9) not null,
       d_month char(9) not null,
       d_year integer not null,
       d_yearmonthnum integer not null,
       d_yearmonth char(7) not null,
       d_daynuminweek integer not null,
       d_daynuminmonth integer not null,
       d_daynuminyear integer not null,
       d_monthnuminyear integer not null,
       d_weeknuminyear integer not null,
       d_sellingseasin varchar(12) not null,
       d_lastdayinweekfl integer not null,
       d_lastdayinmonthfl integer not null,
       d_holidayfl integer not null,
       d_weekdayfl integer not null);
       COPY ddate FROM '{0}/date.tbl' WITH (HEADER false, DELIMITER '|');
    '''

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
        COPY lineorder FROM '{0}/lineorder.tbl' WITH (HEADER false, DELIMITER '|');
        '''

CREATE_PART_TABLE = '''
        CREATE TABLE part (
        P_PARTKEY       UINT32,
        P_NAME          String,
        P_MFGR          String,
        P_CATEGORY      String,
        P_BRAND1        String,
        P_COLOR         String,
        P_TYPE          String,
        P_SIZE          UINT32,
        P_CONTAINER     String
           );
       COPY part FROM '{0}/part.tbl.p' WITH (HEADER false, DELIMITER '|');
    '''

CREATE_SUPPLIER_TABLE = '''
        CREATE TABLE supplier (
        S_SUPPKEY       UINT32,
        S_NAME          String,
        S_ADDRESS       String,
        S_CITY          String,
        S_NATION        String,
        S_REGION        String,
        S_PHONE         String
           );
       COPY supplier FROM '{0}/supplier.tbl.p' WITH (HEADER false, DELIMITER '|');
    '''


def load_table(con, sf):
    relative_path = "../../gpu/data/ssb/data/s{0}/".format(sf)
    con.sql(CREATE_CUSTOMER_QUERY.format(relative_path))
    con.sql(CREATE_LINEORDER_TABLE.format(relative_path))
    con.sql(CREATE_PART_TABLE.format(relative_path))
    con.sql(CREATE_SUPPLIER_TABLE.format(relative_path))
    con.sql(CREATE_DATE_TABLE.format(relative_path))
