- how to generate data for `sf=10`
    ```shell  
     pip3 install duckdb
     make -C ssb/dbgen/
     make -C ssb/loader/
     python3 util.py ssb 10 gen 
     chmod 777 ssb/data/s10/date.tbl
     python3 util.py ssb 10 transform
     python3 util.py ssb 10 sort
     echo "change BASE_PATH in crystal_ssb_utils.h and ssb_utils to the right path"
    ```
  
- temp
  ```shell
     python3 util.py ssb 10 sort_other_way // experimental

 ```