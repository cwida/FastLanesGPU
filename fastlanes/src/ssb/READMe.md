# Queries

- q11 :
    - v1 : is the best, used for paper
    - v2 : SIMDIZED **does not work**
    - v3 : MULTIPLE CHECK
    - v4 : v1 with 8 value at a time **not complete yet**

- q21 :
    - v1
    - v2: 8 value at a time
    - v3: + predicate load on uncompressed data
    - v4 : sorted data **not complete yet**

- q31 :
    - v1 :
    - v2 : combination of shared + register
    - v3 : 8 value at a time
    - v4 : v3 + predicate load on uncompressed data
    - v5 : v4 + sorted data

- q41:
    - v3 : SORTED + FOR ON ORDERDATE
    - v4 : SORTED + FOR ON ORDERDATE and CUSTKEY

---

# Optimizations

- **predicate load on uncompressed data** :
- **8 value at a time**
- **SIMDIZED**
- **MULTIPLE CHECK**