Crystal-Opt GPU Library
=================

The Crystal-Opt library makes additional changes to the original Crystal library for better performance. The original Crystal library implements a collection of block-wide device functions that can be used to implement high performance implementations of SQL queries on GPUs.

You can also refer to the original Crystal library and their papers [here](https://github.com/anilshanbhag/crystal).

Usage
----

```
# Generate the test data and transform into columnar layout
# Substitute <SF> with appropriate scale factor (eg: 1)
python util.py ssb <SF> gen
python util.py ssb <SF> transform
```

* Configure the benchmark settings
```
cd src/ssb/
# Edit SF and BASE_PATH in ssb_utils.h
```

* To run a query, say run q11
```
make bin/ssb/q11
./bin/ssb/q11
```

