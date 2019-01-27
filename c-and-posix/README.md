# C Tools

Some useful and interesting C tools.

## Shuffle

Shuffle 52 cards and print them out

### Usage

```
cd shuffle
make
./shuffle
```

## Every

A function `every(size_t num_runs, size_t interal, fn_t fn);` that runs
function `fn` `num_runs` number of times with `interval` milliseconds apart.

### Usage

```
cd every 
make
./every
```

## Mywc

An implementation of `wc` that counts number of words in a sequence delimetered
by space, tab, new line character or end-of-file.

### Usage

```
cd mywc
make
echo "the quick brown fox jumps over the lazy dog" | ./mywc
```

## Reference

[CSC-231 Lab: Warmup](https://www.cs.grinnell.edu/~curtsinger/teaching/2019S/CSC213/labs/warmup/)
