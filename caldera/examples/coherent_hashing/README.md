# coherent_hashing

This is an implementation of [Coherent Parallel Hashing](http://ggg.udg.edu/publicacions/UsersWebs/cohash_siga2011/index.shtml), which I think this is a useful tool to have in your GPU programming toolbox.

Run using:

```
make && cargo run --release --example coherent_hashing
```

TODO:

- [ ] Visualise table contents
- [ ] Max age table (keep optional)
- [ ] Visualise age histogram
- [ ] Maybe table size controls? (Handle failure properly when set too small)
- [ ] More README
