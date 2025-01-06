# PG-RS

A parity game solver written during the writing of my Master Thesis.
It implements several exponential and two quasi-polynomial algorithms.
The solver is not meant for general use, please refer to [Oink](https://github.com/trolando/oink) for a general-use solver.

## Running
Please ensure you have a nightly Rust installation.
Clone the repository, `cd` into the directory, and subsequently run the command shown below to solve the `trivial.pg` with the exponential Zielonka solver.

`cargo run --release -- solve ./game_examples/trivial.pg explicit zielonka`

For a list of all commands simply run:

`cargo run --release`