Need to verify the following, but running the following command:

* `pg_cli.exe s .\game_examples\amba_decomposed_arbiter_10.tlsf.ehoa.pg symbolic -r 1 -t symbolic` (1)

Will output the following when running as the only 'main' program:

```
0.1s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:load_parity_game: pg_cli::utils: Loaded parity game size=92209
0.1s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Using symbolic solver solver=Zielonka
0.1s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Constructing with register index k=1
0.1s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting edge BDD construction
83.9s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting priority/owner BDD construction
86.9s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting player vertex set construction
86.9s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Creating priority BDDs
86.9s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_move construction
86.9s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_i construction
87.7s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Constructed Register Game elapsed=87.5996436s
87.7s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Converted from parity game to symbolic register game from_vertex=92209 rg_vertex=47211008 rg_bdd_nodes=30183743 ratio=512     
109.4s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Solving done elapsed=21.7763885s
109.4s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Solved with recursive calls n=24
109.6s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Results even_wins=92204 odd_wins=5
109.6s  INFO pg_cli: Runtime: 109.66s - Peak Memory Usage: 2185.081 MB
```

But when the system is fully loaded (like when running the variable order discovery):

* `cargo test --workspace --release --lib symbolic::register_game::test_helpers::tests::test_variable_orders_grouped -- --nocapture`

Then suddenly, running the same command as in (1) gives the following output:

```
0.1s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:load_parity_game: pg_cli::utils: Loaded parity game size=92209
0.1s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Using symbolic solver solver=Zielonka                                                                                          
0.1s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Constructing with register index k=1
0.1s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting edge BDD construction                                                        
28.0s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting priority/owner BDD construction
28.4s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting player vertex set construction
28.6s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Creating priority BDDs                                                             
28.6s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_move construction                                                       
28.6s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_i construction
29.0s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Constructed Register Game elapsed=28.9132189s
32.0s DEBUG Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Converted from parity game to symbolic register game from_vertex=92209 rg_vertex=47211008 rg_bdd_nodes=1355639 ratio=512      
42.6s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Solving done elapsed=10.5798072s
42.6s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Solved with recursive calls n=24
42.6s  INFO Solve Parity Game{path=".\\game_examples\\amba_decomposed_arbiter_10.tlsf.ehoa.pg"}: pg_cli::args::solve: Results even_wins=92204 odd_wins=5
42.7s  INFO pg_cli: Runtime: 42.70s - Peak Memory Usage: 1897.4352 MB
```

Literally `2.5` times faster?!
The only thing I can think of is that we might be allocating too many threads to the BDD manager, causing the CPUs to
stall and sit without work for too long.
This could subsequently cause a downclock of the CPU core. When we've fully loaded the system with work this shouldn't
happen, and they should stay at their boost-clock.
No clue if that's true, but it sounds somewhat plausible!