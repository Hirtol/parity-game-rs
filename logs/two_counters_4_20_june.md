* Note, these results hold up even in single threaded workloads!
* Smallest BDD variable ordering:
```
0.0s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Using symbolic solver solver=Zielonka                                                                                                                
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructing with register index k=2
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting edge BDD construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting priority/owner BDD construction                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting player vertex set construction                                                                   
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Creating priority BDDs                                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_move construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_i construction
1.6s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructed Register Game elapsed=1.6211499s
1.9s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Converted from parity game to symbolic register game from_vertex=68 rg_vertex=35651584 rg_bdd_nodes=108009 ratio=524288
83.7s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solving done elapsed=81.7642712s
83.7s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solved with recursive calls n=641
83.7s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Results even_wins=34 odd_wins=34
83.7s  INFO pg_cli: Runtime: 83.76s - Peak Memory Usage: 307.5991 MB
```

* Smallest BDD variable ordering, but vertex/register/priority most significant to least significant bit:
```
0.0s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Using symbolic solver solver=Zielonka                                                                                                                
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructing with register index k=2
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting edge BDD construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting priority/owner BDD construction                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting player vertex set construction                                                                   
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Creating priority BDDs                                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_move construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_i construction
1.3s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructed Register Game elapsed=1.3118532s
1.6s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Converted from parity game to symbolic register game from_vertex=68 rg_vertex=35651584 rg_bdd_nodes=146663 ratio=524288
47.9s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solving done elapsed=46.2981033s
47.9s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solved with recursive calls n=641
47.9s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Results even_wins=34 odd_wins=34
47.9s  INFO pg_cli: Runtime: 47.92s - Peak Memory Usage: 361.18683 MB
```

* Fastest solve variable ordering, normal least to most significant bit:
```
0.0s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:load_parity_game: pg_cli::utils: Loaded parity game size=68                                                                                                                
0.0s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Using symbolic solver solver=Zielonka                                                                                                                
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructing with register index k=2
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting edge BDD construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting priority/owner BDD construction                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting player vertex set construction                                                                   
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Creating priority BDDs                                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_move construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_i construction
2.0s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructed Register Game elapsed=2.0298048s
2.4s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Converted from parity game to symbolic register game from_vertex=68 rg_vertex=35651584 rg_bdd_nodes=130133 ratio=524288
28.3s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solving done elapsed=25.8800759s
28.3s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solved with recursive calls n=641
28.3s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Results even_wins=34 odd_wins=34
28.3s  INFO pg_cli: Runtime: 28.34s - Peak Memory Usage: 469.39978 MB
```

* Fastest solve variable ordering, most to least significant bit:
```
0.0s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Using symbolic solver solver=Zielonka                                                                                                                
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructing with register index k=2
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting edge BDD construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::parity_game: Starting priority/owner BDD construction                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting player vertex set construction                                                                   
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Creating priority BDDs                                                                                    
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_move construction                                                                              
0.0s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}:Build Symbolic Register Game: pg_graph::symbolic::register_game: Starting E_i construction
1.8s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Constructed Register Game elapsed=1.8422149s
2.2s DEBUG Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Converted from parity game to symbolic register game from_vertex=68 rg_vertex=35651584 rg_bdd_nodes=155986 ratio=524288
17.1s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solving done elapsed=14.8583271s
17.1s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Solved with recursive calls n=641
17.1s  INFO Solve Parity Game{path=".\\game_examples\\two_counters_4.pg"}: pg_cli::args::solve: Results even_wins=34 odd_wins=34
17.1s  INFO pg_cli: Runtime: 17.16s - Peak Memory Usage: 480.798 MB
```
