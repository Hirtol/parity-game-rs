use ahash::{HashMap, HashMapExt};
use itertools::Itertools;
use pg_graph::explicit::ParityGraph;
use pg_graph::symbolic::{SymbolicParityGame, BDD};
use pg_graph::NodeIndex;
use std::collections::VecDeque;
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct TestingCommand {
    /// The `.pg` file to load
    game_path: PathBuf,
    /// Whether to print which vertices are won by which player.
    #[clap(short, global = true)]
    print_solution: bool,
}

macro_rules! timed_solve {
    ($to_run:expr) => {
        timed_solve!($to_run, "Solving done")
    };
    ($to_run:expr, $text:expr) => {
        {
            let now = std::time::Instant::now();
            let out = $to_run;
            tracing::info!(elapsed=?now.elapsed(), $text);
            (out, now.elapsed())
        }
    };
}

impl TestingCommand {
    #[tracing::instrument(name="Test Buckets", skip(self))]
    pub fn run(self) -> eyre::Result<()> {
        let parity_game = pg_graph::load_parity_game(&self.game_path)?;
        // let mut buckets = HashMap::new();
        // let mut subset_buckets = HashMap::new();
        // let mut similarity_matrix = vec![vec![0u8; parity_game.vertex_count()]; parity_game.vertex_count()];
        let mut similarity_matrix = HashMap::new();


        // Calculate similarity matrix
        for v in parity_game.vertices_index() {
            let edges = parity_game.edges(v).collect::<ahash::HashSet<NodeIndex>>();

            for u in parity_game.vertices_index() {
                let mut u_edges = parity_game.edges(u).collect::<ahash::HashSet<NodeIndex>>();
                let intersections = edges.intersection(&u_edges).count() as u8;

                if intersections > 0 {
                    similarity_matrix.entry(v).and_modify(|ac: &mut HashMap<_, _> | {
                        ac.insert(u, intersections);
                    }).or_insert_with(|| {
                        let mut hash = HashMap::new();
                        hash.insert(u, intersections);
                        hash
                    });
                }
                // if u_edges.is_subset(edges)
                // if u_edges.iter().all(|e| edges.contains(&e)) {
                //     subset_buckets.entry(v).and_modify(|v| *v += 1).or_insert(1u32);
                //
                //     // if edges.iter().all(|e| u_edges.contains(&e)) {
                //     //     share_all_buckets.entry(v).and_modify(|v| *v += 1).or_insert(1u32);
                //     // }
                // }

                // similarity_matrix[v.index()][u.index()] = edges.intersection(&u_edges).count() as u8;
            }
        }

        // let encoding = Self::recursive_split(parity_game.vertices_index(), &similarity_matrix);
        // let encoding = encoding.into_iter().map(|v| (v.0.index(), v.1)).collect::<ahash::HashMap<usize, u32>>();
        // tracing::debug!("Encoding: {:#?}", encoding);
        // 
        // // println!("Similarity: {:#?}", similarity_matrix);
        // let spg = SymbolicParityGame::<BDD>::fr(&parity_game, &encoding)?;
        // let other = SymbolicParityGame::<BDD>::from_explicit(&parity_game)?;
        // spg.gc();
        // other.gc();
        // tracing::debug!(nodes = spg.bdd_node_count(), "Created symbolic game");
        // tracing::debug!(nodes = other.bdd_node_count(), "Other symbolic game");
        Ok(())
    }

    fn recursive_split(vertices: impl Iterator<Item=NodeIndex>, similarity_matrix: &HashMap<NodeIndex, HashMap<NodeIndex, u8>>) -> HashMap<NodeIndex, u32> {
        let mut counter = 0;
        let mut encoding = HashMap::new();
        let vertices = vertices.collect_vec();
        let similarities = similarity_matrix.iter().flat_map(|(a,b)| {
            b.iter().map(|(c, d)| {
                Similarity {
                    vertex_a: *a,
                    vertex_b: *c,
                    score: *d,
                }
            })
        }).collect_vec();
        let (most, least) = Self::partition(&vertices, &similarities, similarity_matrix);
        let mut queue = VecDeque::new();

        for id in &most {
            encoding.entry(*id).or_insert(0u32);
        }
        for id in &least {
            encoding.entry(*id).and_modify(|b| *b |= 1 << counter).or_insert(1);
        }
        counter += 1;

        queue.push_back((counter, most));
        queue.push_back((counter, least));

        while let Some((counter, next)) = queue.pop_front() {
            if next.len() < 2 {
                continue;
            } else {
                tracing::debug!("NEXT: {}",next.len());
            }

            let (most, least) = Self::partition(&next, &similarities, similarity_matrix);

            for id in &most {
                encoding.entry(*id).or_insert(0u32);
            }
            for id in &least {
                encoding.entry(*id).and_modify(|b| *b |= 1 << counter).or_insert(1);
            }

            queue.push_back((counter + 1, most));
            queue.push_back((counter + 1, least));
        }

        encoding
    }

    // let mut share_all_buckets = HashMap::new();

    // for v in parity_game.vertices_index() {
    //     let edges = parity_game.edges(v).collect_vec();
    //
    //     for u in parity_game.vertices_index() {
    //         let mut u_edges = parity_game.edges(u).collect_vec();
    //         if u_edges.iter().all(|e| edges.contains(&e)) {
    //             subset_buckets.entry(v).and_modify(|v| *v += 1).or_insert(1u32);
    //
    //             // if edges.iter().all(|e| u_edges.contains(&e)) {
    //             //     share_all_buckets.entry(v).and_modify(|v| *v += 1).or_insert(1u32);
    //             // }
    //         }
    //     }
    //
    // }
    //
    //
    //
    // for edge in parity_game.graph_edges() {
    //     buckets.entry(edge.target()).and_modify(|v| *v += 1).or_insert(1u32);
    // }
    //
    // tracing::debug!("Incoming references: ");
    // let out = buckets.into_iter().sorted_by_key(|k| k.1).rev().take(20).collect_vec();
    //
    // tracing::debug!(?out);
    //
    // tracing::debug!("Subset: ");
    // let out = subset_buckets.iter().sorted_by_key(|k| *k.1).rev().take(20).collect_vec();
    //
    // tracing::debug!(?out);
    //
    // if let Some((&v_idx, count)) = subset_buckets.iter().sorted_by_key(|k| *k.1).rev().next() {
    //     let edges = parity_game.edges(v_idx).collect_vec();
    //
    //     for u in parity_game.vertices_index() {
    //         let mut u_edges = parity_game.edges(u).collect_vec();
    //         if u_edges.iter().all(|e| edges.contains(&e)) {
    //             tracing::debug!(?v_idx, u_idx=?u, edges=?u_edges, "Subset vertices");
    //             subset_buckets.entry(v_idx).and_modify(|v| *v += 1).or_insert(1u32);
    //         }
    //     }
    // }

    // tracing::debug!("Share all: ");
    // let out = share_all_buckets.iter().sorted_by_key(|k| *k.1).rev().take(20).collect_vec();

    // tracing::debug!(?out);

    fn partition(vertices: &[NodeIndex], similarities: &[Similarity], sim_matrix: &ahash::HashMap<NodeIndex, ahash::HashMap<NodeIndex, u8>>) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
        if vertices.len() < 2 {
            return (vertices.to_vec(), vec![]);
        }

        let mut most_similar = Vec::with_capacity(vertices.len() / 2);
        let mut least_similar = Vec::with_capacity(vertices.len() / 2);
        let mut shrunken_matrix: Vec<&Similarity> = similarities.iter().filter(|v| vertices.contains(&v.vertex_a) && vertices.contains(&v.vertex_b))
            .collect();
        shrunken_matrix.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let mut similarity_map: HashMap<_, _> = HashMap::new();
        for sim in shrunken_matrix.iter() {
            similarity_map.insert((sim.vertex_a, sim.vertex_b), sim.score);
            similarity_map.insert((sim.vertex_b, sim.vertex_a), sim.score);
        }

        // Start by taking the pair of vertices with the highest similarity score.
        let highest_similarity = shrunken_matrix[0];
        most_similar.push(highest_similarity.vertex_a);
        least_similar.push(highest_similarity.vertex_b);
        tracing::debug!("SIZE: {}", vertices.len());
        // Assign the rest of the vertices to the closest group.
        for &vertex in vertices.iter() {
            if most_similar.contains(&vertex) || least_similar.contains(&vertex) {
                continue;
            }

            // Calculate the similarity to group_1 and group_2.
            let sim_to_g1: usize = most_similar
                .iter()
                .map(|&v1| *similarity_map.get(&(vertex, v1)).unwrap_or(&0) as usize)
                .sum();

            let sim_to_g2: usize = least_similar
                .iter()
                .map(|&v2| *similarity_map.get(&(vertex, v2)).unwrap_or(&0) as usize)
                .sum();

            // Assign the vertex to the group with the higher similarity.
            if sim_to_g1 >= sim_to_g2 {
                most_similar.push(vertex);
            } else {
                least_similar.push(vertex);
            }
        }
        
        tracing::debug!("{:?}", most_similar);
        tracing::debug!("{:?}", least_similar);

        // for (i, id) in vertices.iter().sorted_by_key(|v| similarity_matrix.get(v).map(|p| p.len()).unwrap_or(0)).enumerate() {
        //     if i >= vertices.len() / 2 {
        //         most_similar.push(*id);
        //     } else {
        //         least_similar.push(*id);
        //     }
        // }

            (most_similar, least_similar)
    }

}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Similarity {
    vertex_a: NodeIndex,
    vertex_b: NodeIndex,
    score: u8, // The similarity score between vertex_a and vertex_b.
}
