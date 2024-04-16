use std::cmp::Ordering;
use std::collections::VecDeque;
use itertools::Itertools;
use crate::Owner;


use crate::parity_game::{ParityGame, Priority, VertexId};

type Progress = u32;
type ProgressMeasureData<'a> = &'a [Progress];

#[derive(Debug)]
pub struct SmallProgressSolver<'a> {
    game: &'a ParityGame,
    progress_measures: ProgressMeasures,
    max_m_even: Vec<Progress>,
    max_m_odd: Vec<Progress>,
    pub prog_count: usize,
}

impl<'a> SmallProgressSolver<'a> {
    pub fn new(game: &'a ParityGame) -> Self {
        // `d` in the original paper
        let tuple_dimension = game.priority_max() + 1;
        let priority_class_counts = game.priorities_class_count();

        // Calculate the max possible values we will allow during lifting for each player.
        let max_m_even = (0..tuple_dimension)
            // .filter(|i| i % 2 == 0)
            .map(|i| if i % 2 == 0 {
                priority_class_counts.get(&i).copied().unwrap_or_default()
            } else {
                0
            })
            .collect();
        let max_m_odd = (0..tuple_dimension)
            // .filter(|i| i % 2 == 1)
            .map(|i| if i % 2 == 1 {
                priority_class_counts.get(&i).copied().unwrap_or_default()
            } else {
                0
            })
            .collect();

        Self {
            // progress_measures: vec![0; game.vertex_count() * tuple_dimension as usize],
            progress_measures: ProgressMeasures(vec![ProgressMeasure::new(Owner::Odd, tuple_dimension as usize); game.vertex_count()]),
            game,
            max_m_even,
            max_m_odd,
            prog_count: 0,
        }
    }

    /// Run the solver and return a Vec corresponding to each [crate::parity_game::Vertex] indicating who wins.
    #[tracing::instrument(name="Run SPM", skip(self))]
    #[profiling::function]
    pub fn run(&mut self) -> Vec<Owner> {
        let mut queue = VecDeque::from((0..self.game.vertex_count()).map(VertexId::new).collect_vec());
        
        while let Some(vertex_id) = queue.pop_back() {
            let made_change = self.lift(vertex_id, Owner::Odd).expect("Impossible");
            
            if made_change {
                queue.extend(self.game.predecessors(vertex_id));
            }
        }
        
        // Final assignment
        self.progress_measures.0.iter().map(|prog| {
            if prog == &ProgressMeasure::Top {
                Owner::Odd
            } else {
                Owner::Even
            }
        }).collect()
    }

    /// Progress the progress measure of the given vertex if possible
    /// 
    /// # Returns
    /// 
    /// * `true` if a change was made, and thus the fixed point has not yet been reached
    /// * `false` if no change was made, and therefore a fixed point _may_ have been reached.
    #[profiling::function]
    fn lift(&mut self, vertex_id: VertexId, for_player: Owner) -> Option<bool> {
        let vertex = self.game.get(vertex_id)?;
        self.prog_count += 1;
        
        let existing_prog = self.progress_measures.get_measure(vertex_id)?;
        let prog_values = self.game.edges(vertex_id).flat_map(|other_id| {
            let next = self.prog(vertex_id, other_id, for_player);
            tracing::trace!("  - From: {} To: {} ({:?}) with value `{:?}`", vertex_id.index(), other_id.index(), self.game.label(other_id), next);
            next
        });
        
        // Always want to maximise the progress if we own the vertex
        let possible_measure = if vertex.owner == for_player {
            prog_values.max()
        } else {
            prog_values.min()
        }?;
        
        // Max(self, new)
        if &possible_measure > existing_prog {
            tracing::trace!("Vertex: {} ({:?}) from measure: `{:?}` to: `{:?}`", vertex_id.index(), self.game.label(vertex_id), existing_prog, possible_measure);
            self.progress_measures.set_measure(vertex_id, possible_measure);
            Some(true)
        } else {
            tracing::trace!("Vertex: {} ({:?}) keeps measure: `{:?}` instead of: `{:?}`", vertex_id.index(), self.game.label(vertex_id), existing_prog, possible_measure);
            Some(false)
        }
    }
    
    fn prog(&self, from: VertexId, to: VertexId, calculating_for: Owner) -> Option<ProgressMeasure> {
        let v_from = self.game.get(from)?;
        let from_progress = self.progress_measures.get_measure(from)?;
        let to_progress = self.progress_measures.get_measure(to)?;
        
        let next = to_progress.calculate_next(calculating_for, v_from.priority, from_progress, self.max_measure(calculating_for));
        tracing::trace!("      * From: `{:?}` To: `{:?}` Next: `{:?}`", from_progress, to_progress, next);
        
        Some(next)
    }

    #[inline]
    fn max_measure(&self, owner: Owner) -> &[Progress] {
        match owner {
            Owner::Even => &self.max_m_even,
            Owner::Odd => &self.max_m_odd,
        }
    }
}

#[derive(Debug)]
struct ProgressMeasures(Vec<ProgressMeasure>);

impl ProgressMeasures {
    pub fn set_measure(&mut self, vertex_id: VertexId, progress_measure: ProgressMeasure) {
        self.0[vertex_id.index()] = progress_measure
    }

    pub fn replace_measure(&mut self, vertex_id: VertexId, measure: ProgressMeasure) -> ProgressMeasure {
        std::mem::replace(&mut self.0[vertex_id.index()], measure)
    }
    
    pub fn get_measure(&self, vertex_id: VertexId) -> Option<&ProgressMeasure> {
        self.0.get(vertex_id.index())
    }

    pub fn get_measure_mut(&mut self, vertex_id: VertexId) -> Option<&mut ProgressMeasure> {
        self.0.get_mut(vertex_id.index())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum ProgressMeasure {
    Top,
    /// Our progress is stored from least to most significant, e.g., [p0, p1, p2, ...]
    /// This helps some parts of the code, but in turn makes lexicographical comparison slightly less efficient.
    Tuple(Vec<Progress>),
}

impl Ord for ProgressMeasure {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (ProgressMeasure::Top, ProgressMeasure::Top) => Ordering::Equal,
            (ProgressMeasure::Tuple(_), ProgressMeasure::Top) => Ordering::Less,
            (ProgressMeasure::Top, ProgressMeasure::Tuple(_)) => Ordering::Greater,
            (ProgressMeasure::Tuple(left), ProgressMeasure::Tuple(right)) => left.iter().rev().cmp(right.iter().rev()),
        }
    }
}

impl PartialOrd for ProgressMeasure {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ProgressMeasure {
    pub fn new(owner: Owner, max_len: usize) -> Self {
        let items = 0..max_len;
        let data_tuple = match owner {
            // Owner::Even => items.filter(|i| i % 2 == 0).map(|_| 0).collect(),
            // Owner::Odd => items.filter(|i| i % 2 == 1).map(|_| 0).collect()
            Owner::Even => items.map(|_| 0).collect(),
            Owner::Odd => items.map(|_| 0).collect()
        };

        Self::Tuple(data_tuple)
    }

    #[inline]
    pub fn compare_limit(&self, min_incl: usize, other: &ProgressMeasure) -> Ordering {
        match (self, other) {
            (ProgressMeasure::Top, ProgressMeasure::Top) => Ordering::Equal,
            (ProgressMeasure::Tuple(_), ProgressMeasure::Top) => Ordering::Less,
            (ProgressMeasure::Top, ProgressMeasure::Tuple(_)) => Ordering::Greater,
            (ProgressMeasure::Tuple(left), ProgressMeasure::Tuple(right)) => {
                left[min_incl..].iter().rev().cmp(right[min_incl..].iter().rev())
            },
        }
    }

    #[profiling::function]
    #[inline]
    pub fn calculate_next(&self, calculate_for: Owner, incoming_priority: Priority, incoming_progress: &ProgressMeasure, max_prog: &[Progress]) -> ProgressMeasure {
        match self {
            ProgressMeasure::Top => ProgressMeasure::Top,
            ProgressMeasure::Tuple(values) => {
                let mut incoming_values = match incoming_progress {
                    ProgressMeasure::Top => return ProgressMeasure::Top,
                    ProgressMeasure::Tuple(vals) => vals.clone()
                };
                let priority_floor = incoming_priority as usize;

                // Always want to _increase_ the progress measure if priority is matched with the given owner
                if calculate_for.priority_aligned(incoming_priority) {
                    // Calculate the indexes to increase
                    for i in (priority_floor..max_prog.len()).filter(|&v| calculate_for.priority_aligned(v as u32)) {
                        let next_value = values[i] + 1;
                        if next_value > max_prog[i] {
                            // Rollover the increase to a lower priority
                            incoming_values[i] = 0;
                        } else {
                            // Ok, minimal increase was done
                            incoming_values[priority_floor..].copy_from_slice(&values[priority_floor..]);
                            incoming_values[i] = next_value;
                            return ProgressMeasure::Tuple(incoming_values);
                        }
                    }

                    // All values were at max and had to be increased, this means the only next value is Top
                    ProgressMeasure::Top
                } else {
                    incoming_values[priority_floor..].copy_from_slice(&values[priority_floor..]);
                    ProgressMeasure::Tuple(incoming_values)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::time::Instant;

    use pg_parser::{parse_pg};
    use crate::{Owner, ParityGame};
    use crate::solvers::small_progress::{Progress, ProgressMeasure, SmallProgressSolver};
    use crate::tests::example_dir;

    #[test]
    pub fn test_solve_tue_example() {
        let input = std::fs::read_to_string(example_dir().join("tue_example.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let mut solver = SmallProgressSolver::new(&game);

        let solution = solver.run();
        
        println!("Solution: {:#?}", solution);

        println!("Parity Game: {:#?}", solver);

        assert_eq!(solution, vec![Owner::Odd; 7]);
    }

    #[test]
    pub fn test_solve_action_converter() {
        let input = std::fs::read_to_string(example_dir().join("ActionConverter.tlsf.ehoa.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let mut solver = SmallProgressSolver::new(&game);
        
        let now = Instant::now();
        let solution = solver.run();

        println!("Solution: {:#?}", solution);
        println!("Prog Count: {}", solver.prog_count);
        println!("Took: {:?}", now.elapsed());
        assert_eq!(solution, vec![
            Owner::Even,
            Owner::Odd,
            Owner::Even,
            Owner::Even,
            Owner::Even,
            Owner::Even,
            Owner::Odd,
            Owner::Odd,
            Owner::Even,
        ])
    }
    
    #[test]
    pub fn perf_test() {
        let input = std::fs::read_to_string(example_dir().join("amba_decomposed_arbiter_6.tlsf.ehoa.pg")).unwrap();
        let pg = parse_pg(&mut input.as_str()).unwrap();
        let game = ParityGame::new(pg).unwrap();
        let mut solver = SmallProgressSolver::new(&game);

        let now = Instant::now();
        let solution = solver.run();

        println!("Prog Count: {}", solver.prog_count);
        println!("Took: {:?}", now.elapsed());
        println!("Solution: {:#?}", solution);
    }

    #[test]
    pub fn test_progress_compare() {
        let top = ProgressMeasure::Top;
        // Need a priority index of at least this
        let min_priority = 0;
        let w_other = ProgressMeasure::Tuple(vec![0, 2, 0, 0]);

        assert_eq!(top.compare_limit(min_priority, &ProgressMeasure::Top), Ordering::Equal);
        assert_eq!(top.compare_limit(min_priority, &w_other), Ordering::Greater);
        assert_eq!(w_other.compare_limit(min_priority, &top), Ordering::Less);
        // Complete comparisons
        assert_eq!(
            w_other.compare_limit(min_priority, &ProgressMeasure::Tuple(vec![0, 1, 0, 0])),
            Ordering::Greater
        );
        assert_eq!(
            w_other.compare_limit(min_priority, &ProgressMeasure::Tuple(vec![0, 0, 0, 1])),
            Ordering::Less
        );
        assert_eq!(
            w_other.compare_limit(min_priority, &ProgressMeasure::Tuple(vec![0, 2, 0, 1])),
            Ordering::Less
        );
        // Partial comparisons
        assert_eq!(
            ProgressMeasure::Tuple(vec![0, 1, 0, 1]).compare_limit(3, &ProgressMeasure::Tuple(vec![0, 2, 0, 1])),
            Ordering::Equal
        );
        assert_eq!(
            ProgressMeasure::Tuple(vec![1, 2, 0, 1]).compare_limit(2, &ProgressMeasure::Tuple(vec![2, 2, 2, 1])),
            Ordering::Less
        );
        assert_eq!(
            ProgressMeasure::Tuple(vec![0, 0, 0, 1]).compare_limit(1, &ProgressMeasure::Tuple(vec![0, 2, 0, 0])),
            Ordering::Greater
        );
    }

    macro_rules! assert_eq_s {
        ($size:expr, $left:expr, $right:expr $(,)?) => {
            match (&$left, &$right) {
                (left_val, right_val) => {
                    if !(left_val.compare_limit($size, &right_val) == std::cmp::Ordering::Equal) {
                        assert!(
                            false,
                            "Compare not as expected, got left: {:#?} vs right {:#?}",
                            left_val, right_val
                        )
                    }
                }
            }
        };
    }

    #[test]
    pub fn test_progress_next() {
        fn t(values: Vec<Progress>) -> ProgressMeasure {
            ProgressMeasure::Tuple(values)
        }

        let origin = t(vec![0, 0, 0, 0]);

        let priority = 3;
        let max_prog = vec![0, 2, 0, 1];

        let successor = t(vec![0, 2, 0, 0]);

        assert_eq_s!(priority, successor.calculate_next(Owner::Odd, 0, &origin, &max_prog), t(vec![0, 2, 0, 0]));
        assert_eq_s!(priority, successor.calculate_next(Owner::Odd, 1, &origin, &max_prog), t(vec![0, 0, 0, 1]));
        // First three elements (p0, p1, p2) are set to `don't care`, thus don't copy them!
        assert_eq_s!(priority, successor.calculate_next(Owner::Odd, 3, &origin, &max_prog), t(vec![0, 0, 0, 1]));
    }
}
