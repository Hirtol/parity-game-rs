use winnow::{
    ascii::{dec_uint, space0},
    combinator::{alt, opt, preceded, separated, terminated},
    error::ParserError,
    Parser,
};

pub fn parse_pg<'a: 'b, 'b>(input: &mut &'a str, builder: &mut impl PgBuilder<'b>) -> winnow::PResult<()> {
    let header_cnt = opt(parse_header).parse_next(input)?;
    if let Some(header) = header_cnt {
        let _ = winnow::ascii::line_ending.parse_next(input)?;
        builder.set_header(header).expect("Builder failure");
    }

    separated(0.., |inp: &mut &'a str | {
        let out = parse_node.parse_next(inp)?;
        builder.add_vertex(out.id, out).expect("Builder failure");
        Ok(())
    }, winnow::ascii::line_ending).parse_next(input)
}

fn parse_header(input: &mut &str) -> winnow::PResult<usize> {
    terminated(preceded(ws("parity"), dec_uint), eol).parse_next(input)
}

fn parse_node<'a>(input: &mut &'a str) -> winnow::PResult<Vertex<'a>> {
    let id = vertex_id.parse_next(input)?;
    let priority: usize = ws(dec_uint).parse_next(input)?;

    let owner = ws(alt(('0'.value(0), '1'.value(1)))).parse_next(input)?;

    let edges: Vec<usize> = ws(separated(0.., vertex_id, ws(','))).parse_next(input)?;

    let label = terminated(opt(ws(vertex_label)), eol).parse_next(input)?;

    Ok(Vertex {
        id,
        priority,
        owner,
        outgoing_edges: edges,
        label,
    })
}

fn vertex_id(input: &mut &str) -> winnow::PResult<usize> {
    Ok(dec_uint.parse_next(input)?)
}

fn vertex_label<'a>(input: &mut &'a str) -> winnow::PResult<&'a str> {
    winnow::combinator::delimited('"', winnow::token::take_until(0.., '"'), '"').parse_next(input)
}

fn eol(input: &mut &str) -> winnow::PResult<()> {
    let _ = ';'.parse_next(input)?;
    Ok(())
}

fn ws<'i, O, E>(parser: impl Parser<&'i str, O, E>) -> impl Parser<&'i str, O, E>
where
    E: ParserError<&'i str>,
{
    winnow::combinator::delimited(space0, parser, space0)
}

pub trait PgBuilder<'a> {
    fn set_header(&mut self, vertex_count: usize) -> eyre::Result<()>;
    
    fn add_vertex(&mut self, id: usize, vertex: Vertex<'a>) -> eyre::Result<()>;
}

#[derive(Debug)]
pub struct Vertex<'a> {
    pub id: usize,
    pub priority: usize,
    pub owner: u8,
    pub outgoing_edges: Vec<usize>,
    pub label: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    use crate::{parse_pg, PgBuilder, Vertex};
    use std::{path::PathBuf, time::Instant};

    #[derive(Debug)]
    struct FauxBuilder<'a>(Vec<Vertex<'a>>);
    
    impl<'a> PgBuilder<'a> for FauxBuilder<'a> {
        fn set_header(&mut self, vertex_count: usize) -> eyre::Result<()> {
            Ok(())
        }

        fn add_vertex(&mut self, id: usize, vertex: Vertex<'a>) -> eyre::Result<()> {
            self.0.push(vertex);
            Ok(())
        }
    }
    
    #[test]
    pub fn test_simple() {
        let mut input = "parity 4;
0 2 1 1;
1 2 1 1,3;
2 2 1 0, 1;
3 2 0 2,1,0;";
        let mut builder = FauxBuilder(Vec::new());
        parse_pg(&mut input, &mut builder).unwrap();

        assert_eq!(builder.0.len(), 4);
        assert_eq!(builder.0[3].outgoing_edges.len(), 3);

        println!("Parity Game: {:#?}", builder);
    }

    #[test]
    pub fn test_with_label() {
        let input = std::fs::read_to_string(example_dir().join("ActionConverter.tlsf.ehoa.pg")).unwrap();
        let mut builder = FauxBuilder(Vec::new());
        parse_pg(&mut input.as_str(), &mut builder).unwrap();

        assert_eq!(builder.0.len(), 9);
        assert_eq!(builder.0[8].label, Some("178"));

        println!("Parity Game: {:#?}", builder);
    }

    #[test]
    pub fn test_no_fail_all_examples() {
        std::fs::read_dir(example_dir()).unwrap().flatten().for_each(|f| {
            let input = std::fs::read_to_string(f.path()).unwrap();

            let now = Instant::now();
            let mut builder = FauxBuilder(Vec::new());
            parse_pg(&mut input.as_str(), &mut builder).unwrap();

            println!(
                "Took: `{:?}` to parse parity game of size: `{}` ({:?})",
                now.elapsed(),
                builder.0.len(),
                f.path().file_name().unwrap()
            );
        });
    }

    fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("game_examples")
    }
}
