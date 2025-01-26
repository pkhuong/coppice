//! Replicates the query from https://www.firebolt.io/free-sample-datasets/nyc-philharmonic
//! on the [New York Philharmonic Performance History](https://github.com/nyphilarchive/PerformanceHistory).
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;

use coppice::aggregates::Counter;
use coppice::aggregates::Histogram;
use coppice::map_map_reduce;
use coppice::map_reduce;

#[allow(dead_code)]
#[derive(Deserialize)]
struct Concert {
    #[serde(rename = "eventType")]
    event_type: String,
    #[serde(rename = "Location")]
    location: String,
    #[serde(rename = "Venue")]
    venue: String,
    #[serde(rename = "Date")]
    date: String,
    #[serde(rename = "Time")]
    time: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Soloist {
    #[serde(rename = "soloistName")]
    name: String,
    #[serde(rename = "soloistInstrument")]
    instrument: String,
    #[serde(rename = "soloistRoles")]
    roles: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Work {
    #[serde(rename = "ID")]
    id: String,
    #[serde(rename = "composerName")]
    composer_name: Option<String>,

    // workTitle isusually a string, sometimes an object...
    // #[serde(rename = "workTitle")]

    // title: Option<String>,
    // Movement is usually a string, sometimes an object like { "_": "Overture To", "em": "Tannhäuser" }...
    // movement: Option<String>,
    soloists: Vec<Option<Soloist>>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Program {
    id: String, // GUID
    #[serde(rename = "programID")] // NYP id
    program_id: String,
    orchestra: String,
    season: String,
    concerts: Vec<Concert>,
    works: Vec<Work>,
}

fn load_json_dump(path: impl AsRef<Path>) -> Result<Vec<Program>, &'static str> {
    #[derive(Deserialize)]
    struct Programs {
        programs: Vec<Program>,
    }

    fn doit(path: &Path) -> std::io::Result<Vec<Program>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let ret: Programs = serde_json::from_reader(reader)?;
        Ok(ret.programs)
    }

    match doit(path.as_ref()) {
        Ok(ret) => Ok(ret),
        Err(e) => {
            eprintln!("failed to load {:?}: {:?}", path.as_ref(), e);
            Err("failed to load json dump")
        }
    }
}

fn list_json_files(base_dir: impl AsRef<Path>) -> std::io::Result<Vec<PathBuf>> {
    let mut data_files: Vec<PathBuf> = Vec::new();

    for dirent in std::fs::read_dir(base_dir.as_ref())? {
        let dirent = dirent?;
        let name = dirent.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".json") && name != "complete.json" {
            data_files.push(dirent.path());
        }
    }

    data_files.sort();
    Ok(data_files)
}

fn count_programs(files: &[PathBuf]) -> Result<u64, &'static str> {
    let ret = map_reduce(
        files,
        &(),
        &(),
        &|path| load_json_dump(path),
        &|_token, _params, _keys, _row| Counter::new(1),
    )?
    .count;
    Ok(ret)
}

fn count_composer_occurrences(files: &[PathBuf]) -> Result<Vec<(String, u64)>, &'static str> {
    let occurrences = map_reduce(
        files,
        &(),
        &(),
        &|path| load_json_dump(path),
        &|_token, _params, _keys, row| {
            let mut ret: Histogram<String> = Default::default();

            for work in row.works.iter() {
                if let Some(composer) = &work.composer_name {
                    ret.observe(composer.to_owned(), Counter::new(1));
                }
            }

            ret
        },
    )?;

    Ok(occurrences.into_popularity_sorted_vec())
}

fn count_venue_occurrences(files: &[PathBuf]) -> Result<Vec<(String, u64)>, &'static str> {
    let occurrences = map_reduce(
        files,
        &(),
        &(),
        &|path| load_json_dump(path),
        &|_token, _params, _keys, row| {
            let mut ret: Histogram<String> = Default::default();

            for concert in row.concerts.iter() {
                ret.observe(concert.venue.to_owned(), Counter::new(1));
            }

            ret
        },
    )?;

    Ok(occurrences.into_popularity_sorted_vec())
}

fn count_composer_cooccurrences(
    files: &[PathBuf],
    venue: String,
    root_composer: Option<String>,
) -> Result<Vec<(String, u64)>, &'static str> {
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;

    let cooccurrences = map_map_reduce(
        files,
        &venue,
        &root_composer,
        &|path| load_json_dump(path),
        &|venue, rows| {
            let venue = venue.clone();
            Ok(rows
                .into_par_iter()
                .filter(move |row| row.concerts.iter().any(|concert| concert.venue == venue))
                .map(|row| {
                    row.works
                        .iter()
                        .map(|work| work.composer_name.clone())
                        .collect::<Vec<Option<String>>>()
                }))
        },
        &|token, _venue, root_composer, composers| {
            let mut ret: Histogram<String> = Default::default();

            let mut maybe_composers: Vec<&Option<String>> = vec![&None];
            maybe_composers.extend(composers.iter());

            let (mut token, root_composer) = token.focus(root_composer);

            if token.eql_any(root_composer, &maybe_composers) {
                for composer in composers.iter().flatten().cloned() {
                    ret.observe(composer, Counter::new(1));
                }
            }

            ret
        },
    )?;

    Ok(cooccurrences.into_popularity_sorted_vec())
}

fn main() -> std::io::Result<()> {
    let other = std::io::Error::other;

    let args = std::env::args().collect::<Vec<String>>();
    let data_dir = args.get(1).ok_or(other(
        "First argument must have path to PerformanceHistory/Programs/json/",
    ))?;
    let data_files = list_json_files(data_dir)?;
    println!("Found {} data files: {:?}", data_files.len(), &data_files);

    {
        let start = std::time::Instant::now();
        let count = count_programs(&data_files).map_err(other)?;
        println!("num_programs (duration={:?}): {}", start.elapsed(), count);
    }

    {
        let start = std::time::Instant::now();
        let composer_occurrences = count_composer_occurrences(&data_files).map_err(other)?;
        println!(
            "top 5 composers cold (duration={:?}):\t{:?}",
            start.elapsed(),
            &composer_occurrences[0..5.min(composer_occurrences.len())]
        );
    }

    {
        let start = std::time::Instant::now();
        let composer_occurrences = count_composer_occurrences(&data_files).map_err(other)?;
        println!(
            "top 10 composers cached (duration={:?}):\t{:?}",
            start.elapsed(),
            composer_occurrences.chunks(10).next().unwrap_or(&[])
        );
    }

    {
        let start = std::time::Instant::now();
        let venue_occurrences = count_venue_occurrences(&data_files).map_err(other)?;
        println!(
            "top 5 venues (duration={:?}):\t{:?}",
            start.elapsed(),
            venue_occurrences.chunks(5).next().unwrap_or(&[])
        );
    }

    println!();

    let venue = "Carnegie Hall".to_string();

    let top_composers: Vec<String>;
    {
        let start = std::time::Instant::now();
        let composer_occurrences =
            count_composer_cooccurrences(&data_files, venue.clone(), None).map_err(other)?;
        println!(
            "top 5 composers at {} (duration={:?}):\n\t{:?}",
            &venue,
            start.elapsed(),
            composer_occurrences.chunks(5).next().unwrap_or(&[])
        );

        top_composers = composer_occurrences
            .into_iter()
            .map(|x| x.0)
            .take(10)
            .collect::<Vec<_>>();
    }

    for root_composer in top_composers.into_iter() {
        let start = std::time::Instant::now();
        let composer_cooccurrences =
            count_composer_cooccurrences(&data_files, venue.clone(), Some(root_composer.clone()))
                .map_err(other)?;
        println!(
            "top 5 composers with '{}' at {} (duration={:?}):\n\t{:?}",
            &root_composer,
            &venue,
            start.elapsed(),
            composer_cooccurrences.chunks(5).next().unwrap_or(&[])
        );
    }

    Ok(())
}
