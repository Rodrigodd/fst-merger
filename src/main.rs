// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// Small utility that reads in a VCD, GHW or FST file with wellen and then
// writes out the FST with the fst-writer library.
// Similar to vcd2fst, just that the input format does not have to be specified
// by the command name.

use clap::Parser;
use fst_reader::{FstReader, FstSignalHandle, FstSignalValue};
use fstapi::Writer;
use std::io::{BufRead, Seek};
use std::sync::mpsc::{channel, Receiver};

#[derive(Parser, Debug)]
#[command(name = "fst-merge")]
#[command(about = "Converts a VCD, GHW or FST file to an FST file.", long_about = None)]
struct Args {
    #[arg(value_name = "INPUTs")]
    inputs: Vec<std::path::PathBuf>,
    #[arg(short, long)]
    output: std::path::PathBuf,
}

const PROGRESS_BAR_TEMPLATE: &str = "\
{elapsed_precise} █{bar:60.cyan/blue}█ {pos:>8}/{len:>8} {per_sec} ({eta})";

fn main() {
    println!("{:?}", std::env::args().collect::<Vec<_>>());
    let args = Args::parse();

    // let mut wave = simple::read(args.input).expect("failed to read input");
    let mut waves: Vec<FstReader<_>> = args
        .inputs
        .iter()
        .map(|path| {
            let file = std::fs::File::open(path).expect("failed to open file");
            FstReader::open_and_read_time_table(std::io::BufReader::new(file))
                .expect("failed to open FST")
        })
        .collect();

    let timescales = waves.iter().map(|wave| {
        let factor = 1;
        let timescale_exponent = wave.get_header().timescale_exponent;

        // while factor % 10 == 0 {
        //     factor /= 10;
        //     timescale_exponent += 1;
        // }

        (factor, timescale_exponent)
    });

    let timescale_exponent = timescales.clone().map(|(_, exp)| exp).min().unwrap();

    let factors = timescales
        .map(|(factor, exp)| {
            let exp = exp - timescale_exponent;
            factor * 10u32.pow(exp as u32)
        })
        .collect::<Vec<_>>();

    // let start_time = waves.iter().map(|wave| wave.time_table()[0]).min().unwrap();
    // let version = waves
    //     .iter()
    //     .map(|wave| wave.hierarchy().version())
    //     .next()
    //     .unwrap()
    //     .to_string();
    // let date = waves
    //     .iter()
    //     .map(|wave| wave.hierarchy().date())
    //     .next()
    //     .unwrap()
    //     .to_string();

    // let info = FstInfo {
    //     start_time,
    //     timescale_exponent,
    //     version,
    //     date,
    //     file_type: FstFileType::Verilog, // TODO
    // };
    //
    // let mut out = open_fst(args.output, &info).expect("failed to open output");
    // let mut out = out
    //     .finish()
    //     .expect("failed to write FST header or hierarchy");
    let mut out = Writer::create(args.output, true)
        .expect("failed to open output")
        .timescale(timescale_exponent as i32);
    let signal_ref_maps = write_hierarchy(waves.iter_mut(), &mut out);

    let waves: Vec<_> = waves
        .into_iter()
        .zip(signal_ref_maps)
        .zip(factors)
        .map(|((wave, signal_ref_map), factor)| (wave, factor, signal_ref_map))
        .collect();
    write_value_changes(waves, &mut out);

    println!("Finishing writing FST file");
    // out.finish().expect("failed to finish writing the FST file");
}

fn reader_to_channel<R: BufRead + Seek + Send + 'static>(
    mut reader: FstReader<R>,
) -> Receiver<(u64, FstSignalHandle, Vec<u8>)> {
    let mut ids = Vec::new();

    reader
        .read_hierarchy(|entry| {
            if let fst_reader::FstHierarchyEntry::Var { handle, .. } = entry {
                ids.push(handle)
            }
        })
        .unwrap();

    let filter = fst_reader::FstFilter::new(0, u64::MAX, ids);

    let (sender, receiver) = channel();

    std::thread::spawn(move || {
        reader.read_signals(&filter, |time_idx, handle, value| match value {
            FstSignalValue::String(value) => {
                sender.send((time_idx, handle, value.to_vec())).unwrap();
            }
            FstSignalValue::Real(_) => unimplemented!(),
        })
    });

    receiver
}

/// Writes all value changes from the source file to the FST.
/// Note this is not the most efficient way to do this!
/// A faster version would write each signal directly to the FST instead
/// of writing changes based on the time step.
fn write_value_changes<R: BufRead + Seek + Send + 'static>(
    waves: Vec<(FstReader<R>, u32, SignalRefMap)>,
    out: &mut Writer,
) {
    // PERF: use a n-way merge sort
    // println!("Sorting time table");
    let mut time_table = waves
        .iter()
        .enumerate()
        .flat_map(|(idx, (wave, factor, _))| {
            wave.get_time_table()
                .unwrap()
                .iter()
                .enumerate()
                .map(move |(time_idx, time)| (*time * *factor as u64, idx, time_idx))
        })
        .collect::<Vec<_>>();
    time_table.sort_unstable();

    let mut signals: Vec<_> = waves
        .into_iter()
        .map(|(wave, _, signal_ids)| (reader_to_channel(wave).into_iter().peekable(), signal_ids))
        .collect();

    println!("Writing value changes");

    let style = indicatif::ProgressStyle::default_bar()
        .template(PROGRESS_BAR_TEMPLATE)
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ");

    let bar = indicatif::ProgressBar::new(time_table.len() as u64).with_style(style);

    bar.set_position(0);

    for (i, (time, wave_idx, time_idx)) in time_table.iter().enumerate() {
        out.emit_time_change(*time).expect("failed time change");

        let time_idx = *time_idx as u64;
        let (signal_iter, signal_ref_map) = &mut signals[*wave_idx];
        while signal_iter
            .peek()
            .map(|(change_idx, _, _)| *change_idx == time_idx)
            .unwrap_or(false)
        {
            let (_, fst_id, value) = signal_iter.next().unwrap();

            out.emit_value_change(*signal_ref_map.get(&fst_id.get_index()).unwrap(), &value)
                .expect("failed to write value change");

            if i % 1_000_000 == 0 {
                bar.set_position(i as u64);
            }
            if i % 10_000_000 == 0 {
                out.flush();
            }
        }
    }
    bar.finish();
}

type SignalRefMap = std::collections::HashMap<usize, fstapi::Handle>;

fn write_hierarchy<'a, R: BufRead + Seek + 'a>(
    hiers: impl Iterator<Item = &'a mut FstReader<R>>,
    out: &mut Writer,
) -> Vec<SignalRefMap> {
    println!("Writing hierarchy");
    let mut signal_ref_maps = Vec::with_capacity(hiers.size_hint().0);
    for (i, hier) in hiers.enumerate() {
        out.set_scope(fstapi::scope_type::VCD_MODULE, &i.to_string(), "")
            .expect("failed to write top scope");
        let mut signal_ref_map = SignalRefMap::new();

        hier.read_hierarchy(|entry| match entry {
            fst_reader::FstHierarchyEntry::Scope {
                tpe,
                name,
                component,
            } => out.set_scope(tpe as u32, &name, &component).unwrap(),
            fst_reader::FstHierarchyEntry::UpScope => out.set_upscope(),
            fst_reader::FstHierarchyEntry::Var {
                tpe,
                direction,
                name,
                length,
                handle,
                is_alias: _,
            } => {
                let alias = signal_ref_map.get(&handle.get_index()).cloned();
                let handle2 = out
                    .create_var(tpe as u32, direction as u32, length, &name, alias)
                    .unwrap();
                if alias.is_none() {
                    signal_ref_map.insert(handle.get_index(), handle2);
                }
            }
            x => unimplemented!("{:?}", x),
        })
        .unwrap();

        signal_ref_maps.push(signal_ref_map);
        out.set_upscope();
    }
    signal_ref_maps
}

/// Lowest common multiple
pub fn lcm(a: u64, b: u64) -> u64 {
    a * b / gcd(a, b)
}

/// Greatest common divisor
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
