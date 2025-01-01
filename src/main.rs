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
use parking_lot::{ArcMutexGuard, Mutex, RawMutex};
use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, Read, Seek},
    sync::Arc,
};

#[derive(Parser, Debug)]
#[command(name = "fst-merge")]
#[command(about = "Converts a VCD, GHW or FST file to an FST file.", long_about = None)]
struct Args {
    /// Paths for the input .fst files that will be merged
    #[arg(value_name = "INPUTs")]
    inputs: Vec<std::path::PathBuf>,

    /// The path to the output .fst file
    #[arg(short, long)]
    output: std::path::PathBuf,

    /// Comma separated list of offsets, one for each input file. Each offset will be added to the
    /// timestamps of the respective input file. Cannot be negative.
    #[arg(long)]
    offsets: Option<String>,

    /// Only get changes in the given range, in the format "start-end", like "0-100ms".
    #[arg(long)]
    range: Option<String>,

    /// Create new debug signals that compare the values of two signals other signals. The format
    /// is `name:left=right,name2:left2=right2,...`. The `name` is the name of the new signal,
    /// which will be created under the `debug` scope. `left` and `right` are the names of the
    /// signals to compare, in the format `input_idx.scope.signal_name`, like `0.gameroy.cpu.pc`.
    #[arg(long)]
    cmp: Option<String>,
}

fn parse_as_ns(s: &str) -> u64 {
    let index = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (value, unit) = s.split_at(index);
    let value = value.parse::<u64>().expect("invalid range value");
    match unit {
        "fs" => value.div_ceil(1_000_000),
        "ps" => value.div_ceil(1_000),
        "ns" => value,
        "us" => value * 1_000,
        "ms" => value * 1_000_000,
        "s" => value * 1_000_000_000,
        "m" => value * 60_000_000_000,
        "" => value,
        _ => panic!("invalid range unit"),
    }
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

    let range = args.range.as_ref().map(|range| {
        let (start, end) = range.split_once('-').expect("invalid range format");
        let start = parse_as_ns(start);
        let end = parse_as_ns(end);
        (start, end)
    });

    let offsets = args
        .offsets
        .as_ref()
        .map(|offsets| {
            offsets
                .split(',')
                .map(|offset| offset.parse::<u64>().expect("invalid offset"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0; waves.len()]);

    let factors = timescales
        .map(|(factor, exp)| {
            let exp = exp - timescale_exponent;
            factor * 10u32.pow(exp as u32)
        })
        .collect::<Vec<_>>();

    if offsets.len() != waves.len() {
        panic!("Number of offsets does not match number of input files");
    }
    println!("Offsets: {:?}", offsets);
    println!("Factors: {:?}", factors);

    let mut compares = Vec::new();
    if let Some(cmp) = args.cmp.as_ref() {
        for c in cmp.split(',') {
            let (name, expr) = c.split_once(':').expect("invalid compare format");
            let (left, right) = expr.split_once('=').expect("invalid compare format");
            compares.push((name.trim(), left.trim(), right.trim()));
        }
    }

    let mut out = Writer::create(args.output, true)
        .expect("failed to open output")
        .timescale(timescale_exponent as i32);
    let (signal_ref_maps, signal_names) = write_hierarchy(waves.iter_mut(), &mut out);

    out.set_scope(fstapi::scope_type::VCD_MODULE, "debug", "")
        .expect("failed to write top scope");
    let compares = compares
        .into_iter()
        .map(|(name, left, right)| {
            let handle = out
                .create_var(0, 0, 1, name, None)
                .expect("failed to create compare signal");
            (
                handle,
                *signal_names.get(left).expect("invalid signal name"),
                *signal_names.get(right).expect("invalid signal name"),
            )
        })
        .collect::<Vec<_>>();
    out.set_upscope();

    let waves: Vec<_> = waves
        .into_iter()
        .zip(signal_ref_maps)
        .zip(factors)
        .zip(offsets)
        .map(|(((wave, signal_ref_map), factor), offset)| (wave, factor, offset, signal_ref_map))
        .collect();
    write_value_changes(timescale_exponent, range, waves, &compares, &mut out);

    println!("Finishing writing FST file");
    // out.finish().expect("failed to finish writing the FST file");
}

struct SignalChannel {
    queue: [Arc<Mutex<Vec<u8>>>; Self::LEN],
    front_lock: ArcMutexGuard<RawMutex, Vec<u8>>,
    read: usize,
}

fn from_ns(time: u64, exp: i8) -> u64 {
    let exp_ns = exp + 9;
    if exp_ns < 0 {
        time * 10u64.pow(-exp_ns as u32)
    } else {
        time.div_ceil(10u64.pow(exp_ns as u32))
    }
}

impl SignalChannel {
    const LEN: usize = 3;

    fn new<R: BufRead + Seek + Send + 'static>(
        mut reader: FstReader<R>,
        factor: u32,
        offset: u64,
        mut range: Option<(u64, u64)>,
    ) -> Self {
        let filter = if let Some((start, end)) = range.as_mut() {
            let exp = reader.get_header().timescale_exponent;
            *start = from_ns(*start, exp);
            *end = from_ns(*end, exp);
            println!("Filtering from {} to {}", start, end);
            fst_reader::FstFilter::filter_time(*start, *end)
        } else {
            fst_reader::FstFilter::all()
        };

        let queue = std::array::from_fn(|_| Arc::new(Mutex::new(Vec::new())));

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        std::thread::spawn({
            let mut queue: [_; Self::LEN] = std::array::from_fn(|i| queue[i].clone());
            let barrier = barrier.clone();
            move || {
                const KB: usize = 1024;
                const MB: usize = 1024 * KB;
                const BUFFER_SIZE: usize = 8 * MB;

                let mut lock: ArcMutexGuard<RawMutex, Vec<u8>> = queue[0].lock_arc();

                barrier.wait();

                let mut last_time = 0;

                reader
                    .read_signals(&filter, move |time, handle, value| match value {
                        FstSignalValue::String(value) => {
                            // assert!(
                            //     time >= last_time,
                            //     "time: {} < last_time: {}",
                            //     time,
                            //     last_time
                            // );
                            if time < last_time {
                                println!("time: {} < last_time: {}", time, last_time);
                                return;
                            }
                            last_time = time;

                            if let Some((start, end)) = range.as_ref() {
                                if time < *start || time > *end {
                                    return;
                                }
                            }

                            if lock.len() + value.len() > BUFFER_SIZE {
                                // swap
                                queue.rotate_left(1);
                                drop(std::mem::replace(&mut lock, queue[0].lock_arc()));
                                debug_assert!(lock.len() == 0);
                            }

                            push_to_buffer(&mut lock, time * factor as u64 + offset, handle, value);
                        }
                        FstSignalValue::Real(_) => unimplemented!(),
                    })
                    .unwrap();
            }
        });

        // wait for the reader to get the lock
        barrier.wait();

        let front_lock = queue[0].lock_arc();
        SignalChannel {
            queue,
            front_lock,
            read: 0,
        }
    }

    fn next_inner(&mut self, consume: bool) -> Option<(u64, FstSignalHandle, &[u8])> {
        loop {
            let start = &self.front_lock[self.read..];
            let after = &mut &start[..];
            if let Some((time_idx, handle, value)) = read_from_buffer(after) {
                if consume {
                    self.read += start.len() - after.len();
                }
                // workaround for borrow checker limitation
                let value = unsafe { std::mem::transmute::<&[u8], &[u8]>(value) };
                return Some((time_idx, handle, value));
            }

            // swap buffers
            self.front_lock.clear();
            self.queue.rotate_left(1);
            drop(std::mem::replace(
                &mut self.front_lock,
                self.queue[0].lock_arc(),
            ));
            self.read = 0;

            if self.front_lock.is_empty() {
                println!("Finished reading");
                return None;
            }
        }
    }

    fn next(&mut self) -> Option<(u64, FstSignalHandle, &[u8])> {
        self.next_inner(true)
    }

    fn peek(&mut self) -> Option<(u64, FstSignalHandle, &[u8])> {
        self.next_inner(false)
    }
}

fn push_to_buffer(buffer: &mut Vec<u8>, time: u64, handle: FstSignalHandle, value: &[u8]) {
    buffer.extend_from_slice(&time.to_ne_bytes());
    buffer.extend_from_slice(&handle.get_index().to_ne_bytes());
    buffer.extend_from_slice(&(value.len() as u16).to_ne_bytes());
    buffer.extend_from_slice(value);
}

fn read_from_buffer<'a>(buffer: &mut &'a [u8]) -> Option<(u64, FstSignalHandle, &'a [u8])> {
    if buffer.is_empty() {
        return None;
    }
    // if it is not empty, it is a logic error if the buffer is too short

    let mut time = [0; u64::BITS as usize / 8];
    buffer.read_exact(&mut time).unwrap();
    let time = u64::from_ne_bytes(time);

    let mut handle = [0; usize::BITS as usize / 8];
    buffer.read_exact(&mut handle).unwrap();
    let handle = FstSignalHandle::from_index(usize::from_ne_bytes(handle));

    let mut len = [0; u16::BITS as usize / 8];
    buffer.read_exact(&mut len).unwrap();
    let len = u16::from_ne_bytes(len);

    let value = &buffer[..len as usize];
    *buffer = &buffer[len as usize..];

    Some((time, handle, value))
}

/// Writes all value changes from the source file to the FST.
/// Note this is not the most efficient way to do this!
/// A faster version would write each signal directly to the FST instead
/// of writing changes based on the time step.
fn write_value_changes<R: BufRead + Seek + Send + 'static>(
    timescale_exponent: i8,
    range: Option<(u64, u64)>,
    waves: Vec<(FstReader<R>, u32, u64, SignalRefMap)>,
    compare: &[(fstapi::Handle, fstapi::Handle, fstapi::Handle)],
    out: &mut Writer,
) {
    // PERF: use a n-way merge sort
    let mut time_table = waves
        .iter()
        .enumerate()
        .flat_map(|(idx, (wave, factor, offset, _))| {
            wave.get_time_table()
                .unwrap()
                .iter()
                .enumerate()
                .map(move |(time_idx, time)| (*time * *factor as u64 + offset, idx, time_idx))
        })
        .filter(|(time, _, _)| {
            if let Some((start, end)) = range {
                *time >= from_ns(start, timescale_exponent)
                    && *time <= from_ns(end, timescale_exponent)
            } else {
                true
            }
        })
        .collect::<Vec<_>>();

    println!("Sorting time table");
    time_table.sort_unstable();

    let mut signals: Vec<_> = waves
        .into_iter()
        .map(|(wave, factor, offset, signal_ids)| {
            (SignalChannel::new(wave, factor, offset, range), signal_ids)
        })
        .collect();

    println!("Writing value changes");

    let style = indicatif::ProgressStyle::default_bar()
        .template(PROGRESS_BAR_TEMPLATE)
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ");

    let bar = indicatif::ProgressBar::new(time_table.len() as u64).with_style(style);

    bar.set_position(0);

    let need_to_be_compares: HashSet<fstapi::Handle> =
        HashSet::from_iter(compare.iter().flat_map(|(_, a, b)| [*a, *b].into_iter()));

    let mut curr_value = HashMap::new();

    let mut last_time = 0;

    for (i, (time, wave_idx, _)) in time_table.iter().enumerate() {
        if last_time != *time {
            for (var, a, b) in compare {
                let a = curr_value.get(a);
                let b = curr_value.get(b);
                out.emit_value_change(*var, &[(a == b) as u8 + b'0'])
                    .expect("failed to write compare");
            }
            last_time = *time;
        }

        // println!("Time: {} {} {}", time_idx, time, wave_idx);
        out.emit_time_change(*time).expect("failed time change");

        let (signal_iter, signal_ref_map) = &mut signals[*wave_idx];
        while signal_iter
            .peek()
            .map(|(change_time, _, _)| {
                assert!(
                    change_time >= *time,
                    "change_time: {} < time: {}",
                    change_time,
                    time
                );
                change_time == *time
            })
            .unwrap_or_else(|| false)
        {
            let (_, fst_id, value) = signal_iter.next().unwrap();

            let handle = signal_ref_map.get(&fst_id.get_index()).unwrap();

            if !need_to_be_compares.is_empty() && need_to_be_compares.contains(handle) {
                curr_value.insert(*handle, value.to_vec());
            }

            out.emit_value_change(*handle, value)
                .expect("failed to write value change");

            if i % 1_000_000 == 0 {
                bar.set_position(i as u64);
            }
            // if i % 10_000_000 == 0 {
            //     out.flush();
            // }
        }
    }
    bar.finish();
}

type SignalRefMap = std::collections::HashMap<usize, fstapi::Handle>;

fn write_hierarchy<'a, R: BufRead + Seek + 'a>(
    hiers: impl Iterator<Item = &'a mut FstReader<R>>,
    out: &mut Writer,
) -> (Vec<SignalRefMap>, HashMap<String, fstapi::Handle>) {
    println!("Writing hierarchy");
    let mut signal_ref_maps = Vec::with_capacity(hiers.size_hint().0);

    let mut signal_names = HashMap::new();

    for (i, hier) in hiers.enumerate() {
        out.set_scope(fstapi::scope_type::VCD_MODULE, &i.to_string(), "")
            .expect("failed to write top scope");
        let mut signal_ref_map = SignalRefMap::new();

        let mut curr_scope: String = String::new();
        let mut scope_path: String = i.to_string();

        hier.read_hierarchy(|entry| match entry {
            fst_reader::FstHierarchyEntry::Scope {
                tpe,
                name,
                component,
            } => {
                curr_scope = name.clone();
                scope_path += ".";
                scope_path += name.as_str();
                out.set_scope(tpe as u32, &name, &component).unwrap()
            }
            fst_reader::FstHierarchyEntry::UpScope => {
                scope_path.drain(scope_path.len() - curr_scope.len() - 1..);
                out.set_upscope()
            }
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

                let name_path = scope_path.clone() + "." + name.as_str();
                signal_names.insert(name_path, handle2);

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

    (signal_ref_maps, signal_names)
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
