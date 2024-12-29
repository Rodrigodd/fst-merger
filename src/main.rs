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
    collections::HashMap,
    io::{BufRead, Read, Seek},
    sync::Arc,
};

#[derive(Parser, Debug)]
#[command(name = "fst-merge")]
#[command(about = "Converts a VCD, GHW or FST file to an FST file.", long_about = None)]
struct Args {
    #[arg(value_name = "INPUTs")]
    inputs: Vec<std::path::PathBuf>,
    #[arg(short, long)]
    output: std::path::PathBuf,

    #[arg(short, long)]
    cmp: Option<String>,
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

struct SignalChannel {
    queue: [Arc<Mutex<Vec<u8>>>; Self::LEN],
    front_lock: ArcMutexGuard<RawMutex, Vec<u8>>,
    read: usize,
}

impl SignalChannel {
    const LEN: usize = 3;

    fn new<R: BufRead + Seek + Send + 'static>(mut reader: FstReader<R>) -> Self {
        let mut ids = Vec::new();

        reader
            .read_hierarchy(|entry| {
                if let fst_reader::FstHierarchyEntry::Var { handle, .. } = entry {
                    ids.push(handle)
                }
            })
            .unwrap();

        let filter = fst_reader::FstFilter::new(0, u64::MAX, ids);

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

                            if lock.len() + value.len() > BUFFER_SIZE {
                                // swap
                                queue.rotate_left(1);
                                drop(std::mem::replace(&mut lock, queue[0].lock_arc()));
                                debug_assert!(lock.len() == 0);
                            }

                            push_to_buffer(&mut lock, time, handle, value);
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

fn push_to_buffer(buffer: &mut Vec<u8>, time_idx: u64, handle: FstSignalHandle, value: &[u8]) {
    buffer.extend_from_slice(&time_idx.to_ne_bytes());
    buffer.extend_from_slice(&handle.get_index().to_ne_bytes());
    buffer.extend_from_slice(&(value.len() as u16).to_ne_bytes());
    buffer.extend_from_slice(value);
}

fn read_from_buffer<'a>(buffer: &mut &'a [u8]) -> Option<(u64, FstSignalHandle, &'a [u8])> {
    if buffer.is_empty() {
        return None;
    }
    // if it is not empty, it is a logic error if the buffer is too short

    let mut time_idx = [0; u64::BITS as usize / 8];
    buffer.read_exact(&mut time_idx).unwrap();
    let time_idx = u64::from_ne_bytes(time_idx);

    let mut handle = [0; usize::BITS as usize / 8];
    buffer.read_exact(&mut handle).unwrap();
    let handle = FstSignalHandle::from_index(usize::from_ne_bytes(handle));

    let mut len = [0; u16::BITS as usize / 8];
    buffer.read_exact(&mut len).unwrap();
    let len = u16::from_ne_bytes(len);

    let value = &buffer[..len as usize];
    *buffer = &buffer[len as usize..];

    Some((time_idx, handle, value))
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
    println!("Sorting time table");
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
        .map(|(wave, _, signal_ids)| (SignalChannel::new(wave), signal_ids))
        .collect();

    println!("Writing value changes");

    let style = indicatif::ProgressStyle::default_bar()
        .template(PROGRESS_BAR_TEMPLATE)
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ");

    let bar = indicatif::ProgressBar::new(time_table.len() as u64).with_style(style);

    bar.set_position(0);

    for (i, (time, wave_idx, time_idx)) in time_table.iter().enumerate() {
        // println!("Time: {} {} {}", time_idx, time, wave_idx);
        out.emit_time_change(*time).expect("failed time change");

        let (signal_iter, signal_ref_map) = &mut signals[*wave_idx];
        while signal_iter
            .peek()
            .map(|(change_time, _, _)| {
                assert!(
                    change_time >= *time,
                    "change_idx: {} < time: {}",
                    change_time,
                    time
                );
                change_time == *time
            })
            .unwrap_or_else(|| false)
        {
            let (_, fst_id, value) = signal_iter.next().unwrap();

            out.emit_value_change(*signal_ref_map.get(&fst_id.get_index()).unwrap(), value)
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
