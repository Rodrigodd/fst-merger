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
use hashbrown::HashMap;
use parking_lot::{ArcMutexGuard, Mutex, RawMutex};
use std::{
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
    let start = std::time::Instant::now();

    let args = Args::parse();

    let offsets = args
        .offsets
        .as_ref()
        .map(|offsets| offsets.split(',').map(parse_as_ns).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![0; args.inputs.len()]);

    if offsets.len() != args.inputs.len() {
        panic!("Number of offsets does not match number of input files");
    }

    let mut compares = Vec::new();
    if let Some(cmp) = args.cmp.as_ref() {
        for c in cmp.split(',') {
            let (name, expr) = c.split_once(':').expect("invalid compare format");
            let (left, right) = expr.split_once('=').expect("invalid compare format");
            compares.push((name.trim(), left.trim(), right.trim()));
        }
    }

    let range = args.range.as_ref().map(|range| {
        let (start, end) = range.split_once('-').expect("invalid range format");
        let start = parse_as_ns(start);
        let end = parse_as_ns(end);
        (start, end)
    });
    let mut input_range = (u64::MAX, 0);

    println!("{:>10.2?} Reading input files", start.elapsed());

    // let mut wave = simple::read(args.input).expect("failed to read input");
    let mut waves: Vec<FstReader<_>> = args
        .inputs
        .iter()
        .map(|path| {
            let file = std::fs::File::open(path).expect("failed to open file");
            let reader =
                FstReader::open(std::io::BufReader::new(file)).expect("failed to open FST");

            let header = reader.get_header();
            let exp = header.timescale_exponent;

            let start_ns = to_ns(header.start_time, exp);
            let end_ns = to_ns(header.end_time, exp);

            if start_ns < input_range.0 {
                input_range.0 = start_ns;
            }
            if end_ns > input_range.1 {
                input_range.1 = end_ns;
            }

            println!(
                "{:>10.2?} Input file {}: interval: {:.2?} to {:.2?}",
                start.elapsed(),
                path.display(),
                std::time::Duration::from_nanos(start_ns),
                std::time::Duration::from_nanos(end_ns)
            );

            reader
        })
        .collect();

    if let Some(range) = range.as_ref() {
        input_range = *range;
    }

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

    println!("{:>10.2?} Writing hierarchy", start.elapsed());
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
    write_value_changes(input_range, start, range, waves, &compares, &mut out);

    println!("{:>10.2?} Finished writing", start.elapsed());
    // out.finish().expect("failed to finish writing the FST file");
}

struct SignalChannel {
    queue: [Arc<Mutex<Vec<u8>>>; Self::LEN],
    front_lock: ArcMutexGuard<RawMutex, Vec<u8>>,
    read: usize,
}

fn to_ns(time: u64, exp: i8) -> u64 {
    let exp_ns = exp + 9;
    if exp_ns < 0 {
        time * 10u64.pow(-exp_ns as u32)
    } else {
        time * 10u64.pow(exp_ns as u32)
    }
}

fn from_ns(ns: u64, exp: i8) -> u64 {
    let exp_ns = exp + 9;
    if exp_ns < 0 {
        ns * 10u64.pow(-exp_ns as u32)
    } else {
        ns.div_ceil(10u64.pow(exp_ns as u32))
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

    fn cmp(&mut self, other: &mut Self) -> std::cmp::Ordering {
        self.peek()
            .map(|(a, _, _)| {
                other
                    .peek()
                    .map(|(b, _, _)| a.partial_cmp(&b).unwrap())
                    .unwrap_or(std::cmp::Ordering::Greater)
            })
            .unwrap_or_else(|| {
                other
                    .peek()
                    .map(|_| std::cmp::Ordering::Less)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}
// The ord implement compares to value of `peek`. This is used to allow merge-sorting multiple
// SignalChannels.
// impl std::cmp::PartialOrd for SignalChannel {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.peek().map(|(a, _, _)| {
//             other
//                 .peek()
//                 .map(|(b, _, _)| a.partial_cmp(&b).unwrap())
//                 .unwrap_or(std::cmp::Ordering::Less)
//         })
//     }
// }

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

fn slice_get_two<T>(slice: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a < b);
    assert!(b < slice.len());
    let (left, right) = slice.split_at_mut(b);
    (&mut left[a], &mut right[0])
}

struct BinaryHeap<T, F> {
    data: Vec<T>,
    comparator: F,
}
impl<T, F: Fn(&mut T, &mut T) -> std::cmp::Ordering> BinaryHeap<T, F> {
    fn new(comparator: F) -> Self {
        BinaryHeap {
            data: Vec::new(),
            comparator,
        }
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn insert(&mut self, value: T) {
        self.data.push(value);
        self.sift_up(self.data.len() - 1);
    }

    fn peek(&mut self) -> Option<&mut T> {
        self.data.first_mut()
    }

    fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.pop().unwrap();
        if !self.data.is_empty() {
            let first = std::mem::replace(&mut self.data[0], last);
            self.sift_down(0);
            Some(first)
        } else {
            Some(last)
        }
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent_idx = (idx - 1) / 2;
            let (parent, child) = slice_get_two(&mut self.data, parent_idx, idx);
            if (self.comparator)(parent, child).is_gt() {
                self.data.swap(parent_idx, idx);
                idx = parent_idx;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        while idx < self.data.len() {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut largest = idx;
            if left < self.data.len() {
                let (larg, lf) = slice_get_two(&mut self.data, largest, left);
                if (self.comparator)(lf, larg).is_lt() {
                    largest = left;
                }
            }
            if right < self.data.len() {
                let (larg, rt) = slice_get_two(&mut self.data, largest, right);
                if (self.comparator)(rt, larg).is_lt() {
                    largest = right;
                }
            }
            if largest != idx {
                self.data.swap(largest, idx);
                idx = largest;
            } else {
                break;
            }
        }
    }
}

/// Writes all value changes from the source file to the FST.
/// Note this is not the most efficient way to do this!
/// A faster version would write each signal directly to the FST instead
/// of writing changes based on the time step.
fn write_value_changes<R: BufRead + Seek + Send + 'static>(
    input_range: (u64, u64),
    start: std::time::Instant,
    range: Option<(u64, u64)>,
    waves: Vec<(FstReader<R>, u32, u64, SignalRefMap)>,
    compare: &[(fstapi::Handle, fstapi::Handle, fstapi::Handle)],
    out: &mut Writer,
) {
    let mut signals =
        BinaryHeap::new(|(x, _): &mut (SignalChannel, SignalRefMap), (y, _)| x.cmp(y));

    for wave in waves.into_iter().map(|(wave, factor, offset, signal_ids)| {
        (SignalChannel::new(wave, factor, offset, range), signal_ids)
    }) {
        signals.insert(wave);
    }

    println!("{:>10.2?} Writing value changes", start.elapsed());

    let style = indicatif::ProgressStyle::default_bar()
        .template(PROGRESS_BAR_TEMPLATE)
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ");

    let bar = indicatif::ProgressBar::new(input_range.1 - input_range.0).with_style(style);

    bar.set_position(0);

    let mut curr_value: HashMap<fstapi::Handle, Vec<u8>> = HashMap::from_iter(
        compare
            .iter()
            .flat_map(|(_, a, b)| [*a, *b].into_iter())
            .map(|handle| (handle, vec![b'x'])),
    );

    let mut last_time = 0;
    let mut i = 0;

    while !signals.is_empty() {
        let (signal_iter, signal_ref_map) = signals.peek().unwrap();
        let Some((change_time, fst_id, value)) = signal_iter.next() else {
            signals.pop();
            continue;
        };

        assert!(
            change_time >= last_time,
            "change_time: {} < last_time: {}",
            change_time,
            last_time,
        );

        if change_time != last_time {
            out.emit_time_change(change_time)
                .expect("failed time change");
            for (var, a, b) in compare {
                let a = curr_value.get(a);
                let b = curr_value.get(b);
                out.emit_value_change(*var, &[(a == b) as u8 + b'0'])
                    .expect("failed to write compare");
            }
            last_time = change_time;
        }

        let handle = signal_ref_map.get(&fst_id.get_index()).unwrap();

        if !curr_value.is_empty() {
            curr_value
                .entry(*handle)
                .and_modify(|v| *v = value.to_vec());
        }

        out.emit_value_change(*handle, value)
            .expect("failed to write value change");

        // give it back to the queue
        signals.sift_down(0);

        if i % 5_000_000 == 0 {
            bar.set_position(last_time - input_range.0);
        }
        // if i % 10_000_000 == 0 {
        //     out.flush();
        // }
        i += 1;
    }
    bar.finish();
}

type SignalRefMap = std::collections::HashMap<usize, fstapi::Handle>;

fn write_hierarchy<'a, R: BufRead + Seek + 'a>(
    hiers: impl Iterator<Item = &'a mut FstReader<R>>,
    out: &mut Writer,
) -> (Vec<SignalRefMap>, HashMap<String, fstapi::Handle>) {
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
