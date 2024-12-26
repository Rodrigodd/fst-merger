// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// Small utility that reads in a VCD, GHW or FST file with wellen and then
// writes out the FST with the fst-writer library.
// Similar to vcd2fst, just that the input format does not have to be specified
// by the command name.

use clap::Parser;
use fstapi::{var_dir, var_type, Writer};
use indicatif::ProgressIterator;
use wellen::{
    simple::{self, Waveform},
    Hierarchy, HierarchyItem, Result, Scope, ScopeType, SignalEncoding, SignalRef, SignalValue,
    TimeTableIdx, Var, VarDirection, VarType,
};

#[derive(Parser, Debug)]
#[command(name = "2fst")]
#[command(author = "Kevin Laeufer <laeufer@cornell.edu>")]
#[command(version)]
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
    let mut waves: Vec<Waveform> = args
        .inputs
        .iter()
        .map(simple::read)
        .map(Result::unwrap)
        .collect();

    let timescales = waves.iter().map(|wave| {
        let timescale = wave.hierarchy().timescale().unwrap_or(wellen::Timescale {
            factor: 1,
            unit: wellen::TimescaleUnit::Unknown,
        });
        let mut timescale_exponent = timescale.unit.to_exponent().unwrap_or(0);
        let mut factor = timescale.factor;

        while factor % 10 == 0 {
            factor /= 10;
            timescale_exponent += 1;
        }

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
    let signal_ref_maps = write_hierarchy(waves.iter().map(|wave| wave.hierarchy()), &mut out);

    // load all signals into memory
    println!("Loading signals");
    for (wave, signal_ref_map) in waves.iter_mut().zip(signal_ref_maps.iter()).progress() {
        let all_signals: Vec<_> = signal_ref_map.keys().cloned().collect();
        wave.load_signals_multi_threaded(&all_signals);
    }

    let waves: Vec<_> = waves
        .into_iter()
        .zip(signal_ref_maps.iter())
        .zip(factors)
        .map(|((wave, signal_ref_map), factor)| {
            // sort signal ids in order to get a deterministic output
            let mut signal_ids: Vec<_> = signal_ref_map.iter().map(|(a, b)| (*a, *b)).collect();
            signal_ids.sort_by_key(|(wellen_id, _)| *wellen_id);

            wave.time_table();
            wave.get_signal(signal_ids[0].0).unwrap().iter_changes();

            (wave, factor, signal_ids)
        })
        .collect();
    write_value_changes(waves.as_slice(), &mut out);

    println!("Finishing writing FST file");
    // out.finish().expect("failed to finish writing the FST file");
}

/// Writes all value changes from the source file to the FST.
/// Note this is not the most efficient way to do this!
/// A faster version would write each signal directly to the FST instead
/// of writing changes based on the time step.
fn write_value_changes(
    waves: &[(Waveform, u32, Vec<(SignalRef, fstapi::Handle)>)],
    out: &mut Writer,
) {
    // PERF: use a n-way merge sort
    // println!("Sorting time table");
    let mut time_table = waves
        .iter()
        .enumerate()
        .flat_map(|(idx, (wave, factor, _))| {
            wave.time_table()
                .iter()
                .enumerate()
                .map(move |(time_idx, time)| (*time * *factor as u64, idx, time_idx))
        })
        .collect::<Vec<_>>();
    time_table.sort_unstable();

    let mut signals: Vec<_> = waves
        .iter()
        .map(|(wave, _, signal_ids)| {
            signal_ids
                .iter()
                .map(move |(wellen_ref, fst_id)| {
                    (
                        fst_id,
                        wave.get_signal(*wellen_ref)
                            .unwrap()
                            .iter_changes()
                            .peekable(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    println!("Writing value changes");

    let style = indicatif::ProgressStyle::default_bar()
        .template(PROGRESS_BAR_TEMPLATE)
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ");

    let bar = indicatif::ProgressBar::new(time_table.len() as u64).with_style(style);

    for (i, (time, wave_idx, time_idx)) in time_table.iter().enumerate() {
        out.emit_time_change(*time).expect("failed time change");

        let time_idx = *time_idx as TimeTableIdx;
        for (fst_id, signal_iter) in &mut signals[*wave_idx] {
            while signal_iter
                .peek()
                .map(|(change_idx, _)| *change_idx == time_idx)
                .unwrap_or(false)
            {
                let (_, value) = signal_iter.next().unwrap();
                if let Some(bit_str) = value.to_bit_string() {
                    out.emit_value_change(**fst_id, bit_str.as_bytes())
                        .expect("failed to write value change");

                    if i % 1_000_000 == 0 {
                        bar.set_position(i as u64);
                    }
                    if i % 10_000_000 == 0 {
                        out.flush();
                    }
                } else if let SignalValue::Real(value) = value {
                    todo!("deal with real value: {value}");
                } else {
                    todo!("deal with var len string");
                }
            }
        }
    }
    bar.finish();
}

type SignalRefMap = std::collections::HashMap<SignalRef, fstapi::Handle>;

fn write_hierarchy<'a>(
    hiers: impl Iterator<Item = &'a Hierarchy>,
    out: &mut Writer,
) -> Vec<SignalRefMap> {
    println!("Writing hierarchy");
    let mut signal_ref_maps = Vec::with_capacity(hiers.size_hint().0);
    for (i, hier) in hiers.enumerate() {
        out.set_scope(fstapi::scope_type::VCD_MODULE, &i.to_string(), "")
            .expect("failed to write top scope");
        let mut signal_ref_map = SignalRefMap::new();
        for item in hier.items() {
            match item {
                HierarchyItem::Scope(scope) => write_scope(hier, out, &mut signal_ref_map, scope),
                HierarchyItem::Var(var) => write_var(hier, out, &mut signal_ref_map, var),
            }
        }
        signal_ref_maps.push(signal_ref_map);
        out.set_upscope();
    }
    signal_ref_maps
}

fn write_scope(
    hier: &Hierarchy,
    out: &mut Writer,
    signal_ref_map: &mut SignalRefMap,
    scope: &Scope,
) {
    let name = scope.name(hier);
    let component = scope.component(hier).unwrap_or("");
    let tpe = match scope.scope_type() {
        ScopeType::Module => fstapi::scope_type::VCD_MODULE,
        ScopeType::Task => todo!(),
        ScopeType::Function => todo!(),
        ScopeType::Begin => todo!(),
        ScopeType::Fork => todo!(),
        ScopeType::Generate => todo!(),
        ScopeType::Struct => todo!(),
        ScopeType::Union => todo!(),
        ScopeType::Class => todo!(),
        ScopeType::Interface => todo!(),
        ScopeType::Package => todo!(),
        ScopeType::Program => todo!(),
        ScopeType::VhdlArchitecture => todo!(),
        ScopeType::VhdlProcedure => todo!(),
        ScopeType::VhdlFunction => todo!(),
        ScopeType::VhdlRecord => todo!(),
        ScopeType::VhdlProcess => todo!(),
        ScopeType::VhdlBlock => todo!(),
        ScopeType::VhdlForGenerate => todo!(),
        ScopeType::VhdlIfGenerate => todo!(),
        ScopeType::VhdlGenerate => todo!(),
        ScopeType::VhdlPackage => todo!(),
        ScopeType::GhwGeneric => todo!(),
        ScopeType::VhdlArray => todo!(),
    };
    out.set_scope(tpe, name, component)
        .expect("failed to write scope");

    for item in scope.items(hier) {
        match item {
            HierarchyItem::Scope(scope) => write_scope(hier, out, signal_ref_map, scope),
            HierarchyItem::Var(var) => write_var(hier, out, signal_ref_map, var),
        }
    }
    out.set_upscope();
}

fn write_var(hier: &Hierarchy, out: &mut Writer, signal_ref_map: &mut SignalRefMap, var: &Var) {
    let name = var.name(hier);
    let length = match var.signal_encoding() {
        SignalEncoding::String => todo!("support varlen!"),
        SignalEncoding::Real => todo!("support real!"),
        SignalEncoding::BitVector(len) => len,
    };
    let tpe = match var.var_type() {
        VarType::Event => fstapi::var_type::VCD_EVENT,
        VarType::Integer => fstapi::var_type::VCD_INTEGER,
        VarType::Parameter => fstapi::var_type::VCD_PARAMETER,
        VarType::Real => fstapi::var_type::VCD_REAL,
        VarType::Reg => fstapi::var_type::VCD_REG,
        VarType::Supply0 => fstapi::var_type::VCD_SUPPLY0,
        VarType::Supply1 => fstapi::var_type::VCD_SUPPLY1,
        VarType::Time => fstapi::var_type::VCD_TIME,
        VarType::Tri => fstapi::var_type::VCD_TRI,
        VarType::TriAnd => fstapi::var_type::VCD_TRIAND,
        VarType::TriOr => fstapi::var_type::VCD_TRIOR,
        VarType::TriReg => fstapi::var_type::VCD_TRIREG,
        VarType::Tri0 => fstapi::var_type::VCD_TRI0,
        VarType::Tri1 => fstapi::var_type::VCD_TRI1,
        VarType::WAnd => fstapi::var_type::VCD_WAND,
        VarType::Wire => fstapi::var_type::VCD_WIRE,
        VarType::WOr => fstapi::var_type::VCD_WOR,
        VarType::String => fstapi::var_type::GEN_STRING,
        VarType::Port => fstapi::var_type::VCD_PORT,
        VarType::SparseArray => fstapi::var_type::VCD_SPARRAY,
        VarType::RealTime => fstapi::var_type::VCD_REALTIME,
        VarType::Bit => fstapi::var_type::SV_BIT,
        VarType::Logic => fstapi::var_type::SV_LOGIC,
        VarType::Int => fstapi::var_type::SV_INT,
        VarType::ShortInt => fstapi::var_type::SV_SHORTINT,
        VarType::LongInt => fstapi::var_type::SV_LONGINT,
        VarType::Byte => fstapi::var_type::SV_BYTE,
        VarType::Enum => fstapi::var_type::SV_ENUM,
        VarType::ShortReal => fstapi::var_type::SV_SHORTREAL,
        VarType::Boolean => todo!(),
        VarType::BitVector => todo!(),
        VarType::StdLogic => todo!(),
        VarType::StdLogicVector => todo!(),
        VarType::StdULogic => todo!(),
        VarType::StdULogicVector => todo!(),
    };
    let dir = match var.direction() {
        VarDirection::Unknown => 0,
        VarDirection::Implicit => fstapi::var_dir::IMPLICIT,
        VarDirection::Input => fstapi::var_dir::INPUT,
        VarDirection::Output => fstapi::var_dir::OUTPUT,
        VarDirection::InOut => fstapi::var_dir::INOUT,
        VarDirection::Buffer => fstapi::var_dir::BUFFER,
        VarDirection::Linkage => fstapi::var_dir::LINKAGE,
    };

    let alias = signal_ref_map.get(&var.signal_ref()).cloned();
    let fst_signal_id = out
        .create_var(tpe, dir, length.get(), name, alias)
        .expect("failed to write variable");
    if alias.is_none() {
        signal_ref_map.insert(var.signal_ref(), fst_signal_id);
    }
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
