# FST Merger

A tool for merging multiple FST (Fast Signal Trace) files together. This will
concatenate all signals from all the input files, side-by-side, merge-sorting
the timestamps, making it easier to view all files at the same time in a wave
visualizer, like GTKWave or Surfer.

## Example

![2024-02-29_22h29m15s_screenshot](https://github.com/Rodrigodd/vcd-merger/assets/51273772/1e21c935-eca1-42e0-ba80-709f416410ad)

In the example two traces ("gameroy" and "SM83_Run") where combined in a single
file for easier visualization. Also aditional signals where added for comparing
the registers A, F and the current instruction.

```
fst-merge ../gameroy/wave_trace/trace.fst ../sm83/Icarus/dmg_waves3.fst -o out.fst \
 --cmp 'f:0.gameroy.cpu.f=1.SM83_Run.dmgcore.alu_inst.F [7:0],a:0.gameroy.cpu.a=1.SM83_Run.dmgcore.bot.regs.A [7:0],op:0.gameroy.cpu.op=1.SM83_Run.dmgcore.seq.IR [7:0],dbus:0.gameroy.data_bus=1.SM83_Run.hw.databus [7:0]' \
 --offsets 0,0 \
 --range 0-480ms \
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

