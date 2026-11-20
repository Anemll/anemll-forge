# IOReport counters on the M6: GPU, Neural Engine, DRAM and SoC without root

A reference for measuring the GPU and the Apple Neural Engine (ANE) at work, and for monitors such as mactop.
Measured on a Mac18,5 (Apple M6, 32 GB, 12 GPU cores, two ANE clusters), macOS 27.3 build 26C5035r, 18 November
2026. IOReport is the private interface `powermetrics` reads; any user process can read it. This OS lists 8,027
channels at idle (`results/18nov/coupling/iorep_channels_idle.txt`, from `iorep_list.c`).

What is here: the counters we found useful, how to decode each, what they read during our loads, and what is
missing. The values quoted come from the coupling tests in [../../../ARTICLE_CONTENTION_M6.md](../../../ARTICLE_CONTENTION_M6.md).

## Reading IOReport

```c
// link with -framework CoreFoundation -lIOReport
CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef group, CFStringRef subgroup, uint64_t, uint64_t, uint64_t);
CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
void IOReportMergeChannels(CFDictionaryRef into, CFDictionaryRef from, CFTypeRef);
IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef want, CFMutableDictionaryRef *subbed, uint64_t, CFTypeRef);
CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef subbed, CFTypeRef);
CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef before, CFDictionaryRef after, CFTypeRef);
// per channel of the delta's "IOReportChannels" array:
CFStringRef IOReportChannelGetGroup(CFDictionaryRef), IOReportChannelGetSubGroup(CFDictionaryRef), IOReportChannelGetChannelName(CFDictionaryRef);
int IOReportChannelGetFormat(CFDictionaryRef);           // 1 simple, 2 state, 3 histogram
int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t);
int32_t IOReportStateGetCount(CFDictionaryRef);
CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);
```

Subscribe once to the groups you need (`GPU Stats`, `Energy Model`, `GPU HW Idle`, `PMP`, `SoC Stats`,
`GPU UT AggD Stats`), take a sample every interval and read the delta. [`iorep_sample.c`](iorep_sample.c) does this at
100 ms and writes JSON lines; [`telemetry.py`](telemetry.py) decodes them.

**Units.** State residencies in `24Mticks` are 24 MHz ticks (divide by 24e6 for seconds). The PMP group's
histograms count `events`: one event per PMP sample of the quantity, about 2,600 per second. The `Energy Model`
values are energy in the channel's unit (nJ for the GPU).

**PMP histograms** are state channels whose state names are bin labels (`4GB/s`, `1W`, `64Cyc`, ...). A label is the
bin's upper edge: the mean is `sum(count x (label - step / 2)) / sum(count)`. The last bin also collects everything
above it, so report the share of events in the last bin: when it is large, the mean is an underestimate.

## Calibration: known bytes at controlled rates

[`cal_analyze.py`](cal_analyze.py) on loads with known traffic (`coupling_run.py cal`): an ANE program that must read
16.8 MB per call (8 distinct 1 MB-class weights) with 0 to 4 ms of sleep between calls, a GPU 1 GiB stream at duty
cycles 1 to 0.1, and an INT8 GPU matmul streaming 128 MiB per dispatch at three weight reuses. Ratio = counter /
expected average rate (decimal GB/s):

| counter | expected 3 to 30 GB/s | 30 to 70 GB/s | 70 to 150 GB/s | behaviour |
|---|---|---|---|---|
| `PMP / AFI DCS BW` (ANE DRAM) | 1.09 to 1.52 | 0.93 to 1.01 | | tracks the average; reads about 2 GB/s too high at low rates (its lowest bin, 0 to 4 GB/s, counts as 2) |
| `PMP / DCS BW / AMCC` (DRAM total) | 1.06 to 1.18 | 0.85 to 0.90 | 0.83 (clipped) | tracks the average up to the 128 GB/s clip |
| `PMP / DCS BW / AGX` (GPU DRAM) | 3.3 to 6.4 (duty-cycled) | 1.6 (duty 0.5) | 0.93 to 1.04 (continuous) | **the rate while the GPU is active, not the average**: correct for continuous GPU work, too high for bursty work |
| `PMP / DCS BW / ANE* L*` (per ANE link) | 0.68 to 0.98 | 0.30 to 0.36 (bursty) | 0.89 (continuous) | correct only for continuous ANE work |

No constant scale factor: the counters are in decimal GB/s as labelled. For bursty GPU work (decode), take the GPU's
DRAM traffic as AMCC minus the other agents, or weight the AGX histogram by the GPU's active share.

## GPU

| channel (group / subgroup / name) | format | decode | what we saw |
|---|---|---|---|
| `GPU Stats / GPU Performance States / GPUPH` | state: OFF, P1 to P15 | residency share per state; MHz from the table below; active share = 1 - OFF | 1735 MHz (P13) under attention, matrix work and FMA; 1266 to 1431 MHz for a memory-bound stream |
| `GPU Stats / GPU Boost Controller Performance States / BSTGPUPH` | state | the boost controller's target state | |
| `GPU Stats / GPU Power Controller States / PWRCTRL` | state: IDLE_OFF, PERF, POWER, PWR_ZONE, FAST_DIE, UT, PRFLIMIT, PPMLIMIT, CLTMLIMI, PPMTARGE, DEADLINE, DEADAWAR, PRFBOOST, PMULOOP, STANDBY, SE, CORE_EFF, TURB0, AVGPERF, AVMR | which controller sets the clock; POWER, PWR_ZONE, PPMLIMIT, CLTMLIMI mean a power or thermal limit | DEADLINE and PERF under compute; CORE_EFF (lower clock for memory-bound work) under the stream; never a limit state |
| `GPU Stats / PMU Loop Lost Performance Reason Code States / PMU_RC` | state: NONE, GPUn_TH, GPUn_EMT, AFRn_*, FAB0_* | reasons for lost performance (thermal, EMT) | NONE throughout |
| `GPU Stats / PPM Target as % of Max GPU Power / GPU_PPM` | state: 0-9% to 100% | the power manager's target | 100% throughout |
| `GPU Stats / CLTM-induced GPU Performance States / GPU_CLTM` | state | thermal-management-induced states | NO_CLTM throughout |
| `GPU Stats / GPU Discrete Power Zone Residency / PZRSDNCY` | state: NO_ZONE, SLOWCLTM, ZONE0-4 | power zones | NO_ZONE |
| `GPU Stats / GPU Throttler Counters / *` | simple | throttle counts (HWPI, DPE, IVDM, C1PPT, ...) | 0 |
| `GPU UT AggD Stats / Count, Sum, Not_Engaged_Count / AGX.*_Perf_State_N` | simple | utilization-target engagement per P-state | attention and matrix work at P13; the stream spread over P7 to P13 |
| `GPU HW Idle / GPX_PS, GFX_PS` | state: IDLE, ACTIVE | hardware idle residency | |
| `Energy Model / GPU Energy` | simple, nJ | watts = nJ x 1e-9 / interval | close to `powermetrics`' GPU power |
| `PMP / Energy / AGX`, `AGX SRAM` | histogram, 1 W bins to 32 W (SRAM 0.125 W to 4 W) | mean watts (add both) | 16 to 21 W under matrix work |
| `PMP / AGX / Mem Latency` | histogram, 64-cycle bins to 4,096 | the GPU's memory latency | 190 to 221 cycles for attention alone, 382 to 429 next to the ANE FFN; the one counter that rises beyond the sum of two loads |
| `PMP / AGX / #Mem Trans` | histogram, steps of 16 to 1,024 | outstanding memory transactions | |
| `PMP / Volt-Temp HM / AGX_V_T` | histogram, states `mV_C` (for example `675_45`) | voltage and temperature bins of the GPU | 55 to 85 C during the tests |

GPU frequencies come from the IORegistry, not IOReport: the `pmgr` node's property `voltage-states9` holds pairs of
little-endian uint32 (frequency in Hz, voltage in mV). On this M6: index 0 = 0 (off), P1 to P13 = 502, 669, 834, 1002,
1090, 1178, 1266, 1354, 1442, 1530, 1620, 1680, 1735 MHz (`ioreg -l -w0 | grep voltage-states9`). GPUPH state `Pn`
maps to entry n.

## Neural Engine

The ANE has no clock or power channel in IOReport on this OS; its traffic is well covered.

| channel | format | decode | what we saw |
|---|---|---|---|
| `PMP / AFI DCS BW / AMCC RD, WR, RD+WR` | histogram, 4 GB/s bins to 128 | the ANE's DRAM traffic (responds only to ANE loads; equals the sum of the four ANE links) | FFN 85 GB/s cool, 66 to 75 warm; `hot` 100+ with 50 to 60% of samples in the top bin; in V8 prefill about 100 MB per token, so it tracks the prefill rate; in decode 98-104 on average with 58% of samples in the top bin (the true streaming rate is above 128: watch the clip share) |
| `PMP / AFI BW / AMCC RD, WR, RD+WR` | histogram, 4 GB/s bins to 128 | the ANE's fabric traffic; fabric minus DRAM = served by the system cache | |
| `PMP / DCS BW / ANE0 L0, ANE0 L1, ANE1 L0, ANE1 L1` (RD, WR, RD+WR) | histogram, 1 GB/s bins to 32 | DRAM traffic per cluster and link | always split evenly over the four links |
| `PMP / AF BW / ANE0 L0 ... ANE1 L1` | histogram, 1 GB/s bins to 32 | fabric traffic per cluster and link | |
| `PMP / SOC-NI Util BW / SOC-NI3 ANE0_UP, SOC-NI3 ANEXL1_, SOC-NI4 ANE1_UP, SOC-NI4 ANEXL0_` | histogram, 2 GB/s bins to 64 | utilization of each ANE port on the SoC network and of the cross-links between the clusters; NOT the ANE's DRAM traffic (see below) | FFN 26 to 28 per port; `hot` 32 to 38; 16 to 20 next to GPU attention; during a 64K prefill the ANE_UP pair fell 37 -> 31 while the ANE's DRAM traffic fell 64 -> 31 GB/s |
| `PMP / SOC-NI Util BW / SOC-NI3 ALL, SOC-NI4 ALL` | histogram, 2 GB/s bins to 64 | total per network interface | NI3 at its 64 GB/s top bin under `hot` |
| `PMP / DCS Floor / ANE-DCS-BW`, `ANE0`, `ANE1` | state: F2 to F10 | the ANE's vote for the DRAM clock | F10 when cool, F7 to F8 after minutes of work |
| `PMP / SOC Floor / ANE0-AF-BW, ANE1-AF-BW, ANE0-LINK0-AF-B, ANE0-LINK1-AF-B, ANE1-LINK0-AF-B, ANE1-LINK1-AF-B` | state: VMIN, VNOM, VMAX, VOVD, VOVD2 | the ANE's vote for the SoC voltage from its bandwidth | VOVD2 (the top) under the FFN |
| `PMP / Fast-Die CE / ANE0, ANE1` | histogram 0 to 100% | fast-die controller | 0 |
| `SoC Stats / Events / ANE_THROTTLE_ADCLK_TRIG, ANE_THROTTLE_DITHER_TRIG, ANE_THROTTLE_EXT_TRIG0-2` | state: INACT, ACT | ANE throttle triggers (residency in ticks) | 0.5 to 20 ms per 14 s phase, together with every other unit's trigger and `SOCHOT0_L`, only in high-current pairs |
| `SoC Stats / PMGR Counters / ANE0_SYS_IDLE, ANE0_CPU_IDLE, ANE1_*, PLL_ANE0_PLL0_CLK_OFF ... PLL_ANE1_PLL1_CLK_OFF, ANE0_DPE_CLK_ON, ANE0_SYS_CLK_ON, ANE0_BYP_CLK_ON` | simple | ANE idle and clock-gating counters | |
| `ANE`, `ANE1 / IOP State / status` | state: Off, Sleep, ..., Running | the ANE firmware processors' state | |
| `Interrupt Statistics (by index) / ane 2, ane1 2 / ... Handler Count` | simple | interrupts per interval | about 3 per FFN call |

`powermetrics --samplers ane_power` (root) gives the ANE's power ("ane_power" in mW in the plist's `processor`
dictionary); the plist has no ANE frequency on this OS. [`pm_service.sh`](pm_service.sh) runs it on request.

## DRAM, fabric and the system cache

| channel | format | decode | what we saw |
|---|---|---|---|
| `PMP / DCS BW / AMCC, AGX, PACC0, EACC0, ANE*` (RD, WR, RD+WR) | histogram (4 GB/s to 128 for AMCC, AGX, CPU) | DRAM traffic, total (AMCC) and per agent | GPU attention@60K 93 GB/s; total clips at 128 |
| `PMP / AF BW / same agents` | histogram | fabric traffic per agent; fabric minus DRAM = system-cache hits | INT8 matmul in cache: fabric 123, DRAM 3 |
| `SoC Stats / Events / DCS_F0 ... DCS_F10` | state: INACT, ACT | the DRAM clock state (the ACT residency of each) | F10 whenever the GPU is busy; F7 to F8 with the warm ANE alone |
| `PMP / DCS Floor / DCS-BW, AGX-DCS-BW, ANE-DCS-BW, PACC0, ...` | state: F2 to F10 | DRAM-clock votes per agent | the GPU always votes F10 |
| `PMP / DCS Ceiling / CTRL` | state: F2 to F10 | the DRAM clock ceiling | F10 |
| `PMP / PMC 2D State / SOCxDCS` | state: VMINxF2 ... VOVD2xF10 | joint SoC voltage x DRAM clock residency | |
| `SoC Stats / PMGR Counters / DCS0_IDLE ... DCS7_IDLE, DCS0_POWER_ON ...` | simple | the eight DRAM channels' idle and power-on counters | |
| `PMP / DSID Stats / GrpN Hits, GrpN Misses` | simple, bytes | system-cache hits and misses per group | always 0 on this OS |

GPU-side cache sizes from a working-set sweep (`gpu_load bw`): GPU caches about 1 to 2 MB, system cache about 8 MB.

## SoC power, voltage and thermal

| channel | format | decode | what we saw |
|---|---|---|---|
| `PMP / Energy / PACC0, PACC0 SRAM, EACC0, EACC0 SRAM` | histogram (P cluster 1 W bins, E cluster 0.25 W) | CPU cluster power | 10 to 11 W with eight spinning threads |
| `PMP / Volt-Temp HM / PACC0_V_T, EACC0_V_T` | histogram `mV_C` | cluster voltage and temperature | P cluster 75 to 95 C over a four-minute set |
| `SoC Stats / Events / SOC_VMIN ... SOC_VOVD2` | state | SoC voltage level residency | VOVD2 under any heavy load |
| `SoC Stats / Events / SOCHOT0_L` | state | the SoC's over-current / hot signal | brief, with the throttle triggers |
| `SoC Stats / Events / {PCPU,ECPU,GPU,ANE,SOC,DRAM,C1PPT}_THROTTLE_{SW,HW,ADCLK,DITHER,PPT,EXT}_TRIG*` | state | throttle triggers per unit | ADCLK and DITHER, briefly, all units at once |
| `PMP / PMC SOC Floor, PMC DCS Floor / CLPC, GFX, AP, BWR, ...` | state | SoC voltage and DRAM clock requests per requester | |
| `PPM Stats / *` | state | the power manager's droop, CPMS and ferocity controllers | 0% at idle |
| `CPU Stats / CPU Complex Performance States / EACC, PACC0`, `CPU Core Performance States / *` | state `VnPm` | CPU cluster and core P-states | |

## Missing or empty on this OS

- `Energy Model` lists only `GPU Energy` (and PCIe ports): no CPU, ANE or DRAM energy. Monitors that read CPU or ANE
  power from this group (macmon does) show nothing for them here; use the PMP power histograms for the CPU clusters
  and the GPU, and `powermetrics` (root) for the ANE.
- No ANE clock, P-state or temperature channel. The ANE's heat throttle shows only as lower ANE power and DRAM vote at
  constant work.
- `PMP / DSID Stats` reads 0.
- The bandwidth histograms clip at 128 GB/s (AMCC, AGX, AFI) and 32 GB/s per ANE link. Above that, only the share
  of top-bin samples tells how far.

## For a monitor (mactop)

Worth adding, all without root:

- ANE bandwidth: `AFI DCS BW` (DRAM; it matches the sum of the four `DCS BW ANE*` links within 1%), with the top-bin
  share. The `SOC-NI` ANE port histograms move with ANE activity but are not a bandwidth: do not sum them as one.
- DRAM bandwidth per unit (`DCS BW` AGX, ANE links, CPU clusters) and the DRAM clock (`DCS_F*`).
- GPU memory latency (`PMP / AGX / Mem Latency`): the clearest sign of memory contention.
- GPU limit reasons (`PWRCTRL`, `PMU_RC`, `GPU_PPM`), so a slow GPU can be told apart from a throttled one.
- Throttle and SOCHOT events, and the voltage-temperature histograms for temperatures.

Sampling every 100 ms gives about 260 PMP events per histogram, enough for a stable mean. Decode histograms as
above and show the clip share next to any bandwidth near the top bin.

## Tools

- [`iorep_list.c`](iorep_list.c): every channel with one delta sample.
- [`iorep_sample.c`](iorep_sample.c): the channels above as JSON lines (`IOREP_ALL=1`: every channel).
- [`telemetry.py`](telemetry.py): decoding and window summaries; [`scan_analyze.py`](scan_analyze.py): which counters
  respond to a load and which are not additive in a pair.
- [`pm_service.sh`](pm_service.sh): `powermetrics` on request for the ANE's power.
