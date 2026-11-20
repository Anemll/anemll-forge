// IOReport sampler for the GPU / ANE coupling tests (18 Nov 2026, M6, no root). Every INTERVAL_MS it writes one JSON
// line of channel deltas: {"t": end epoch s, "dt": s, "c": {"group/subgroup/channel": value | {state: residency}}},
// nonzero entries only. Channels: GPU Stats (P-states, power-controller and limit reasons, PPM target, throttles),
// Energy Model (GPU energy), GPU HW Idle, PMP (power histograms, per-agent DRAM (DCS) and fabric (AF) bandwidth
// histograms, system-cache DSID hit / miss bytes, DCS floors, fast-die), SoC Stats PMGR counters of ANE / GFX.
//   iorep_sample INTERVAL_MS SECONDS OUT.jsonl     (SECONDS 0 = until SIGINT / SIGTERM; IOREP_ALL=1: all channels)
#include <CoreFoundation/CoreFoundation.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
typedef struct IOReportSubscription *IOReportSubscriptionRef;
extern CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
extern CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern void IOReportMergeChannels(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef), IOReportChannelGetSubGroup(CFDictionaryRef), IOReportChannelGetChannelName(CFDictionaryRef);
extern int IOReportChannelGetFormat(CFDictionaryRef);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t);
extern int32_t IOReportStateGetCount(CFDictionaryRef);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
extern int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);

static volatile sig_atomic_t stop_flag = 0;
static void on_sig(int s) { (void)s; stop_flag = 1; }
static const char *cs(CFStringRef x, char *b, int n) { b[0] = 0; if (x) CFStringGetCString(x, b, n, kCFStringEncodingUTF8); return b; }
static void jstr(FILE *f, const char *s) {   // JSON string, trimmed
    while (*s == ' ') s++;
    size_t n = strlen(s); while (n && s[n - 1] == ' ') n--;
    fputc('"', f);
    for (size_t i = 0; i < n; i++) { if (s[i] == '"' || s[i] == '\\') fputc('\\', f); fputc((unsigned char)s[i] < 32 ? ' ' : s[i], f); }
    fputc('"', f);
}
static int has(const char *s, const char *t) { return strstr(s, t) != NULL; }
static int ALL = 0;   // IOREP_ALL=1: every channel of every group (the full scan)
static int keep(const char *g, const char *sg, const char *n) {
    if (ALL) return 1;
    if (!strcmp(g, "Energy Model")) return has(n, "GPU");
    if (!strcmp(g, "GPU HW Idle")) return 1;
    if (!strcmp(g, "GPU Stats"))
        return has(sg, "Performance States") || has(sg, "Power Controller") || has(sg, "Lost Performance") || has(sg, "PPM Target")
            || has(sg, "Throttler") || has(sg, "Power Zone") || has(sg, "DVD Request") || has(sg, "Fender State") || has(sg, "UV Warn")
            || has(sg, "THM Counters");
    if (!strcmp(g, "PMP")) {
        if (!strcmp(sg, "Energy") || !strcmp(sg, "DSID Stats") || !strcmp(sg, "Fast-Die CE") || !strcmp(sg, "DCS Floor")
            || !strcmp(sg, "DCS Ceiling") || !strcmp(sg, "PMC 2D State") || !strcmp(sg, "AGX") || !strcmp(sg, "SOC-NI Util BW")) return 1;
        if (!strcmp(sg, "PMC DCS Floor") || !strcmp(sg, "PMC SOC Floor")) return 1;
        // added after the full scan (18 Nov): ANE fabric / DRAM traffic (AFI), SoC voltage floors per agent, CPU
        // stall histograms, every voltage-temperature histogram, PMC driver residencies
        if (!strcmp(sg, "AFI BW") || !strcmp(sg, "AFI DCS BW") || !strcmp(sg, "SOC Floor") || !strcmp(sg, "PACC0")
            || !strcmp(sg, "EACC0") || !strcmp(sg, "Volt-Temp HM") || !strcmp(sg, "PMC Driver")) return 1;
        if (!strcmp(sg, "DCS BW") || !strcmp(sg, "AF BW"))
            return has(n, "AMCC") || has(n, "AGX") || has(n, "ANE") || has(n, "PACC") || has(n, "EACC");
        return 0;
    }
    if (!strcmp(g, "SoC Stats"))   // + throttle / SOCHOT / DCS and SoC voltage events and event counts (full scan)
        return (!strcmp(sg, "PMGR Counters") && (has(n, "ANE") || has(n, "GFX") || has(n, "AFGGFX") || has(n, "DCS")))
            || !strcmp(sg, "Events") || !strcmp(sg, "Event Counts");
    if (!strcmp(g, "GPU UT AggD Stats")) return 1;
    if (!strcmp(g, "Interrupt Statistics (by index)")) return !strcmp(sg, "ane 2") || !strcmp(sg, "ane1 2");   // per ANE cluster
    return 0;
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "iorep_sample INTERVAL_MS SECONDS OUT.jsonl\n"); return 2; }
    int iv = atoi(argv[1]); double secs = atof(argv[2]);
    FILE *f = fopen(argv[3], "w"); if (!f) { perror("open"); return 1; }
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    const char *grp[][2] = {{"GPU Stats", NULL}, {"Energy Model", NULL}, {"GPU HW Idle", NULL}, {"PMP", NULL}, {"SoC Stats", NULL}, {"GPU UT AggD Stats", NULL},
                            {"Interrupt Statistics (by index)", "ane 2"}, {"Interrupt Statistics (by index)", "ane1 2"}};
    ALL = getenv("IOREP_ALL") && !strcmp(getenv("IOREP_ALL"), "1");
    CFDictionaryRef chans = ALL ? IOReportCopyAllChannels(0, 0) : NULL;
    for (size_t i = 0; !ALL && i < sizeof grp / sizeof grp[0]; i++) {
        CFStringRef g = CFStringCreateWithCString(NULL, grp[i][0], kCFStringEncodingUTF8);
        CFStringRef sg = grp[i][1] ? CFStringCreateWithCString(NULL, grp[i][1], kCFStringEncodingUTF8) : NULL;
        CFDictionaryRef c = IOReportCopyChannelsInGroup(g, sg, 0, 0, 0);
        if (!c) continue;
        if (!chans) chans = c; else IOReportMergeChannels(chans, c, NULL);
    }
    CFMutableDictionaryRef want = CFDictionaryCreateMutableCopy(NULL, 0, chans), subbed = NULL;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, want, &subbed, 0, NULL);
    if (!sub) { fprintf(stderr, "no IOReport subscription\n"); return 1; }
    CFDictionaryRef prev = IOReportCreateSamples(sub, subbed, NULL);
    double tp = now(), t0 = tp, next = tp;
    char g[256], sg[256], n[256], st[256];
    while (!stop_flag && (secs <= 0 || now() - t0 < secs)) {
        next += iv * 1e-3;
        double w = next - now(); if (w > 0) usleep((useconds_t)(w * 1e6));
        CFDictionaryRef cur = IOReportCreateSamples(sub, subbed, NULL);
        double t = now();
        CFDictionaryRef d = IOReportCreateSamplesDelta(prev, cur, NULL);
        CFRelease(prev); prev = cur;
        CFArrayRef ch = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
        fprintf(f, "{\"t\":%.4f,\"dt\":%.4f,\"c\":{", t, t - tp); tp = t;
        int first = 1;
        for (CFIndex i = 0; ch && i < CFArrayGetCount(ch); i++) {
            CFDictionaryRef c = CFArrayGetValueAtIndex(ch, i);
            cs(IOReportChannelGetGroup(c), g, 256); cs(IOReportChannelGetSubGroup(c), sg, 256); cs(IOReportChannelGetChannelName(c), n, 256);
            if (!keep(g, sg, n)) continue;
            int fm = IOReportChannelGetFormat(c);
            char key[800]; snprintf(key, sizeof key, "%s/%s/%s", g, sg, n);
            if (fm == 1) {
                int64_t v = IOReportSimpleGetIntegerValue(c, 0);
                if (!v) continue;
                if (!first) fputc(',', f); first = 0; jstr(f, key); fprintf(f, ":%lld", v);
            } else if (fm == 2) {
                int k = IOReportStateGetCount(c), any = 0;
                for (int j = 0; j < k; j++) if (IOReportStateGetResidency(c, j)) { any = 1; break; }
                if (!any) continue;
                if (!first) fputc(',', f); first = 0; jstr(f, key); fputs(":{", f);
                int f2 = 1;
                for (int j = 0; j < k; j++) {
                    int64_t r = IOReportStateGetResidency(c, j); if (!r) continue;
                    char lab[300]; cs(IOReportStateGetNameForIndex(c, j), st, 256);
                    snprintf(lab, sizeof lab, "%s#%d", st, j);   // index kept: some state names repeat or are empty
                    if (!f2) fputc(',', f); f2 = 0; jstr(f, lab); fprintf(f, ":%lld", r);
                }
                fputc('}', f);
            }
        }
        fputs("}}\n", f); fflush(f);
        CFRelease(d);
    }
    fclose(f);
    return 0;
}
