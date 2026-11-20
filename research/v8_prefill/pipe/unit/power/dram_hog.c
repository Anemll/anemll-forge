// DRAM pressure from the CPU (19 Nov 2026): N threads streaming 512 MB buffers (read + write, memcpy) for S seconds;
// prints the achieved copy bandwidth. A stand-in for the ANE's weight stream in GPU-alone contention tests.
//   dram_hog THREADS SECONDS
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double secs; static volatile long bytes[32];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *run(void *a) {
    long id = (long)a; size_t n = 512ul << 20; char *s = malloc(n), *d = malloc(n);
    memset(s, 1, n); memset(d, 2, n);
    double t0 = now();
    while (now() - t0 < secs) { memcpy(d, s, n); __asm__ volatile("" :: "r"(d), "r"(s) : "memory"); bytes[id] += 2 * n; }
    free(s); free(d); return 0;
}
int main(int c, char **v) {
    int nt = atoi(v[1]); secs = atof(v[2]); pthread_t th[32];
    double t0 = now();
    for (long i = 0; i < nt; i++) pthread_create(&th[i], 0, run, (void *)i);
    for (int i = 0; i < nt; i++) pthread_join(th[i], 0);
    long tot = 0; for (int i = 0; i < nt; i++) tot += bytes[i];
    printf("dram_hog %d threads: %.1f GB/s (read + write)\n", nt, tot / (now() - t0) / 1e9);
    return 0;
}
