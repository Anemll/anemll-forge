// CPU power load for the coupling tests (18 Nov 2026): N threads of an FMA chain in registers (no memory traffic), for
// SECONDS. Adds CPU power to the SoC budget without touching the matrix units or the caches.   cpu_spin N SECONDS
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double secs;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *spin(void *p) {
    double a = 1.0 + (long)p * 1e-9, b = 0.999999, c = 1e-7, d = 1.000001, t0 = now(); long n = 0;
    while (now() - t0 < secs) { for (int i = 0; i < 1000000; i++) { a = a * b + c; d = d * b + c; } n++; }
    if (a + d == 12345.0) printf("%ld\n", n);
    return NULL;
}
int main(int argc, char **argv) {
    int n = atoi(argv[1]); secs = atof(argv[2]);
    pthread_t th[64];
    for (long i = 0; i < n && i < 64; i++) pthread_create(&th[i], NULL, spin, (void *)i);
    printf("ready %d threads\n", n); fflush(stdout);
    for (int i = 0; i < n && i < 64; i++) pthread_join(th[i], NULL);
    return 0;
}
