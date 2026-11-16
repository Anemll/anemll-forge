// A8 CPU expander (16 Nov 2026): LUT -> INT8 codes of one FFN chunk matrix on the CPU, the same output as the GPU's
// a8_expand (patch_a8.py): NEON 64-entry table lookup (vqtbl4q_u8, one table per vector component), GCD over
// (tile of 32 row-consecutive subgroups) x (64-column block); per column the tile's 32 subgroups write 1 KB contiguous.
// Reads the indices straight from the HWX mapping (zero-copy). tab: (pal_off, sg_off, row0) per subgroup, sorted by
// row0, G 16 (12 bytes of 6-bit indices per column after the subgroup's 64 bytes of fp16 row scales).
//   mode 0 (gate / up): dst [17][5120][1024];  mode 1 (down): dst [17][1024][5120], input channel x = xoff + c
// Build: clang -O3 -shared -o liba8cpu.dylib a8_cpu.c
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <string.h>

int a8_cpu_expand(const uint8_t *k, const uint32_t *tab, int nsub, int cin, int mode, int xoff, const int8_t *pal, int8_t *dst) {
    if (nsub % 32 || cin % 64) return -1;
    uint8_t t0[64], t1[64];
    for (int i = 0; i < 64; i++) { t0[i] = (uint8_t)pal[2 * i]; t1[i] = (uint8_t)pal[2 * i + 1]; }
    const uint8x16x4_t T0 = vld1q_u8_x4(t0), T1 = vld1q_u8_x4(t1);
    const int ntile = nsub / 32, ncb = cin / 64;
    dispatch_apply((size_t)ntile * ncb, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t it) {
        const int t = (int)(it / ncb), cb = (int)(it % ncb);
        const uint8_t *src[32];
        uint32_t o0[32];
        for (int s = 0; s < 32; s++) { src[s] = k + tab[3 * (t * 32 + s) + 1] + 64; o0[s] = 2 * tab[3 * (t * 32 + s) + 2]; }
        for (int c = cb * 64; c < cb * 64 + 64; c++) {
            const uint32_t x = (uint32_t)(xoff + c);
            int8_t *row = mode == 0 ? dst + (size_t)(o0[0] >> 10) * 5120 * 1024 + (size_t)c * 1024
                                    : dst + (size_t)(x >> 10) * 1024 * 5120 + (size_t)(x & 1023) * 5120;
            for (int s = 0; s < 32; s++) {
                const uint8_t *b = src[s] + (size_t)c * 12;
                uint64_t v0 = 0, v1 = 0;
                memcpy(&v0, b, 6); memcpy(&v1, b + 6, 6);
                uint8_t idx[16];
                for (int r = 0; r < 8; r++) { idx[r] = (uint8_t)((v0 >> (6 * r)) & 63); idx[8 + r] = (uint8_t)((v1 >> (6 * r)) & 63); }
                const uint8x16_t I = vld1q_u8(idx);
                const uint8x16x2_t z = vzipq_u8(vqtbl4q_u8(T0, I), vqtbl4q_u8(T1, I));
                uint8_t *d = (uint8_t *)row + (mode == 0 ? (o0[s] & 1023) : o0[s]);
                vst1q_u8(d, z.val[0]); vst1q_u8(d + 16, z.val[1]);
            }
        }
    });
    return 0;
}
