// List every IOReport channel (group / subgroup / name / format / unit) with one 200 ms delta sample, no root.
#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <unistd.h>
typedef struct IOReportSubscription *IOReportSubscriptionRef;
extern CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef), IOReportChannelGetSubGroup(CFDictionaryRef), IOReportChannelGetChannelName(CFDictionaryRef), IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int IOReportChannelGetFormat(CFDictionaryRef);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t);
extern int32_t IOReportStateGetCount(CFDictionaryRef);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
extern int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);
static const char *s(CFStringRef x, char *b) { b[0] = 0; if (x) CFStringGetCString(x, b, 256, kCFStringEncodingUTF8); return b; }
int main(void) {
    CFDictionaryRef all = IOReportCopyAllChannels(0, 0);
    CFMutableDictionaryRef want = CFDictionaryCreateMutableCopy(NULL, 0, all), subbed = NULL;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, want, &subbed, 0, NULL);
    if (!sub) { fprintf(stderr, "no subscription\n"); return 1; }
    CFDictionaryRef a = IOReportCreateSamples(sub, subbed, NULL); usleep(200000);
    CFDictionaryRef b = IOReportCreateSamples(sub, subbed, NULL), d = IOReportCreateSamplesDelta(a, b, NULL);
    CFArrayRef ch = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
    char g[256], sg[256], n[256], u[256], st[256];
    for (CFIndex i = 0; i < CFArrayGetCount(ch); i++) {
        CFDictionaryRef c = CFArrayGetValueAtIndex(ch, i);
        int f = IOReportChannelGetFormat(c);
        printf("%s\t%s\t%s\tfmt=%d\tunit=%s\t", s(IOReportChannelGetGroup(c), g), s(IOReportChannelGetSubGroup(c), sg),
               s(IOReportChannelGetChannelName(c), n), f, s(IOReportChannelGetUnitLabel(c), u));
        if (f == 1) printf("val=%lld", IOReportSimpleGetIntegerValue(c, 0));
        else if (f == 2) { int k = IOReportStateGetCount(c); printf("states=%d:", k);
            for (int j = 0; j < k; j++) printf(" %s=%lld", s(IOReportStateGetNameForIndex(c, j), st), IOReportStateGetResidency(c, j)); }
        printf("\n");
    }
    return 0;
}
