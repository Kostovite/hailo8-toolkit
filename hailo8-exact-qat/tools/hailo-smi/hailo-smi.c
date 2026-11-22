/*
 * hailo-smi: nvidia-smi style status for local Hailo PCIe devices.
 *
 * Per device: board, architecture, firmware, serial, NN core clock, LCS, chip temperature (2 sensors), throttling
 * protection, PCIe link speed/width (sysfs), ACTIVITY, and the processes that hold /dev/hailoN open (pid, user,
 * elapsed, command line). Read-only firmware control calls.
 *
 * Activity = interrupts/s of the device's MSI vectors (/proc/interrupts, world readable). The driver registers them
 * only while some process holds /dev/hailoN, and every context switch / transfer completion raises one, so the rate
 * is ~0 when idle and hundreds-thousands/s while inferring. Sampled BEFORE this tool opens the device (its own
 * control requests interrupt too).
 *
 * Other users' processes: opening /proc/PID/fd of another uid needs CAP_DAC_READ_SEARCH, reading its links
 * CAP_SYS_PTRACE. `sudo make install` installs the binary setuid root; at start it keeps ONLY those two as permitted
 * (not effective), drops to the calling user, and raises them just for the /proc scan.
 *
 * Build: gcc -O2 -Wall -o hailo-smi hailo-smi.c -lhailort
 * Usage: hailo-smi [-l SECONDS]      (-l: refresh every SECONDS, like nvidia-smi -l; also works under `watch`)
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <hailo/hailort.h>
#include <limits.h>
#include <linux/capability.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAX_DEVICES 16

static int can_see_all;  /* root, or setuid install kept the two capabilities */
#define SCAN_CAPS ((1u << CAP_SYS_PTRACE) | (1u << CAP_DAC_READ_SEARCH))

static void read_sysfs(const char *path, char *out, size_t n)
{
    FILE *f = fopen(path, "r");
    out[0] = '\0';
    if (!f)
        return;
    if (fgets(out, (int)n, f))
        out[strcspn(out, "\n")] = '\0';
    fclose(f);
}

static void trimcpy(char *dst, const char *src, size_t len, size_t cap)
{
    size_t n = len < cap - 1 ? len : cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
    while (n && isspace((unsigned char)dst[n - 1]))
        dst[--n] = '\0';
}

static int set_caps(unsigned effective)
{
    struct __user_cap_header_struct h = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct d[2];
    memset(d, 0, sizeof(d));
    d[0].permitted = SCAN_CAPS;
    d[0].effective = effective;
    return (int)syscall(SYS_capset, &h, d);
}

/* setuid-root install: keep the /proc scan capabilities only (permitted, not effective), become the real user. */
static void drop_to_scan_caps(void)
{
    uid_t r = getuid();
    if (r == 0) {
        can_see_all = 1;
        return;
    }
    if (geteuid() != 0)
        return;
    prctl(PR_SET_KEEPCAPS, 1);
    if (setresgid(getgid(), getgid(), getgid()) || setresuid(r, r, r)) {
        perror("hailo-smi: drop privileges");
        exit(1);
    }
    can_see_all = set_caps(0) == 0;
    prctl(PR_SET_KEEPCAPS, 0);
}

static const char *arch_name(hailo_device_architecture_t a)
{
    switch (a) {
    case HAILO_ARCH_HAILO8_A0: return "Hailo-8 A0";
    case HAILO_ARCH_HAILO8: return "Hailo-8";
    case HAILO_ARCH_HAILO8L: return "Hailo-8L";
    default: return "other";
    }
}

/* Character-device minor of /dev/hailoN mapped to its PCIe BDF via sysfs (class hailo_chardev). */
static int dev_node_for_bdf(const char *bdf, char *node, size_t n)
{
    char path[PATH_MAX];
    DIR *d;
    struct dirent *e;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/hailo_chardev", bdf);
    d = opendir(path);
    if (!d)
        return -1;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hailo", 5) == 0) {
            snprintf(node, n, "/dev/%s", e->d_name);
            closedir(d);
            return 0;
        }
    }
    closedir(d);
    return -1;
}

/* Sum over CPUs of every MSI vector the device has registered right now (none while nobody holds it). */
static unsigned long long irq_count(const char *bdf)
{
    char path[PATH_MAX], row[8192];
    int irqs[64], n = 0;
    unsigned long long total = 0;
    DIR *d;
    FILE *f;
    struct dirent *e;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/msi_irqs", bdf);
    d = opendir(path);
    if (!d)
        return 0;
    while ((e = readdir(d)) && n < 64)
        if (isdigit((unsigned char)e->d_name[0]))
            irqs[n++] = atoi(e->d_name);
    closedir(d);
    if (!n || !(f = fopen("/proc/interrupts", "r")))
        return 0;
    while (fgets(row, sizeof(row), f)) {
        char *s = row, *end;
        int irq;
        while (*s == ' ')
            s++;
        if (!isdigit((unsigned char)*s))
            continue;
        irq = atoi(s);
        for (int i = 0; i < n; i++) {
            if (irqs[i] != irq)
                continue;
            s = strchr(s, ':') + 1;
            for (;;) {
                unsigned long long v = strtoull(s, &end, 10);
                if (end == s)
                    break;
                total += v;
                s = end;
            }
        }
    }
    fclose(f);
    return total;
}

static double now_mono(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* "pid  user  elapsed  cmdline" for one process. */
static void print_proc(const char *pid)
{
    char path[PATH_MAX], cmd[512] = "", user[32] = "?", el[16] = "?", stat_line[1024], up[64], row[640];
    struct stat st;
    FILE *f;
    size_t n = 0;
    snprintf(path, sizeof(path), "/proc/%s", pid);
    if (stat(path, &st) == 0) {
        struct passwd *pw = getpwuid(st.st_uid);
        if (pw)
            snprintf(user, sizeof(user), "%s", pw->pw_name);
        else
            snprintf(user, sizeof(user), "%u", (unsigned)st.st_uid);
    }
    snprintf(path, sizeof(path), "/proc/%s/cmdline", pid);
    if ((f = fopen(path, "r"))) {
        n = fread(cmd, 1, sizeof(cmd) - 1, f);
        fclose(f);
    }
    for (size_t i = 0; i + 1 < n; i++)
        if (!cmd[i])
            cmd[i] = ' ';
    cmd[n] = '\0';
    if (!cmd[0]) {
        snprintf(path, sizeof(path), "/proc/%s/comm", pid);
        read_sysfs(path, cmd, sizeof(cmd));
    }
    snprintf(path, sizeof(path), "/proc/%s/stat", pid);
    read_sysfs(path, stat_line, sizeof(stat_line));
    read_sysfs("/proc/uptime", up, sizeof(up));
    {   /* field 22 = starttime (clock ticks since boot); fields restart after the last ')' of comm */
        char *s = strrchr(stat_line, ')');
        int field = 2;
        while (s && *s && field < 22)
            if (*s++ == ' ')
                field++;
        if (s && *s) {
            long t = (long)(atof(up) - strtoull(s, NULL, 10) / (double)sysconf(_SC_CLK_TCK));
            if (t >= 86400)
                snprintf(el, sizeof(el), "%ldd%02ldh", t / 86400, t % 86400 / 3600);
            else
                snprintf(el, sizeof(el), "%02ld:%02ld:%02ld", t / 3600, t % 3600 / 60, t % 60);
        }
    }
    snprintf(row, sizeof(row), "pid %-9s  %-10.10s  %-10s  %s", pid, user, el, cmd);
    printf("|    %-86.86s |\n", row);
}

/* Print every process (other than this one) with an open fd on `node`. */
static int print_users(const char *node)
{
    DIR *proc = opendir("/proc");
    struct dirent *p;
    int count = 0, self = getpid();
    if (!proc)
        return 0;
    while ((p = readdir(proc))) {
        char fddir[PATH_MAX], link[PATH_MAX + 300], target[PATH_MAX];
        DIR *fds;
        struct dirent *fd;
        int hit = 0;
        if (!isdigit((unsigned char)p->d_name[0]) || atoi(p->d_name) == self)
            continue;
        snprintf(fddir, sizeof(fddir), "/proc/%s/fd", p->d_name);
        fds = opendir(fddir);
        if (!fds)
            continue;  /* other users' processes are unreadable without CAP_SYS_PTRACE */
        while (!hit && (fd = readdir(fds))) {
            ssize_t len;
            if (fd->d_name[0] == '.')
                continue;
            snprintf(link, sizeof(link), "%s/%s", fddir, fd->d_name);
            len = readlink(link, target, sizeof(target) - 1);
            if (len > 0) {
                target[len] = '\0';
                hit = strcmp(target, node) == 0;
            }
        }
        closedir(fds);
        if (!hit)
            continue;
        print_proc(p->d_name);
        count++;
    }
    closedir(proc);
    return count;
}

static void line(void) { puts("+-------------------------------------------------------------------------------------------+"); }

static int report(double window)
{
    hailo_device_id_t ids[MAX_DEVICES];
    unsigned long long irq0[MAX_DEVICES], irq1[MAX_DEVICES];
    size_t n = MAX_DEVICES;
    hailo_version_t lib;
    char drv[64], buf[64];
    time_t now = time(NULL);
    double t0, dt;
    hailo_status st = hailo_scan_devices(NULL, ids, &n);
    if (st != HAILO_SUCCESS) {
        fprintf(stderr, "hailo_scan_devices: %s\n", hailo_get_status_message(st));
        return 1;
    }
    t0 = now_mono();
    for (size_t i = 0; i < n; i++)
        irq0[i] = irq_count(ids[i].id);
    usleep((useconds_t)(window * 1e6));
    for (size_t i = 0; i < n; i++)
        irq1[i] = irq_count(ids[i].id);
    dt = now_mono() - t0;

    hailo_get_library_version(&lib);
    read_sysfs("/sys/module/hailo_pci/version", drv, sizeof(drv));
    strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Y", localtime(&now));
    printf("%s\n", buf);
    line();
    {
        char hdr[128];
        snprintf(hdr, sizeof(hdr), "HAILO-SMI   HailoRT %u.%u.%u   driver %s   devices %zu", lib.major, lib.minor, lib.revision,
                 drv[0] ? drv : "n/a", n);
        printf("| %-89s |\n", hdr);
    }
    line();
    printf("| %-2s   %-12s   %-8s   %-7s   %-11s   %-8s   %-8s   %-12s |\n", "#", "BDF", "Arch", "FW", "Temp C", "PCIe", "NN clock",
           "Activity");
    line();
    for (size_t i = 0; i < n; i++) {
        hailo_device dev;
        hailo_device_identity_t idn;
        hailo_extended_device_information_t ext;
        hailo_chip_temperature_info_t t;
        bool throttle = false;
        char serial[HAILO_MAX_SERIAL_NUMBER_LENGTH + 1] = "", temp[24] = "n/a", clock[16] = "n/a", act[24];
        char fw[24] = "n/a", arch[16] = "n/a", speed[32], width[16], path[PATH_MAX], link[32], node[300] = "";
        int have_ext, users = 0;
        double rate = dt > 0 && irq1[i] >= irq0[i] ? (irq1[i] - irq0[i]) / dt : 0;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/current_link_speed", ids[i].id);
        read_sysfs(path, speed, sizeof(speed));
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/current_link_width", ids[i].id);
        read_sysfs(path, width, sizeof(width));
        {   /* "8.0 GT/s PCIe" -> "Gen3 x4" */
            double gt = atof(speed);
            int gen = gt >= 31 ? 5 : gt >= 15 ? 4 : gt >= 7 ? 3 : gt >= 4 ? 2 : 1;
            snprintf(link, sizeof(link), "Gen%d x%s", gen, width[0] ? width : "?");
        }

        st = hailo_create_device_by_id(&ids[i], &dev);
        if (st != HAILO_SUCCESS) {
            printf("| %-2zu   %-12s   open failed: %-56.56s |\n", i, ids[i].id, hailo_get_status_message(st));
            continue;
        }
        if (hailo_identify(dev, &idn) == HAILO_SUCCESS) {
            snprintf(fw, sizeof(fw), "%u.%u.%u", idn.fw_version.major, idn.fw_version.minor, idn.fw_version.revision);
            snprintf(arch, sizeof(arch), "%s", arch_name(idn.device_architecture));
            trimcpy(serial, idn.serial_number, idn.serial_number_length, sizeof(serial));
        }
        have_ext = hailo_get_extended_device_information(dev, &ext) == HAILO_SUCCESS;
        if (have_ext)
            snprintf(clock, sizeof(clock), "%uMHz", ext.neural_network_core_clock_rate / 1000000);
        if (hailo_get_chip_temperature(dev, &t) == HAILO_SUCCESS)
            snprintf(temp, sizeof(temp), "%.1f/%.1f", t.ts0_temperature, t.ts1_temperature);
        hailo_get_throttling_state(dev, &throttle);
        hailo_release_device(dev);
        if (rate >= 1)
            snprintf(act, sizeof(act), "BUSY %.0f/s", rate);
        else
            snprintf(act, sizeof(act), "%s", irq1[i] ? "open, idle" : "idle");
        printf("| %-2zu   %-12s   %-8.8s   %-7s   %-11s   %-8s   %-8s   %-12.12s |\n", i, ids[i].id, arch, fw, temp, link, clock, act);
        {
            char info[128];
            int have_node = dev_node_for_bdf(ids[i].id, node, sizeof(node)) == 0;
            snprintf(info, sizeof(info), "%s   serial %s   LCS %s   throttle-protect %s", have_node ? node : "?",
                     serial[0] ? serial : "-", have_ext ? (snprintf(buf, sizeof(buf), "%u", ext.lcs), buf) : "?",
                     throttle ? "enabled" : "disabled");
            printf("|    %-86s |\n", info);
            if (have_node) {
                if (can_see_all && getuid())
                    set_caps(SCAN_CAPS);
                users = print_users(node);
                if (can_see_all && getuid())
                    set_caps(0);
            }
            if (!users)
                printf("|    %-86s |\n", "no processes");
        }
        line();
    }
    if (!can_see_all)
        printf("  only your own processes are listed; for all users: sudo make install (setuid; keeps only the /proc scan capabilities)\n");
    return 0;
}

int main(int argc, char **argv)
{
    double interval = 0;
    drop_to_scan_caps();
    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-l") || !strcmp(argv[i], "--loop")) && i + 1 < argc)
            interval = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-l SECONDS]\n", argv[0]);
            return 2;
        }
    }
    do {
        /* the activity window is the refresh interval itself (0.5 s for a one-shot) */
        double window = interval > 0.5 ? interval - 0.2 : 0.5;
        if (interval)
            printf("\033[H\033[2J");
        if (report(window))
            return 1;
        fflush(stdout);
    } while (interval);
    return 0;
}
