/*
 * hailo-smi: nvidia-smi style status for local Hailo PCIe devices.
 *
 * Per device: board, architecture, firmware, serial, NN core clock, LCS, chip temperature (2 sensors), throttling
 * protection, PCIe link speed/width (sysfs), and the processes that hold /dev/hailoN open. Read-only firmware control calls.
 *
 * Build: gcc -O2 -Wall -o hailo-smi hailo-smi.c -lhailort
 * Usage: hailo-smi [-l SECONDS]      (-l: refresh every SECONDS, like nvidia-smi -l)
 */
#include <ctype.h>
#include <dirent.h>
#include <hailo/hailort.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_DEVICES 16

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

/* Print every process with an open fd on `node`. */
static int print_users(const char *node)
{
    DIR *proc = opendir("/proc");
    struct dirent *p;
    int count = 0;
    if (!proc)
        return 0;
    while ((p = readdir(proc))) {
        char fddir[PATH_MAX], link[PATH_MAX + 300], target[PATH_MAX], cmd[256], comm_path[PATH_MAX];
        DIR *fds;
        struct dirent *fd;
        int hit = 0;
        if (!isdigit((unsigned char)p->d_name[0]))
            continue;
        snprintf(fddir, sizeof(fddir), "/proc/%s/fd", p->d_name);
        fds = opendir(fddir);
        if (!fds)
            continue;  /* other users' processes are unreadable without root */
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
        snprintf(comm_path, sizeof(comm_path), "/proc/%s/comm", p->d_name);
        read_sysfs(comm_path, cmd, sizeof(cmd));
        printf("|    pid %-8s %-59s |\n", p->d_name, cmd);
        count++;
    }
    closedir(proc);
    return count;
}

static void line(void) { puts("+-----------------------------------------------------------------------------+"); }

static int report(void)
{
    hailo_device_id_t ids[MAX_DEVICES];
    size_t n = MAX_DEVICES;
    hailo_version_t lib;
    char drv[64], buf[64];
    time_t now = time(NULL);
    hailo_status st = hailo_scan_devices(NULL, ids, &n);
    if (st != HAILO_SUCCESS) {
        fprintf(stderr, "hailo_scan_devices: %s\n", hailo_get_status_message(st));
        return 1;
    }
    hailo_get_library_version(&lib);
    read_sysfs("/sys/module/hailo_pci/version", drv, sizeof(drv));
    strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Y", localtime(&now));
    printf("%s\n", buf);
    line();
    {
        char hdr[128];
        snprintf(hdr, sizeof(hdr), "HAILO-SMI   HailoRT %u.%u.%u   driver %s   devices %zu", lib.major, lib.minor, lib.revision,
                 drv[0] ? drv : "n/a", n);
        printf("| %-75s |\n", hdr);
    }
    line();
    printf("| %-2s %-13s %-10s %-8s %-11s %-9s %-17s |\n", "#", "BDF", "Arch", "FW", "Temp C", "PCIe", "NNclock");
    line();
    for (size_t i = 0; i < n; i++) {
        hailo_device dev;
        hailo_device_identity_t idn;
        hailo_extended_device_information_t ext;
        hailo_chip_temperature_info_t t;
        bool throttle = false;
        char serial[HAILO_MAX_SERIAL_NUMBER_LENGTH + 1] = "", temp[24] = "n/a", clock[16] = "n/a";
        char fw[24] = "n/a", arch[16] = "n/a", speed[32], width[16], path[PATH_MAX], link[32], node[300] = "";
        int have_ext;

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
            printf("| %-2zu %-13s  open failed: %-46s |\n", i, ids[i].id, hailo_get_status_message(st));
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
        printf("| %-2zu %-13s %-10s %-8s %-11s %-9s %-17s |\n", i, ids[i].id, arch, fw, temp, link, clock);
        {
            char info[128];
            snprintf(info, sizeof(info), "serial %s   LCS %s   throttle-protect %s", serial[0] ? serial : "-",
                     have_ext ? (snprintf(buf, sizeof(buf), "%u", ext.lcs), buf) : "?", throttle ? "enabled" : "disabled");
            printf("|    %-72s |\n", info);
        }
        hailo_release_device(dev);
        if (dev_node_for_bdf(ids[i].id, node, sizeof(node)) == 0) {
            printf("|    %-72s |\n", node);  /* char device */
            print_users(node);
        }
        line();
    }
    return 0;
}

int main(int argc, char **argv)
{
    int interval = 0;
    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-l") || !strcmp(argv[i], "--loop")) && i + 1 < argc)
            interval = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-l SECONDS]\n", argv[0]);
            return 2;
        }
    }
    do {
        if (interval)
            printf("\033[H\033[2J");
        if (report())
            return 1;
        fflush(stdout);
        if (interval)
            sleep((unsigned)interval);
    } while (interval);
    return 0;
}
