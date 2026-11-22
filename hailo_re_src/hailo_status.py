"""Status of every local Hailo-8: identity, firmware, temperature, throttling, power (one sample per device).

Usage (hailo env): python hailo_status.py [--watch SECONDS]
Read-only firmware control calls; safe while a network is running, but avoid it during a timing measurement.
"""
import sys
import time

from hailo_platform import Device


def sample(dev):
    c = dev.control
    out = {}
    try:
        t = c.get_chip_temperature()
        out["temp_C"] = f"{t.ts0_temperature:.1f}/{t.ts1_temperature:.1f}"
    except Exception as e:  # noqa: BLE001
        out["temp_C"] = f"n/a ({type(e).__name__})"
    try:
        out["throttle_protect"] = "enabled" if c.get_throttling_state() else "disabled"
    except Exception as e:  # noqa: BLE001
        out["throttle_protect"] = f"n/a ({type(e).__name__})"
    try:
        out["power_W"] = f"{c.power_measurement():.2f}"
    except Exception as e:  # noqa: BLE001
        out["power_W"] = "n/a (board has no current sensor)" if "Unsupported" in type(e).__name__ else f"n/a ({type(e).__name__})"
    return out


def main():
    watch = float(sys.argv[sys.argv.index("--watch") + 1]) if "--watch" in sys.argv else 0
    ids = Device.scan()
    devs = [Device(i) for i in ids]
    for i, d in zip(ids, devs):
        idn = d.control.identify()
        print(f"{i}: {idn.board_name.strip()} fw {idn.firmware_version} serial {idn.serial_number.strip()}")
    while True:
        print(time.strftime("%H:%M:%S"), "  ".join(f"[{i}] " + " ".join(f"{k}={v}" for k, v in sample(d).items())
                                                for i, d in zip(ids, devs)), flush=True)
        if not watch:
            break
        time.sleep(watch)


if __name__ == "__main__":
    main()
