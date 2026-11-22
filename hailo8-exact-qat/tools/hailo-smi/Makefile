hailo-smi: hailo-smi.c
	gcc -O2 -Wall -o $@ $< -lhailort

# setuid root so every user sees every process on the chips; the binary keeps only CAP_SYS_PTRACE and drops to the
# caller at start. (/dev/hailo* is already world rw via HailoRT's 51-hailo-udev.rules.)
install: hailo-smi
	install -o root -g root -m 4755 hailo-smi /usr/local/bin/
