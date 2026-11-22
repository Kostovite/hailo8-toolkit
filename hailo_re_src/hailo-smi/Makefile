hailo-smi: hailo-smi.c
	gcc -O2 -Wall -o $@ $< -lhailort

install: hailo-smi
	install -m 755 hailo-smi /usr/local/bin/
