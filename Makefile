NUM_DEVICES ?= 1

train-cbow:
	python cli.py --train --num_devices $(NUM_DEVICES) --out_dir out/cbow --device_type gpu

test-cbow:
	python cli.py --out_dir out/cbow

train-rnn:
	python cli.py --mode rnn --train --num_devices $(NUM_DEVICES) --out_dir out/rnn --device_type gpu --fresh

test-rnn:
	python cli.py --mode rnn --out_dir out/rnn

train-att:
	python cli.py --init_lr 0.9 --batch_size 512 --mode att --num_steps 1000 --train --num_devices $(NUM_DEVICES) --out_dir out/att --device_type gpu --fresh

test-att:
	python cli.py --mode att --out_dir out/att
