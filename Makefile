.PHONY: clean-ptx clean test

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > lantern-kernels/src/lib.rs
	touch lantern-kernels/build.rs
	touch lantern-examples/build.rs
	touch lantern-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
