LLAMA   := third_party/llama.cpp
CFLAGS  := -O2 -Wall -Wno-unused-result -I$(LLAMA)/include -I$(LLAMA)/ggml/include
LDFLAGS := -L$(LLAMA)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,@executable_path/../$(LLAMA)/build/bin -lm

all: bin/sim bin/live bin/prefill_bench

bin/sim: src/sim.c src/common.h
	@mkdir -p bin
	$(CC) -O2 -Wall -o $@ src/sim.c -lm

bin/live: src/live.c src/kvengine.c src/kvengine.h src/common.h $(LLAMA)/build/bin/libllama.dylib
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ src/live.c src/kvengine.c $(LDFLAGS)

bin/prefill_bench: src/prefill_bench.c src/common.h $(LLAMA)/build/bin/libllama.dylib
	@mkdir -p bin
	$(CC) $(CFLAGS) -Wno-unused-function -o $@ src/prefill_bench.c $(LDFLAGS)

$(LLAMA)/build/bin/libllama.dylib:
	git clone --depth 1 https://github.com/ggml-org/llama.cpp $(LLAMA)
	cmake -S $(LLAMA) -B $(LLAMA)/build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON \
	      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
	cmake --build $(LLAMA)/build -j --target llama

models/qwen3-1.7b-q4_k_m.gguf:
	@mkdir -p models
	curl -L -o $@ https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf

clean:
	rm -rf bin data/kv_ssd_*

.PHONY: all clean
