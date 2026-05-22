Build bpe-tokenizer.cpp via

cmake -S . -B build build
cmake --build build -j 4

Then you may need to update your PYTHONPATH to point to the generated bpe_tokenizer.so shared lib

uv run tokenizer.ppy
