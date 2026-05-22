import os
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path
import regex as re
import bpe_tokenizer
from collections import Counter
import pickle


PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pre_tokenize(corpus_with_tokens: tuple[str, list[str]], encoding=False) -> dict[str, int] | Iterator:
    freq = {}
    corpus_chunk, special_tokens = corpus_with_tokens
    if encoding:
        # while encoding, keep the special tokens
        escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]                
        if escaped_tokens:                                                               
            split_pattern = f"({'|'.join(escaped_tokens)})"
            parts = re.split(split_pattern, corpus_chunk)                                
            tokens = []                                                                  
            for part in parts:                                                         
                if part in special_tokens:                                               
                    tokens.append(re.match(f"({'|'.join(escaped_tokens)})", part))       
                else:                                                                  
                    tokens.extend(re.finditer(PAT, part))                                
            yield from iter(tokens)
            return                                                          
        yield from re.finditer(PAT, corpus_chunk) 

    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = f"{"|".join(escaped_tokens)}"
    delimited_corpus = re.split(pattern, corpus_chunk)
    for corpus in delimited_corpus:
        for pre_token in re.finditer(PAT, corpus):
            encoded_tup = pre_token.group()
            if encoded_tup not in freq:
                freq[encoded_tup] = 0

            freq[encoded_tup] += 1
    return freq

class BPETokenizer():
    
    def __init__(self, vocab: dict[tuple[int, bytes]] | None = None, merges: list[tuple[bytes, bytes]] | None = None, special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.inv_vocab = {val: key for (key, val) in vocab.items()} if vocab else {}
        self.merges = merges if merges is not None else []
        self.special_tokens = special_tokens if special_tokens is not None else []
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
            Construct Tokenizer from a vocab and merges filepath
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
            with open(merges_filepath, "rb") as merge:
                merges = pickle.load(merge)
                return cls(vocab, merges, special_tokens)
            
    def encode(self, text: str) -> list[int]:
        """
            Encode a given string into a list of tokens
        """
        tokens = []
        for token in _pre_tokenize((text, self.special_tokens), encoding=True):
            pre_token = token.group()
            if pre_token in self.special_tokens:
                tokens.append(self.inv_vocab[pre_token.encode("utf-8")])
                continue
            
            merged_idx = 0
            encoded_token = pre_token.encode("utf-8")
            merged_pairs = [bytes([b]) for b in encoded_token]
            merged_candidates = Counter([(bytes([encoded_token[i]]), bytes([encoded_token[i+1]])) for i in range(len(encoded_token)-1)])
            while len(merged_pairs) != 1 and merged_idx < len(self.merges):
                if self.merges[merged_idx] in merged_candidates:
                    indices = [i for i in range(len(merged_pairs)-2, -1, -1) if (merged_pairs[i], merged_pairs[i+1]) == self.merges[merged_idx]]
                    for idx in indices:
                        # remove idx+1 as that has been merged with idx
                        old_left = merged_pairs[idx]
                        old_right = merged_pairs[idx+1]
                        merged_pairs[idx] = merged_pairs[idx]+merged_pairs[idx+1]

                        if idx+2 < len(merged_pairs):
                            merged_candidates[(old_right, merged_pairs[idx+2])] -= 1
                            if merged_candidates[(old_right, merged_pairs[idx+2])] == 0:
                                del merged_candidates[(old_right, merged_pairs[idx+2])]

                            merged_candidates[(merged_pairs[idx], merged_pairs[idx+2])] += 1

                        if idx > 0:
                            merged_candidates[(merged_pairs[idx-1], old_left)] -= 1
                            if merged_candidates[(merged_pairs[idx-1], old_left)] == 0:
                                del merged_candidates[(merged_pairs[idx-1], old_left)]

                            merged_candidates[(merged_pairs[idx-1], merged_pairs[idx])] += 1
                        
                        merged_candidates[(old_left, old_right)] -= 1

                        del merged_pairs[idx+1]
                merged_idx += 1
            
            for m in merged_pairs:
                tokens.append(self.inv_vocab[m])

        return tokens
    
    def encode_iterable(self, text: Iterable) -> Iterator:
        for t in text:
            yield from self.encode(t)
    
    def decode(self, token_ids: list[int]) -> str:
        decoded = bytearray()
        for token_id in token_ids:
            decoded.extend(self.vocab[token_id])
        
        try:
            return decoded.decode("utf-8")
        except UnicodeDecodeError:
            return decoded.decode("latin-1")
    
    def save_to(self, vocab_filepath: str, merges_filepath: str):
        with open(vocab_filepath, "wb") as f:
            pickle.dump(self.vocab, f)
        
        with open(merges_filepath, "wb") as f:
            pickle.dump(self.merges, f)
    
    @staticmethod
    def train(corpus: Path, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(corpus, "rb") as f:
            num_workers = 8
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

            chunks = []
        
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            
            with Pool(num_workers) as pool:
                # parallelize with n_workers processes and create Counters
                counters = [Counter(freq) for freq in pool.map(_pre_tokenize, [(chunk, special_tokens) for chunk in chunks])]
                freq_counter = counters[0]
                for counter in counters[1:]:
                    freq_counter += counter
                
                bytes_freq = dict(freq_counter)
                trained_tokenizer = bpe_tokenizer.train_bpe_tokenizer(bytes_freq, vocab_size-(256+len(special_tokens)), 255+len(special_tokens))
                base = 256 + len(special_tokens)
                token_ids = {i: bytes([i]) for i in bytes(range(256))} | {256+i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))} | {base+i: trained_tokenizer.token_ids[base+i] for i in range(len(trained_tokenizer.merged_pairs))}
                merges = [(merged_pair[0], merged_pair[1]) for merged_pair in trained_tokenizer.merged_pairs]
                return token_ids, merges
                
    
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    # vocab, merges = BPETokenizer().train(corpus=Path("data/openwebtext.txt"), vocab_size=10000, special_tokens=["<|endoftext|>"])
    # tokenizer = BPETokenizer(vocab, merges, ["<|endoftext|>"])
    # tokenizer.save_to("data/corpus_vocab.pkl", "data/corpus_merges.pkl")
    # tokenizer = BPETokenizer.from_files("data/corpus_vocab.pkl", "data/corpus_merges.pkl", ["<|endoftext|>"])
    # tokens = tokenizer.encode("this is Aryan testing his <|endoftext|> tokenizer out")
    #print(tokens)
    #print(len(tokens))
    #print(tokenizer.decode(tokens))


                

