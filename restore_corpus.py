from pathlib import Path
import gzip
import hashlib

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "第二步材料处理/复核修订版/chunks.jsonl"
EXPECTED = "3c714ad58408eabcdf03e7bcb2c37d5e696e507f28c86b445bd30af1ee4c7820"

def main():
    if TARGET.exists():
        if hashlib.sha256(TARGET.read_bytes()).hexdigest() != EXPECTED:
            raise SystemExit("Existing corpus differs; back it up before restoring. Nothing overwritten.")
        print("Corpus already verified.")
        return
    data = gzip.decompress(Path(str(TARGET) + ".gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise SystemExit("Corpus checksum mismatch; download the archive again.")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    print("Restored and verified 9564 chunks.")

if __name__ == "__main__":
    main()
