# .spk Reader (tv/Mfile native format) — Design Spec

Date: 2026-07-09

## Purpose

Add support for loading gamma-ray spectra from tv's native `.spk` files,
alongside the existing ASCII `.txt` and gf3-native `.spe` formats. Second
of three sub-projects (`.spe` reader → `.spk` reader → peak fitting)
building toward the "PeakFinderFitting" project's next phase.

## Correction to prior research

`docs/research/2026-07-09-peak-finding-fitting-and-spe-spk-formats.md`
concluded `.spk` support exists only in gf3 (`HHIRFSPK` container), not in
tv/libmfile at all. That conclusion was wrong — it came from grepping the
source for the literal substring "spk", which tv/libmfile's own code never
contains (format identification is done by magic number/trailer content,
never by name). The real format is Stefan Esser's **Mfile library**
(`libmfile-1.0.7`), which tv's manual describes as its standard spectrum
I/O layer ("tv führt I/O Operationen mit der Mfile-Bibliothek durch").
gf3's `HHIRFSPK` container is unrelated and out of scope here — see
"Out of scope" below. A correction note pointing here has been added to
the research doc.

## Format (empirically confirmed against two real files: `demo.spk` and `pg_25um_f.spk`)

Mfile identifies formats purely by content — never by file extension — so
`load_spk` mirrors that: try each signature in turn.

### Variant 1: MAT_LC ("line-compressed") — confirmed via `demo.spk`

Byte 0: magic `0x80FFFF10` (uint32, little-endian). Header is 44 bytes,
11 little-endian uint32 fields:

```
offset  field
 0      magic            (0x80FFFF10)
 4      version          (1 = LC1, 2 = LC2)
 8      levels
12      lines
16      columns          channel count
20      poslentablepos   byte offset of the position/length table
24      freepos          next free byte offset (unused for reading)
28      freelistpos       (unused for reading)
32      used              (unused, "not yet implemented" per source)
36      free              (unused for reading)
40      status            (unused for reading)
```

At `poslentablepos`: `levels * lines` entries of `(pos: uint32, len:
uint32)` — one entry per "line" of the matrix. A 1-D spectrum has
`levels == lines == 1`, i.e. exactly one entry, pointing to the
compressed byte range holding all `columns` channel values.

Verified exactly against `demo.spk`: magic matches, version=2, levels=1,
lines=1, columns=4096, poslentablepos=44 (= header size), and the single
poslentable entry's `pos`/`len` account for every remaining byte in the
file with no gap or overlap.

**Decompression** (`lc1_uncompress`/`lc2_uncompress`, ported from
`libmfile-1.0.7/src/lc_c1.c`/`lc_c2.c`): both are running-delta codecs —
each decoded value updates a `last` accumulator, encoded via zigzag
(`(i<<1)` for `i>=0`, `~(i<<1)` for `i<0`) and packed into
variable-width tag bytes (3 values/byte, 2 values/byte, 1 value/byte, or
an extended multi-byte form for large deltas). LC2 additionally has a
run-length "same value repeated" tag that LC1 lacks. Full bit-level
layout traced from source; ported byte-for-byte, decompression only (no
compressor — this app only reads `.spk`, never writes it).

If a poslentable entry's `len` is 0, the line is defined as all-zero
(matches `lc_getput.c`'s `readline`).

### Variant 2: "oldmat" raw array (e.g. MAT_LF4) — confirmed via `pg_25um_f.spk`

No header — the file *is* the raw array, starting at byte 0: `columns`
consecutive fixed-width values in one of a small set of dtypes. Format is
identified by a **64-byte trailer at the end of the file**: ASCII text
`"\nMatFmt: "` followed by `<cols>[k].<fmtname>[:<version>]` (e.g. our
sample: `"16k.lf4:2"` → columns=16384, format=lf4), then a NUL, then
uninitialized padding out to the fixed 64-byte block (`oldmat_header` is
literally `char[64]` in the source; the padding is stack garbage in the
reference implementation and must simply be ignored, not interpreted).

Verified exactly against `pg_25um_f.spk`: trailer text matches, and
`16384 columns × 4 bytes (lf4 = little-endian float32) + 64-byte trailer
= 65600 bytes` matches the file's exact size.

Supported `fmtname` → numpy dtype table (only the simple, unambiguous,
1-D dtypes — see "Out of scope" for exclusions):

| fmtname | dtype |
|---|---|
| `le4` | `<i4` |
| `he4` | `>i4` |
| `le2` | `<u2` |
| `he2` | `>u2` |
| `le2s` | `<i2` |
| `he2s` | `>i2` |
| `lf4` | `<f4` |
| `hf4` | `>f4` |

The format-string grammar (ported from `libmfile-1.0.7/src/minfo.c`'s
`mtxttoinfo`) allows up to three dot-separated leading numbers
(`levels.lines.columns`, each with an optional `k` suffix meaning
`*1024`); a single leading number (our real file's case) means just
`columns`, with `levels`/`lines` defaulting to 1.

## Module: `spk_io.py`

New module, mirroring `spe_io.py`'s shape:

```python
def load_spk(path: str) -> np.ndarray
```

Dispatch:
1. First 4 bytes decode as little-endian uint32 `0x80FFFF10` → MAT_LC
   path. Reject (`ParseError`) if header `levels != 1` or `lines != 1`
   (a true 2-D coincidence matrix — out of scope) or `version` isn't 1 or
   2. Read the one poslentable entry, decompress with LC1 or LC2 per the
   header's `version`. Decompression bounds-checks every read against
   the declared compressed-block length and the requested channel count
   (mirroring the C code's own `nleft < 0` guards), raising `ParseError`
   on a truncated or malformed stream rather than reading past the end
   of the buffer.
2. Else, if the file is at least 64 bytes and the last 64 bytes start
   with `"\nMatFmt: "` → oldmat path. Parse the trailer with the ported
   `mtxttoinfo` grammar, raising `ParseError` on any grammar violation
   (not just an unrecognized `fmtname`) or on `levels/lines != 1`;
   verify total file size equals `columns * itemsize + 64` exactly
   (`ParseError` if not — a size mismatch means the file is corrupt or
   truncated), then `np.frombuffer` the raw array.
3. Neither signature found → `ParseError` with a clear message (not a
   guess — see "Out of scope").

Returns channel counts as `np.int64` (rounding for the float dtypes),
matching `histogram_io.load_histogram` and `spe_io.load_spe`'s
convention.

## Integration into the existing loading pipeline

- `main_window._try_load_spectrum(path)` gains a third dispatch branch:
  `.spk` → `spk_io.load_spk`.
- `_open_file_dialog`'s filter string grows to include `*.spk`:
  `"Spectrum files (*.txt *.spe *.spk);;Text files (*.txt);;SPE files
  (*.spe);;SPK files (*.spk);;All files (*)"`.
- Both `.spk` variants always yield exactly one channel array per file —
  unlike gf3's `HHIRFSPK` container (which can hold up to 254 named
  spectra), there is no multi-spectrum-per-file case to handle. No
  changes needed anywhere else in the pipeline.

## Testing

Following this project's established pattern: `pytest` unit tests for the
pure-parsing module (`spk_io.py`), same style as `tests/test_spe_io.py`.

- `demo.spk` and `pg_25um_f.spk` copied into `tests/fixtures/`:
  - `demo.spk`: assert 4096 channels; cross-check plausibility via a
    visual render (same secondary verification used for `eu.spe` in the
    `.spe` sub-project) since there's no independent ground-truth export
    to compare exact values against.
  - `pg_25um_f.spk`: assert 16384 channels; spot-check specific channel
    values via an **independent** one-line `struct.unpack` directly on
    known byte offsets within the test itself (a real independent check,
    since this format is uncompressed — unlike the LC case).
- Hand-built synthetic byte sequences (via `struct`, same style as
  `test_spe_io.py`) exercising: LC1's and LC2's distinct tag branches
  (3-value, 2-value, 1-value, and LC2's run-length "same" tag), the
  oldmat trailer parser (with a small `le4` or `lf4` synthetic array),
  and error cases (bad/missing magic and trailer, truncated data,
  mismatched declared vs. actual size, `levels`/`lines != 1` rejection).

## Out of scope

- gf3's `HHIRFSPK` multi-spectrum container format (the actual subject of
  the research doc's original, now-corrected, `.spk` section).
- `lf8`/`hf8` (float64): the reference source's `lf8_get`/`hf8_get`
  (`oldmat_getput.c`) compute the file offset with the same `fpos(4)`
  macro used for 4-byte types instead of an 8-byte-scaled one — an
  apparent bug in the reference implementation itself. Porting it would
  mean either faithfully reproducing a bug or silently "fixing" behavior
  with no real file to verify against either way. Excluded; revisit only
  if a real float64 `.spk` file surfaces.
- `le4t`/`he4t`/`le2t`/`he2t` ("triangular" matrix variants) and the
  `MAT_MATE`/`MAT_TRIXI`/`MAT_GF2`/`MAT_HGF2`/`MAT_SHM` formats: all
  inherently 2-D/matrix-shaped (gamma-gamma coincidence data), not 1-D
  spectra: no fit with this app's data model, and no sample files
  requiring them.
- VAX floating point (`vaxf`/`vaxg`): defined in the format table but
  commented out/never compiled into the reference `mat_types.c` itself —
  genuinely dead code upstream, not just unused by us.
- The oldmat heuristic/statistical "guess the format" fallback
  (`guessdatatype` in `oldmat_minfo.c`, used only when no `MatFmt`
  trailer is present): inherently probabilistic, not deterministic;
  excluded since both real sample files carry proper explicit signatures
  and a wrong guess would silently corrupt data rather than fail loudly.
- True 2-D matrices (`levels > 1` or `lines > 1` in either variant):
  rejected with `ParseError` rather than exposed partially — this app
  has no 2-D-matrix UI or data model.
- Writing `.spk` files (only reading is needed).
