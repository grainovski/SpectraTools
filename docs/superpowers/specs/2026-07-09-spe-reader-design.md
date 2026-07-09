# .spe Reader (Radware/gf3 native format) — Design Spec

Date: 2026-07-09

## Purpose

Add support for loading gamma-ray spectra from gf3's (RadWare) native
binary `.spe` format, alongside the existing ASCII `.txt` format. First of
three sub-projects (`.spe` reader → `.spk` reader → peak fitting) building
toward the "PeakFinderFitting" project's next phase. See
`docs/research/2026-07-09-peak-finding-fitting-and-spe-spk-formats.md`
for the full format research this is based on.

## Format (empirically confirmed against `eu.spe`)

Fortran-unformatted sequential binary file, two records, each bracketed
by a leading and trailing 4-byte record-length marker:

```
Record 1 (24-byte payload):
  [int32 marker = 24]
  name       char[8]   space-padded, not null-terminated
  idim1      int32     channel count
  idim2      int32     always 1 for 1-D spectra
  ired1      int32     always 1
  ired2      int32     always 1
  [int32 marker = 24]  (repeats the leading marker)

Record 2 (idim1*4-byte payload):
  [int32 marker = idim1*4]
  channel data: idim1 x float32, one 4-byte word per channel
  [int32 marker = idim1*4]  (repeats the leading marker)
```

**Endianness:** no explicit field. Detect by reading the first int32
as-is; if it isn't a small sane value (expect exactly 24), byte-swap and
retry. Once determined, apply consistently to both header ints and every
channel float in record 2. `eu.spe` itself is big-endian.

## Module: `spe_io.py`

New module, mirroring `histogram_io.py`'s shape:

```python
def load_spe(path: str) -> np.ndarray
```

- Reads and validates both record markers (leading == trailing == 24 for
  record 1, leading == trailing == idim1*4 for record 2), raising
  `ParseError` (reusing `histogram_io.ParseError` — no new exception type)
  with a clear message if a marker is missing/mismatched/the file is too
  short.
- Returns a plain `numpy` array of the channel counts (as-is length,
  `idim1` channels — **no power-of-two bucketing**; that rule is specific
  to the ASCII format's convention of inferring a truncated channel count,
  and doesn't apply here since `.spe` always states its exact channel
  count explicitly).
- `idim2`/`ired1`/`ired2` are read (they're part of the fixed 24-byte
  record-1 layout regardless) but not validated or used beyond that —
  this app has no concept of 2-D spectra, and there's no evidence in the
  research that real 1-D `.spe` files ever set `idim2` to anything but 1,
  so there's nothing concrete to defend against here.
- The `name` field is read but not surfaced anywhere yet (no UI currently
  shows a per-spectrum display name distinct from the filename) — parsed
  and discarded, not an error to leave unused.

## Integration into the existing loading pipeline

- `main_window._try_load_spectrum(path)` dispatches on file extension:
  `.txt` → `histogram_io.load_histogram`, `.spe` → `spe_io.load_spe`.
  Anything else falls back to `.txt` handling (preserves today's behavior
  for extensionless/other files) — actually: to avoid silently
  mis-parsing an unrelated file as ASCII, unrecognized extensions should
  still be attempted as `.txt` (today's only-ever behavior), since that's
  a no-worse-than-before default and this isn't the place to add a broader
  "unknown format" error path.
- `_open_file_dialog`'s filter string grows to
  `"Spectrum files (*.txt *.spe);;Text files (*.txt);;SPE files (*.spe);;All files (*)"`.
- No changes needed anywhere else — `LoadedSpectrum`, the spectrum list
  panel, overlay plotting, zoom, Show/Active, all already operate on the
  resulting `numpy` array with no assumptions about source format.

## Testing

Following this project's established pattern: `pytest` unit tests for the
pure-parsing module (`spe_io.py`), same style as `tests/test_histogram_io.py`.

- Parse `eu.spe` (copied into `tests/fixtures/`): assert 4096 channels,
  first 5 values `[4, 0, 1, 0, 1]` — this is the empirically-verified
  known-good case (matches `test.txt`'s values exactly, confirmed by
  manual byte-level inspection during design).
- Synthetic little-endian fixture (built in a test via `tmp_path`,
  constructing the byte layout directly with `struct`): confirms the
  endianness auto-detection branch actually gets exercised, not just the
  big-endian path `eu.spe` happens to hit.
- Malformed file (truncated / marker mismatch): asserts `ParseError`.

## Out of scope

- The *other* `.spe` variant found in `tv`/`libmfile` (36-byte header, no
  Fortran record brackets) — a genuinely different, incompatible format
  that happens to share the same file extension. Not implemented; if a
  need for it ever arises, it'd need its own explicit format-detection
  path (e.g. try gf3-style first, fall back to gf2-style, or make it
  user-selectable) since both variants start with a 4-byte value of `24`
  and can't be told apart by magic number alone.
- Surfacing the `.spe` file's internal `name` field anywhere in the UI.
- The ORTEC/Maestro ASCII `.SPE` format (not implemented in either
  reference codebase; a different request entirely if ever needed).
