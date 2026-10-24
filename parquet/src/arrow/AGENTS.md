# Native decoder ownership

- Keep decoder memory admission opt-in and stock reader defaults unchanged. Admit row-group predicate metadata, current page backing/scratch, and selected output batches before their corresponding allocations.
- Page backing must follow immutable Bytes and view arrays; dictionary/value scratch follows its decoder. Output charges must follow nested and projected Arrow buffers after reader EOF. EOF itself must not require another reservation.
- Admit variable byte growth using native decoded lengths before reserve/copy. Do not estimate selected payload from row averages or writer page targets. Nested fixed cells may conservatively use physical leaf value counts; logical row counts alone cannot bound list/map children.
