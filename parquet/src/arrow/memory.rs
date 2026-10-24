// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Optional fallible memory admission for the push decoder.
//!
//! Page backing, decoder scratch, and emitted batches have independent owners.
//! Implementations must reserve before returning and release through RAII.
use crate::{
    arrow::{
        ProjectionMask,
        arrow_reader::{RowSelection, RowSelector},
    },
    basic::{Encoding, Type},
    errors::{ParquetError, Result},
    file::metadata::ParquetMetaData,
};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use std::{collections::VecDeque, fmt::Debug, ops::Range, sync::Arc};

/// An allocation charged until its last owner drops.
pub trait Allocation: Debug + Send + Sync {
    /// Admit additional bytes before native buffer growth. On failure, no
    /// corresponding allocation or mutation has occurred.
    fn grow(&self, bytes: usize) -> Result<()>;
    /// Release bytes after their allocation has been replaced or freed.
    fn shrink(&self, bytes: usize);
}
/// Owns one emitted batch's cells and variable payload.
pub trait BatchOwner: Allocation {
    /// Attach the admitted charge to all returned backing allocations, including
    /// nested children and validity buffers. Page backing is already page-owned.
    fn retain(&self, batch: RecordBatch) -> Result<RecordBatch>;
}
/// Admission result. The decoder validates the selected row count.
#[derive(Debug)]
pub struct BatchReservation {
    /// Maximum selected rows admitted for this batch.
    pub rows: usize,
    /// Owner of the admitted buffers.
    pub owner: Arc<dyn BatchOwner>,
    /// Batch-local levels/read padding, released with the reader rather than output.
    pub scratch: Arc<dyn Allocation>,
}
/// Independent decompressed backing and decoder scratch ownership.
#[derive(Debug)]
pub struct PageReservation {
    /// Moves with immutable page Bytes, including view-array references.
    pub backing: Arc<dyn Allocation>,
    /// Moves with dictionary/value decoders, not ordinary page backing.
    pub scratch: Arc<dyn Allocation>,
}
/// Native row-group selection and predicate bookkeeping, without decoded data.
#[derive(Debug)]
pub struct RowGroupMemory<'a> {
    /// Frozen file metadata.
    pub metadata: &'a ParquetMetaData,
    /// Physical row-group index.
    pub row_group: usize,
    /// Whether predicates will be evaluated.
    pub predicates: bool,
    /// Exact incoming selection entries, before native reader copies.
    pub selectors: usize,
}
/// A validated page about to be decompressed and decoded.
#[derive(Debug)]
pub struct PageMemory {
    /// Physical row group.
    pub row_group: usize,
    /// Physical leaf column.
    pub column: usize,
    /// Physical Parquet value type.
    pub physical_type: Type,
    /// Native value encoding.
    pub encoding: Encoding,
    /// Validated uncompressed page bytes.
    pub uncompressed_bytes: usize,
    /// Number of dictionary values or data levels.
    pub num_values: usize,
    /// Whether this is a dictionary page.
    pub dictionary: bool,
    /// Whether decompression allocates new page backing.
    pub compressed: bool,
}
/// A native selected output or predicate batch about to allocate cells.
#[derive(Debug)]
pub struct BatchMemory<'a> {
    /// Frozen file metadata, including optional per-page size statistics.
    pub metadata: &'a ParquetMetaData,
    /// Physical row-group index.
    pub row_group: usize,
    /// Actual native output/predicate projection.
    pub projection: &'a ProjectionMask,
    /// Actual physical Arrow schema.
    pub schema: &'a Schema,
    /// Whether this batch is used to evaluate a predicate.
    pub predicate: bool,
    /// Largest selected batch requested. Admission may choose fewer rows.
    pub rows: usize,
    selection: &'a VecDeque<RowSelector>,
    position: usize,
}
impl BatchMemory<'_> {
    /// Exact physical row ranges for the next `rows` selected records. No
    /// average-byte estimate or independent predicate evaluator is involved.
    pub fn ranges(&self, rows: usize) -> Result<Vec<Range<usize>>> {
        if rows == 0 || rows > self.rows {
            return Err(invalid());
        }
        let mut left = rows;
        let mut position = self.position;
        let mut ranges = Vec::new();
        for selector in self.selection {
            if selector.skip {
                position = position
                    .checked_add(selector.row_count)
                    .ok_or_else(invalid)?;
                continue;
            }
            let count = left.min(selector.row_count);
            if count != 0 {
                ranges.push(position..position.checked_add(count).ok_or_else(invalid)?);
            }
            left -= count;
            if left == 0 {
                break;
            }
            position = position
                .checked_add(selector.row_count)
                .ok_or_else(invalid)?;
        }
        if left != 0 {
            return Err(invalid());
        }
        Ok(ranges)
    }
}
/// Measured work for governed fixed-cell cache reuse.
#[derive(Debug, Default, Clone, Copy)]
pub struct CacheObservation {
    /// Rows decoded from Parquet into cache backing.
    pub decoded: usize,
    /// Selected output rows served from cache.
    pub reused: usize,
    /// Native inner-reader decoding time.
    pub decode_nanos: u64,
    /// Wrapper bookkeeping and selected-array materialization time.
    pub overhead_nanos: u64,
}

/// Pre-admitted cache capacity. Retention transfers backing charges to each
/// immutable array so eviction/EOF cannot release externally retained buffers.
pub trait CacheOwner: Debug + Send + Sync {
    /// Record batch-level work without retaining per-batch metadata.
    fn observe(&self, _observation: CacheObservation) {}
    /// Transfer already-admitted backing capacity to this newly decoded batch.
    fn retain(&self, batch: RecordBatch) -> Result<RecordBatch>;
}

/// Whole-row-group predicate cache admission, before cache readers allocate.
#[derive(Debug)]
pub struct CacheMemory<'a> {
    /// Frozen file metadata.
    pub metadata: &'a ParquetMetaData,
    /// Physical row-group index.
    pub row_group: usize,
    /// Columns reused by predicates and output.
    pub projection: &'a ProjectionMask,
    /// Physical Arrow schema.
    pub schema: &'a Schema,
    /// Native cache alignment.
    pub batch_size: usize,
    /// Maximum admitted cache bytes.
    pub limit: usize,
}

/// Opt-in admission. Stock readers do not construct or call this interface.
pub trait DecoderMemory: Debug + Send + Sync {
    /// None declines caching without failing the required decode. The owner
    /// must cover full cache batches, metadata, and materialization scratch.
    fn cache(&self, _request: &CacheMemory<'_>) -> Result<Option<Arc<dyn CacheOwner>>> {
        Ok(None)
    }

    /// Admit native predicate masks/selection metadata through row-group EOF.
    fn row_group(&self, request: &RowGroupMemory<'_>) -> Result<Arc<dyn Allocation>>;
    /// Admit decompressed backing, dictionary cells, and native page scratch.
    fn page(&self, request: &PageMemory) -> Result<PageReservation>;
    /// Admit selected batch cells before array-reader construction/polling.
    /// Variable payload calls `BatchOwner::grow` using actual native lengths.
    fn batch(&self, request: &BatchMemory<'_>) -> Result<BatchReservation>;
}
fn invalid() -> ParquetError {
    ParquetError::General("Invalid decoder memory admission".into())
}

#[derive(Debug, Clone)]
pub(crate) struct Context {
    pub memory: Arc<dyn DecoderMemory>,
    pub metadata: Arc<ParquetMetaData>,
    pub schema: Arc<Schema>,
    pub projection: ProjectionMask,
    pub row_group: usize,
    pub predicate: bool,
    pub _scope: Arc<dyn Allocation>,
}
#[derive(Debug)]
pub(crate) struct BatchState {
    pub context: Context,
    selection: VecDeque<RowSelector>,
    position: usize,
    pending: Option<BatchReservation>,
    pub scratch: Option<Arc<dyn Allocation>>,
}
impl BatchState {
    pub fn new(context: Context, selection: Option<&RowSelection>, rows: usize) -> Self {
        let selection = selection
            .map(|s| s.iter().copied().collect())
            .unwrap_or_else(|| VecDeque::from([RowSelector::select(rows)]));
        Self {
            context,
            selection,
            position: 0,
            pending: None,
            scratch: None,
        }
    }
    pub fn remaining(&self) -> usize {
        self.selection
            .iter()
            .filter(|s| !s.skip)
            .map(|s| s.row_count)
            .sum()
    }
    pub fn prepare(&mut self, max: usize) -> Result<usize> {
        let max = max.min(self.remaining());
        if max == 0 {
            return Err(invalid());
        }
        let request = BatchMemory {
            metadata: &self.context.metadata,
            row_group: self.context.row_group,
            projection: &self.context.projection,
            schema: &self.context.schema,
            predicate: self.context.predicate,
            rows: max,
            selection: &self.selection,
            position: self.position,
        };
        let reservation = self.context.memory.batch(&request)?;
        if reservation.rows == 0 || reservation.rows > max {
            return Err(invalid());
        }
        let rows = reservation.rows;
        self.pending = Some(reservation);
        Ok(rows)
    }
    pub fn next(&mut self, max: usize) -> Result<BatchReservation> {
        if self.pending.is_none() {
            self.prepare(max)?;
        }
        self.pending.take().ok_or_else(invalid)
    }
    pub fn advance(&mut self, mut rows: usize) {
        while rows != 0 {
            let Some(mut front) = self.selection.pop_front() else {
                break;
            };
            if front.skip {
                self.position += front.row_count;
                continue;
            }
            let count = rows.min(front.row_count);
            self.position += count;
            rows -= count;
            front.row_count -= count;
            if front.row_count != 0 {
                self.selection.push_front(front);
            }
        }
    }
}

/// Reserve a native byte-vector growth before allocating. Keeping the old
/// capacity charged as well covers the temporary reallocation overlap.
pub(crate) fn reserve(
    bytes: &mut Vec<u8>,
    additional: usize,
    memory: Option<&dyn Allocation>,
) -> Result<()> {
    let wanted = bytes.len().checked_add(additional).ok_or_else(invalid)?;
    if wanted <= bytes.capacity() {
        return Ok(());
    }
    if let Some(memory) = memory {
        // Geometric growth avoids one reallocation per selected value. The
        // whole new allocation is admitted while the old allocation is live.
        let capacity = wanted.max(bytes.capacity().saturating_mul(2)).max(8);
        memory.grow(capacity)?;
        let previous = bytes.capacity();
        bytes.reserve_exact(capacity - bytes.len());
        memory.shrink(previous);
    } else {
        bytes.reserve(additional);
    }
    Ok(())
}

pub(crate) fn copied_bytes(data: &arrow_data::ArrayData) -> usize {
    match data.data_type() {
        arrow_schema::DataType::Utf8
        | arrow_schema::DataType::LargeUtf8
        | arrow_schema::DataType::Binary
        | arrow_schema::DataType::LargeBinary => {
            data.buffers().get(1).map_or(0, |buffer| buffer.len())
        }
        arrow_schema::DataType::Dictionary(_, _)
        | arrow_schema::DataType::Utf8View
        | arrow_schema::DataType::BinaryView => 0,
        _ => data.child_data().iter().map(copied_bytes).sum(),
    }
}

pub(crate) fn retain_array(
    array: arrow_array::ArrayRef,
    memory: Arc<dyn Allocation>,
) -> Result<arrow_array::ArrayRef> {
    fn retain(
        data: arrow_data::ArrayData,
        memory: &Arc<dyn Allocation>,
    ) -> Result<arrow_data::ArrayData> {
        fn buffer(
            buffer: &arrow_buffer::Buffer,
            memory: &Arc<dyn Allocation>,
        ) -> Result<arrow_buffer::Buffer> {
            struct Owner {
                _buffer: arrow_buffer::Buffer,
                _memory: std::sync::Mutex<Arc<dyn Allocation>>,
            }
            let pointer = std::ptr::NonNull::new(buffer.as_ptr().cast_mut()).ok_or_else(invalid)?;
            let owner = Arc::new(Owner {
                _buffer: buffer.clone(),
                _memory: std::sync::Mutex::new(memory.clone()),
            });
            // SAFETY: the owner retains the identical immutable allocation for
            // the full pointer/length lifetime, including nested projections.
            Ok(unsafe {
                arrow_buffer::Buffer::from_custom_allocation(pointer, buffer.len(), owner)
            })
        }
        let buffers = data
            .buffers()
            .iter()
            .map(|value| buffer(value, memory))
            .collect::<Result<Vec<_>>>()?;
        let children = data
            .child_data()
            .iter()
            .map(|value| retain(value.clone(), memory))
            .collect::<Result<Vec<_>>>()?;
        let nulls = data
            .nulls()
            .map(|value| {
                buffer(value.buffer(), memory).map(|buffer| {
                    arrow_buffer::NullBuffer::new(arrow_buffer::BooleanBuffer::new(
                        buffer,
                        value.offset(),
                        value.len(),
                    ))
                })
            })
            .transpose()?;
        Ok(data
            .into_builder()
            .buffers(buffers)
            .child_data(children)
            .nulls(nulls)
            .build()?)
    }
    Ok(arrow_array::make_array(retain(array.to_data(), &memory)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[derive(Debug)]
    struct Limit(AtomicUsize);
    impl Allocation for Limit {
        fn grow(&self, bytes: usize) -> Result<()> {
            self.0
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(bytes))
                .map(|_| ())
                .map_err(|_| invalid())
        }
        fn shrink(&self, bytes: usize) {
            self.0.fetch_add(bytes, Ordering::SeqCst);
        }
    }
    // Law: a refused allocation leaves bytes, length, and capacity unchanged;
    // geometric boundaries and skewed values use actual bytes, never averages.
    #[test]
    fn admission_precedes_variable_buffer_growth() {
        for power in 0..16 {
            for extra in [0, 1, 3, 127] {
                let size = (1usize << power) + extra;
                let mut bytes = vec![7u8; size];
                let before = (bytes.len(), bytes.capacity(), bytes.clone());
                let limit = Limit(AtomicUsize::new(0));
                assert!(reserve(&mut bytes, size + 1, Some(&limit)).is_err());
                assert_eq!((bytes.len(), bytes.capacity(), bytes), before);
            }
        }
    }
    use crate::{
        DecodeResult,
        arrow::{
            ArrowWriter,
            arrow_reader::{
                ArrowPredicateFn, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowFilter,
            },
            push_decoder::{ParquetPushDecoder, ParquetPushDecoderBuilder},
        },
        basic::{Compression, ZstdLevel},
        file::properties::{WriterProperties, WriterVersion},
    };
    use arrow_array::{Array, ArrayRef, Int64Array, StringArray, StringViewArray};
    use std::sync::Mutex;
    #[derive(Debug)]
    struct Pool {
        limit: AtomicUsize,
        used: AtomicUsize,
        peak: AtomicUsize,
        batches: AtomicUsize,
        ranges: Mutex<Vec<Range<usize>>>,
    }
    #[derive(Debug)]
    struct Charge {
        pool: Arc<Pool>,
        bytes: AtomicUsize,
    }
    impl Drop for Charge {
        fn drop(&mut self) {
            self.pool
                .used
                .fetch_sub(self.bytes.load(Ordering::SeqCst), Ordering::SeqCst);
        }
    }
    #[derive(Clone, Debug)]
    struct Lease(Arc<Charge>);
    impl Allocation for Lease {
        fn grow(&self, bytes: usize) -> Result<()> {
            let next = self
                .0
                .pool
                .used
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                    used.checked_add(bytes)
                        .filter(|next| *next <= self.0.pool.limit.load(Ordering::SeqCst))
                })
                .map_err(|_| ParquetError::General("governed memory refused".into()))?
                + bytes;
            self.0.pool.peak.fetch_max(next, Ordering::SeqCst);
            self.0.bytes.fetch_add(bytes, Ordering::SeqCst);
            Ok(())
        }
        fn shrink(&self, bytes: usize) {
            assert!(self.0.bytes.fetch_sub(bytes, Ordering::SeqCst) >= bytes);
            self.0.pool.used.fetch_sub(bytes, Ordering::SeqCst);
        }
    }
    impl BatchOwner for Lease {
        fn retain(&self, batch: RecordBatch) -> Result<RecordBatch> {
            let columns = batch
                .columns()
                .iter()
                .map(|array| retain_array(array.clone(), Arc::new(self.clone())))
                .collect::<Result<Vec<_>>>()?;
            Ok(RecordBatch::try_new(batch.schema(), columns)?)
        }
    }
    impl Pool {
        fn new(limit: usize) -> Arc<Self> {
            Arc::new(Self {
                limit: AtomicUsize::new(limit),
                used: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                batches: AtomicUsize::new(0),
                ranges: Mutex::new(vec![]),
            })
        }
        fn lease(self: &Arc<Self>, bytes: usize) -> Result<Lease> {
            let lease = Lease(Arc::new(Charge {
                pool: self.clone(),
                bytes: AtomicUsize::new(0),
            }));
            lease.grow(bytes)?;
            Ok(lease)
        }
    }
    #[derive(Debug)]
    struct Governor(Arc<Pool>);
    impl DecoderMemory for Governor {
        fn row_group(&self, request: &RowGroupMemory<'_>) -> Result<Arc<dyn Allocation>> {
            Ok(Arc::new(self.0.lease(
                request.metadata.row_group(request.row_group).num_rows() as usize * 64 + 4096,
            )?))
        }
        fn page(&self, request: &PageMemory) -> Result<PageReservation> {
            Ok(PageReservation {
                backing: Arc::new(self.0.lease(if request.compressed {
                    request.uncompressed_bytes
                } else {
                    0
                })?),
                scratch: Arc::new(self.0.lease(request.num_values * 16 + 4096)?),
            })
        }
        fn batch(&self, request: &BatchMemory<'_>) -> Result<BatchReservation> {
            self.0.batches.fetch_add(1, Ordering::SeqCst);
            let rows = request.rows.min(32);
            self.0.ranges.lock().unwrap().extend(request.ranges(rows)?);
            Ok(BatchReservation {
                rows,
                scratch: Arc::new(self.0.lease(8192)?),
                owner: Arc::new(
                    self.0.lease(
                        request
                            .metadata
                            .row_group(request.row_group)
                            .columns()
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| request.projection.leaf_included(*index))
                            .map(|(_, column)| column.num_values() as usize * 32)
                            .sum::<usize>()
                            + 8192,
                    )?,
                ),
            })
        }
    }
    fn file(
        encoding: Encoding,
        dictionary: bool,
        rows: usize,
        width: usize,
    ) -> (bytes::Bytes, Vec<String>) {
        let values = (0..rows)
            .map(|row| format!("{row:08}-{}", "x".repeat(width)))
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_from_iter(vec![
            (
                "id",
                Arc::new(Int64Array::from_iter_values(
                    (0..rows).map(|row| row as i64),
                )) as ArrayRef,
            ),
            (
                "value",
                Arc::new(StringArray::from(values.clone())) as ArrayRef,
            ),
        ])
        .unwrap();
        let properties = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_column_encoding(crate::schema::types::ColumnPath::from("value"), encoding)
            .set_dictionary_enabled(dictionary)
            .set_data_page_size_limit(64 * 1024)
            .set_dictionary_page_size_limit(64 * 1024)
            .set_max_row_group_row_count(Some(rows))
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::new(), batch.schema(), Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        (bytes::Bytes::from(writer.into_inner().unwrap()), values)
    }
    fn decoder(
        bytes: &bytes::Bytes,
        pool: Arc<Pool>,
        view: bool,
        selection: bool,
        predicate: bool,
    ) -> ParquetPushDecoder {
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes.clone()).unwrap();
        let mut builder = ParquetPushDecoderBuilder::try_new_decoder(reader.metadata().clone())
            .unwrap()
            .with_batch_size(512)
            .with_decoder_memory(Arc::new(Governor(pool.clone())));
        if view {
            let schema = Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false),
                arrow_schema::Field::new("value", arrow_schema::DataType::Utf8View, false),
            ]));
            builder = ParquetPushDecoderBuilder::try_new_decoder_with_options(
                reader.metadata().clone(),
                ArrowReaderOptions::new().with_schema(schema),
            )
            .unwrap()
            .with_batch_size(512)
            .with_decoder_memory(Arc::new(Governor(pool)));
        }
        let rows = reader.metadata().file_metadata().num_rows() as usize;
        if selection {
            builder = builder.with_row_selection(RowSelection::from(vec![
                RowSelector::skip(1),
                RowSelector::select(1),
                RowSelector::skip(rows - 3),
                RowSelector::select(1),
            ]));
        }
        if predicate {
            let projection = ProjectionMask::leaves(reader.parquet_schema(), [0]);
            let predicate = ArrowPredicateFn::new(projection, move |batch| {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                Ok(arrow_array::BooleanArray::from(
                    ids.values()
                        .iter()
                        .map(|id| *id == 1 || *id == rows as i64 - 1)
                        .collect::<Vec<_>>(),
                ))
            });
            builder = builder.with_row_filter(RowFilter::new(vec![Box::new(predicate)]));
        }
        builder.build().unwrap()
    }
    fn next_reader(
        decoder: &mut ParquetPushDecoder,
        bytes: &bytes::Bytes,
    ) -> Result<Option<crate::arrow::arrow_reader::ParquetRecordBatchReader>> {
        loop {
            match decoder.try_next_reader()? {
                DecodeResult::Data(reader) => return Ok(Some(reader)),
                DecodeResult::Finished => return Ok(None),
                DecodeResult::NeedsData(ranges) => {
                    let data = ranges
                        .iter()
                        .map(|range| bytes.slice(range.start as usize..range.end as usize))
                        .collect();
                    decoder.push_ranges(ranges, data)?;
                }
            }
        }
    }
    // Real-boundary law: exact native selected values survive every byte-array
    // encoding and view representation, while old unrelated rows never consume
    // output payload allowance. Stock native SQL/reader is the value oracle.
    #[test]
    fn native_selected_batches_pages_and_eof_keep_independent_owners() {
        for encoding in [
            Encoding::PLAIN,
            Encoding::DELTA_LENGTH_BYTE_ARRAY,
            Encoding::DELTA_BYTE_ARRAY,
        ] {
            for dictionary in [false, true] {
                for view in [false, true] {
                    let (bytes, values) = file(encoding, dictionary, 1024, 8192);
                    let pool = Pool::new(4 * 1024 * 1024);
                    let mut decoder = decoder(&bytes, pool.clone(), view, false, true);
                    let mut reader = next_reader(&mut decoder, &bytes).unwrap().unwrap();
                    let batch = reader.next().unwrap().unwrap();
                    let output = batch.column(1).clone();
                    let actual = if view {
                        output
                            .as_any()
                            .downcast_ref::<StringViewArray>()
                            .unwrap()
                            .iter()
                            .map(|value| value.unwrap().to_owned())
                            .collect::<Vec<_>>()
                    } else {
                        output
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .map(|value| value.unwrap().to_owned())
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(actual, [values[1].clone(), values[1023].clone()]);
                    drop(batch);
                    let calls = pool.batches.load(Ordering::SeqCst);
                    pool.limit
                        .store(pool.used.load(Ordering::SeqCst), Ordering::SeqCst);
                    assert!(reader.next().is_none(), "EOF must not request memory");
                    assert_eq!(pool.batches.load(Ordering::SeqCst), calls);
                    drop(reader);
                    drop(decoder);
                    assert!(
                        pool.used.load(Ordering::SeqCst) > 0,
                        "projected backing must retain its owner"
                    );
                    drop(output);
                    assert_eq!(pool.used.load(Ordering::SeqCst), 0);
                    assert!(pool.peak.load(Ordering::SeqCst) < 4 * 1024 * 1024);
                }
            }
        }
    }

    // A downstream queue may retain the final batch without polling again.
    // Completed native readers must release their real scratch first; only
    // the projected allocation owners may survive that backpressure boundary.
    #[test]
    fn final_queued_batch_releases_reader_scratch_before_another_poll() {
        for dictionary in [false, true] {
            for view in [false, true] {
                let (bytes, values) = file(Encoding::PLAIN, dictionary, 1024, 8192);
                let pool = Pool::new(4 * 1024 * 1024);
                let mut decoder = decoder(&bytes, pool.clone(), view, true, false);
                let batch = loop {
                    match decoder.try_decode().unwrap() {
                        DecodeResult::Data(batch) => break batch,
                        DecodeResult::NeedsData(ranges) => {
                            let data = ranges
                                .iter()
                                .map(|range| bytes.slice(range.start as usize..range.end as usize))
                                .collect();
                            decoder.push_ranges(ranges, data).unwrap();
                        }
                        DecodeResult::Finished => panic!("selected values disappeared"),
                    }
                };
                let output = batch.column(1).clone();
                assert_eq!(batch.num_rows(), 2);
                let actual = if view {
                    output
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap()
                        .iter()
                        .map(|value| value.unwrap().to_owned())
                        .collect::<Vec<_>>()
                } else {
                    output
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .iter()
                        .map(|value| value.unwrap().to_owned())
                        .collect::<Vec<_>>()
                };
                assert_eq!(actual, [values[1].clone(), values[1023].clone()]);
                drop(batch);
                let queued = pool.used.load(Ordering::SeqCst);
                assert!(queued > 0, "projected arrays must retain allocation owners");
                drop(decoder);
                assert_eq!(
                    queued,
                    pool.used.load(Ordering::SeqCst),
                    "a queued final batch must not retain reader-local scratch"
                );
                drop(output);
                assert_eq!(pool.used.load(Ordering::SeqCst), 0);
            }
        }
    }

    // Incoming page selections allocate native cursor copies even without a
    // RowFilter. Refusal must precede those copies and all batch/page decoding.
    #[test]
    fn incoming_selectors_are_admitted_without_predicates() {
        #[derive(Debug)]
        struct Selection {
            pool: Arc<Pool>,
            count: AtomicUsize,
        }
        impl DecoderMemory for Selection {
            fn row_group(&self, request: &RowGroupMemory<'_>) -> Result<Arc<dyn Allocation>> {
                assert!(!request.predicates);
                self.count.store(request.selectors, Ordering::SeqCst);
                Ok(Arc::new(self.pool.lease(
                    request.selectors * std::mem::size_of::<RowSelector>() * 2,
                )?))
            }
            fn page(&self, _: &PageMemory) -> Result<PageReservation> {
                panic!("selection refusal must precede page decode")
            }
            fn batch(&self, _: &BatchMemory<'_>) -> Result<BatchReservation> {
                panic!("selection refusal must precede batch construction")
            }
        }
        let (bytes, _) = file(Encoding::PLAIN, false, 4096, 0);
        let metadata = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .metadata()
            .clone();
        for count in [1, 17, 129, 1025] {
            let selectors = (0..count)
                .flat_map(|_| [RowSelector::select(1), RowSelector::skip(1)])
                .collect::<Vec<_>>();
            let requested = selectors.len() * std::mem::size_of::<RowSelector>() * 2;
            let pool = Pool::new(requested - 1);
            let memory = Arc::new(Selection {
                pool: pool.clone(),
                count: AtomicUsize::new(0),
            });
            let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata.clone())
                .unwrap()
                .with_row_selection(RowSelection::from(selectors))
                .with_decoder_memory(memory.clone())
                .build()
                .unwrap();
            let error = decoder.try_decode().unwrap_err();
            assert!(error.to_string().contains("governed memory refused"));
            assert_eq!(memory.count.load(Ordering::SeqCst), count * 2);
            drop(decoder);
            assert_eq!(pool.used.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn native_refuses_oversized_pages_and_releases_on_cancel() {
        let (bytes, _) = file(Encoding::PLAIN, false, 2, 1024 * 1024);
        let pool = Pool::new(64 * 1024);
        let mut decoder = decoder(&bytes, pool.clone(), false, false, false);
        // Reader construction admits cells; the huge required page is refused
        // before decompression, rather than silently exceeding the pool.
        match next_reader(&mut decoder, &bytes) {
            Ok(Some(mut reader)) => assert!(reader.next().unwrap().is_err()),
            Err(_) => (),
            Ok(None) => panic!("selected row disappeared"),
        }
        drop(decoder);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn native_full_scan_matches_stock_without_retaining_row_group_output() {
        let (bytes, expected) = file(Encoding::DELTA_BYTE_ARRAY, false, 4096, 4096);
        let stock = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .with_batch_size(127)
            .build()
            .unwrap()
            .flat_map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.unwrap().to_owned())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(stock, expected);
        let pool = Pool::new(4 * 1024 * 1024);
        let mut decoder = decoder(&bytes, pool.clone(), false, false, false);
        let mut actual = Vec::new();
        while let Some(mut reader) = next_reader(&mut decoder, &bytes).unwrap() {
            for batch in &mut reader {
                let batch = batch.unwrap();
                actual.extend(
                    batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .iter()
                        .map(|value| value.unwrap().to_owned()),
                );
            }
        }
        drop(decoder);
        assert_eq!(actual, stock);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn native_nested_projection_retains_children_and_rejects_large_numeric_lists_before_decode() {
        use arrow_array::{
            ListArray,
            builder::{ListBuilder, StringBuilder},
        };
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in 0..4 {
            for value in 0..row * 3 {
                builder
                    .values()
                    .append_value(format!("{row}-{value}-{}", "x".repeat(1024)));
            }
            builder.append(row != 1);
        }
        let batch =
            RecordBatch::try_from_iter(vec![("nested", Arc::new(builder.finish()) as ArrayRef)])
                .unwrap();
        let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        let bytes = bytes::Bytes::from(writer.into_inner().unwrap());
        let metadata = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let pool = Pool::new(1024 * 1024);
        let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)
            .unwrap()
            .with_decoder_memory(Arc::new(Governor(pool.clone())))
            .build()
            .unwrap();
        let mut reader = next_reader(&mut decoder, &bytes).unwrap().unwrap();
        let output = reader.next().unwrap().unwrap();
        assert_eq!(output, batch);
        let child = output
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .clone();
        drop(output);
        assert!(reader.next().is_none());
        drop(reader);
        drop(decoder);
        assert!(pool.used.load(Ordering::SeqCst) > 0);
        drop(child);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);

        let list = ListArray::from_iter_primitive::<arrow_array::types::Int64Type, _, _>([Some(
            (0..100_000).map(Some),
        )]);
        let batch =
            RecordBatch::try_from_iter(vec![("nested", Arc::new(list) as ArrayRef)]).unwrap();
        let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        let bytes = bytes::Bytes::from(writer.into_inner().unwrap());
        let metadata = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let pool = Pool::new(128 * 1024);
        let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)
            .unwrap()
            .with_decoder_memory(Arc::new(Governor(pool.clone())))
            .build()
            .unwrap();
        assert!(next_reader(&mut decoder, &bytes).is_err());
        drop(decoder);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);
    }

    // Parquet null fixed-list entries carry definition levels, not every padded
    // Arrow child slot. Admission must multiply the physical entry count by
    // fixed-size multiplicity even beneath a repeated outer list.
    #[test]
    fn repeated_null_fixed_lists_expand_physical_level_counts() {
        use arrow_array::{FixedSizeListArray, ListArray};
        use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
        use arrow_schema::{DataType, Field};
        for entries in [1, 17, 129, 1025, 4097] {
            let field = Arc::new(Field::new("item", DataType::Int64, true));
            let values = Arc::new(Int64Array::from(vec![0; entries * 8])) as ArrayRef;
            let inner =
                FixedSizeListArray::try_new(field, 8, values, Some(NullBuffer::new_null(entries)))
                    .unwrap();
            let field = Arc::new(Field::new("item", inner.data_type().clone(), true));
            let outer = ListArray::try_new(
                field,
                OffsetBuffer::new(ScalarBuffer::from(vec![0, entries as i32])),
                Arc::new(inner),
                None,
            )
            .unwrap();
            let batch =
                RecordBatch::try_from_iter([("nested", Arc::new(outer) as ArrayRef)]).unwrap();
            let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), None).unwrap();
            writer.write(&batch).unwrap();
            let bytes = bytes::Bytes::from(writer.into_inner().unwrap());
            let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
            let physical = builder.metadata().row_group(0).column(0).num_values() as usize;
            let decoded = builder.build().unwrap().next().unwrap().unwrap();
            let outer = decoded
                .column(0)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let inner = outer
                .values()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            assert_eq!(decoded.num_rows(), 1);
            assert_eq!(inner.len(), entries);
            assert_eq!(inner.null_count(), entries);
            assert_eq!(physical, entries);
            assert_eq!(inner.values().len(), physical * 8);
            assert!(inner.values().get_array_memory_size() >= physical * 8 * 8);
            println!(
                "null_fixed_list entries={entries} physical_values={physical} child_slots={} array_bytes={}",
                inner.values().len(),
                decoded.get_array_memory_size()
            );
        }
    }
    #[test]
    fn native_arrow_dictionary_fallback_retains_shared_values() {
        use arrow_array::{DictionaryArray, builder::StringDictionaryBuilder, types::Int16Type};
        let mut values = StringDictionaryBuilder::<Int16Type>::new();
        for value in [Some("a"), None, Some("b"), Some("a")] {
            match value {
                Some(value) => {
                    values
                        .append(format!("{value}{}", "x".repeat(4096)))
                        .unwrap();
                }
                None => values.append_null(),
            }
        }
        let input = values.finish();
        let batch =
            RecordBatch::try_from_iter(vec![("dictionary", Arc::new(input) as ArrayRef)]).unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::new(), batch.schema(), Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        let bytes = bytes::Bytes::from(writer.into_inner().unwrap());
        let metadata = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let pool = Pool::new(1024 * 1024);
        let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)
            .unwrap()
            .with_decoder_memory(Arc::new(Governor(pool.clone())))
            .build()
            .unwrap();
        let mut reader = next_reader(&mut decoder, &bytes).unwrap().unwrap();
        let batch = reader.next().unwrap().unwrap();
        let output = arrow_cast::cast(batch.column(0), &arrow_schema::DataType::Utf8).unwrap();
        let expected = Arc::new(StringArray::from(vec![
            Some(format!("a{}", "x".repeat(4096))),
            None,
            Some(format!("b{}", "x".repeat(4096))),
            Some(format!("a{}", "x".repeat(4096))),
        ])) as ArrayRef;
        assert_eq!(output.to_data(), expected.to_data());
        let dictionary = batch
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int16Type>>()
            .unwrap()
            .values()
            .clone();
        drop(output);
        drop(expected);
        drop(batch);
        assert!(reader.next().is_none());
        drop(reader);
        drop(decoder);
        assert!(pool.used.load(Ordering::SeqCst) > 0);
        drop(dictionary);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn native_selected_output_growth_refuses_before_exceeding_pool() {
        let (bytes, _) = file(Encoding::PLAIN, false, 32, 128 * 1024);
        let pool = Pool::new(1024 * 1024);
        let mut decoder = decoder(&bytes, pool.clone(), false, false, false);
        let mut reader = next_reader(&mut decoder, &bytes).unwrap().unwrap();
        let error = reader.next().unwrap().unwrap_err();
        assert!(error.to_string().contains("governed memory refused"));
        assert!(pool.peak.load(Ordering::SeqCst) <= 1024 * 1024);
        drop(reader);
        drop(decoder);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn native_refuses_page_value_counts_beyond_admitted_metadata() {
        let (bytes, _) = file(Encoding::PLAIN, false, 64, 32);
        let original = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let mut columns = original.row_group(0).columns().to_vec();
        columns[1] = columns[1]
            .clone()
            .into_builder()
            .set_num_values(1)
            .build()
            .unwrap();
        let group = original
            .row_group(0)
            .clone()
            .into_builder()
            .set_column_metadata(columns)
            .build()
            .unwrap();
        let metadata = Arc::new(
            original
                .as_ref()
                .clone()
                .into_builder()
                .set_row_groups(vec![group])
                .build(),
        );
        let pool = Pool::new(1024 * 1024);
        let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)
            .unwrap()
            .with_decoder_memory(Arc::new(Governor(pool.clone())))
            .build()
            .unwrap();
        let mut reader = next_reader(&mut decoder, &bytes).unwrap().unwrap();
        let error = reader.next().unwrap().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Data page values exceed column metadata")
        );
        drop(reader);
        drop(decoder);
        assert_eq!(pool.used.load(Ordering::SeqCst), 0);
    }
}
