use crate::Result;
use metal::Buffer;
use std::sync::{Arc, Mutex};

use super::{MetalDevice, MetalError};

const ALIGNMENT: usize = 256;

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    ((value + alignment - 1) / alignment) * alignment
}

#[derive(Debug)]
struct PoolState {
    free: Vec<(usize, usize)>,
    used: usize,
}

impl PoolState {
    fn new(capacity: usize) -> Self {
        Self {
            free: vec![(0, capacity)],
            used: 0,
        }
    }

    fn allocate(&mut self, size: usize, alignment: usize, capacity: usize) -> Option<usize> {
        let alignment = alignment.max(ALIGNMENT);
        for idx in 0..self.free.len() {
            let (start, len) = self.free[idx];
            let aligned_start = align_up(start, alignment);
            if aligned_start >= start + len {
                continue;
            }
            let padding = aligned_start - start;
            let available = len.saturating_sub(padding);
            if available < size {
                continue;
            }
            let remaining = available - size;
            self.free.remove(idx);
            if padding > 0 {
                self.free.insert(idx, (start, padding));
            }
            if remaining > 0 {
                self.free.insert(
                    idx + (padding > 0) as usize,
                    (aligned_start + size, remaining),
                );
            }
            self.used += size;
            debug_assert!(self.used <= capacity);
            return Some(aligned_start);
        }
        None
    }

    fn free(&mut self, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        self.used = self.used.saturating_sub(size);
        let mut insert_pos = 0;
        while insert_pos < self.free.len() && self.free[insert_pos].0 < offset {
            insert_pos += 1;
        }
        self.free.insert(insert_pos, (offset, size));
        // Merge with previous block if adjacent.
        if insert_pos > 0 {
            if let Some(merged) = try_merge(self.free[insert_pos - 1], self.free[insert_pos]) {
                self.free[insert_pos - 1] = merged;
                self.free.remove(insert_pos);
                insert_pos -= 1;
            }
        }
        // Merge with next block if adjacent.
        if insert_pos + 1 < self.free.len() {
            if let Some(merged) = try_merge(self.free[insert_pos], self.free[insert_pos + 1]) {
                self.free[insert_pos] = merged;
                self.free.remove(insert_pos + 1);
            }
        }
    }
}

fn try_merge(lhs: (usize, usize), rhs: (usize, usize)) -> Option<(usize, usize)> {
    if lhs.0 + lhs.1 == rhs.0 {
        Some((lhs.0, lhs.1 + rhs.1))
    } else {
        None
    }
}

#[derive(Debug)]
pub struct MetalTensorPool {
    device: MetalDevice,
    capacity: usize,
    buffer: Arc<Buffer>,
    state: Mutex<PoolState>,
}

impl MetalTensorPool {
    pub fn new(device: MetalDevice, capacity: usize) -> Result<Arc<Self>> {
        if capacity == 0 {
            crate::bail!("Pool capacity must be greater than 0");
        }
        let buffer = Arc::new(device.device().new_buffer(
            capacity as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        ));
        Ok(Arc::new(Self {
            device,
            capacity,
            buffer,
            state: Mutex::new(PoolState::new(capacity)),
        }))
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    pub fn allocate(
        self: &Arc<Self>,
        size_in_bytes: usize,
        alignment: usize,
    ) -> Result<Arc<MetalPoolAllocation>> {
        if size_in_bytes == 0 {
            crate::bail!("Cannot allocate zero bytes from pool");
        }
        let mut state = self.state.lock().map_err(MetalError::from)?;
        if let Some(offset) = state.allocate(size_in_bytes, alignment, self.capacity) {
            Ok(Arc::new(MetalPoolAllocation {
                pool: Arc::clone(self),
                offset,
                size: size_in_bytes,
            }))
        } else {
            crate::bail!(
                "Metal tensor pool exhausted: requested {size_in_bytes} bytes, capacity {} bytes",
                self.capacity
            )
        }
    }

    fn release(&self, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        if let Ok(mut state) = self.state.lock() {
            state.free(offset, size);
        }
    }
}

#[derive(Debug)]
pub struct MetalPoolAllocation {
    pool: Arc<MetalTensorPool>,
    offset: usize,
    size: usize,
}

impl MetalPoolAllocation {
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.pool.buffer
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn pool(&self) -> Arc<MetalTensorPool> {
        Arc::clone(&self.pool)
    }
}

impl Drop for MetalPoolAllocation {
    fn drop(&mut self) {
        self.pool.release(self.offset, self.size);
    }
}
