use crate::Result;
use metal::Buffer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{MetalDevice, MetalError};

const ALIGNMENT: usize = 256;

fn align_size(size: usize) -> usize {
    if size == 0 {
        0
    } else {
        ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT
    }
}

#[derive(Debug)]
struct PoolState {
    used: usize,
    free: HashMap<usize, Vec<Arc<Buffer>>>,
}

impl PoolState {
    fn new() -> Self {
        Self {
            used: 0,
            free: HashMap::new(),
        }
    }

    fn pop(&mut self, size: usize) -> Option<Arc<Buffer>> {
        let mut remove_entry = false;
        let buffer = self.free.get_mut(&size).and_then(|list| {
            let buffer = list.pop();
            if list.is_empty() {
                remove_entry = true;
            }
            buffer
        });
        if remove_entry {
            self.free.remove(&size);
        }
        buffer
    }

    fn push(&mut self, size: usize, buffer: Arc<Buffer>) {
        self.free.entry(size).or_default().push(buffer);
    }
}

#[derive(Debug)]
pub struct MetalTensorPool {
    device: MetalDevice,
    capacity: usize,
    state: Mutex<PoolState>,
}

impl MetalTensorPool {
    pub fn new(device: MetalDevice, capacity: usize) -> Result<Arc<Self>> {
        if capacity == 0 {
            crate::bail!("Pool capacity must be greater than 0");
        }
        Ok(Arc::new(Self {
            device,
            capacity,
            state: Mutex::new(PoolState::new()),
        }))
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn allocate(self: &Arc<Self>, size_in_bytes: usize) -> Result<Arc<MetalPoolAllocation>> {
        if size_in_bytes == 0 {
            crate::bail!("Cannot allocate zero bytes from pool");
        }
        let aligned = align_size(size_in_bytes);
        let mut state = self.state.lock().map_err(MetalError::from)?;
        if state.used + aligned > self.capacity {
            crate::bail!(
                "Metal tensor pool exhausted: requested {aligned} bytes, capacity {} bytes",
                self.capacity
            );
        }
        let buffer = if let Some(buffer) = state.pop(aligned) {
            buffer
        } else {
            Arc::new(self.device.device().new_buffer(
                aligned as u64,
                metal::MTLResourceOptions::StorageModePrivate,
            ))
        };
        state.used += aligned;
        drop(state);
        Ok(Arc::new(MetalPoolAllocation {
            pool: Arc::clone(self),
            buffer,
            size: aligned,
        }))
    }

    fn release(&self, size: usize, buffer: Arc<Buffer>) {
        if let Ok(mut state) = self.state.lock() {
            state.used = state.used.saturating_sub(size);
            state.push(size, buffer);
        }
    }
}

#[derive(Debug)]
pub struct MetalPoolAllocation {
    pool: Arc<MetalTensorPool>,
    buffer: Arc<Buffer>,
    size: usize,
}

impl MetalPoolAllocation {
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
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
        let buffer = Arc::clone(&self.buffer);
        self.pool.release(self.size, buffer);
    }
}
