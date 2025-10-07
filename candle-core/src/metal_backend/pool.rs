use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crate::{Error, Result};

use super::MetalDevice;
use metal::{Buffer, HeapDescriptor, MTLResourceOptions, MTLStorageMode, NSUInteger};

#[derive(Debug)]
struct MetalPoolInner {
    device: MetalDevice,
    heap: metal::Heap,
    capacity: u64,
    id: usize,
}

#[derive(Clone, Debug)]
pub struct MetalTensorPool {
    inner: Arc<MetalPoolInner>,
}

impl MetalTensorPool {
    pub fn new(device: &MetalDevice, size_in_bytes: usize) -> Result<Self> {
        if size_in_bytes == 0 {
            crate::bail!("metal pool size must be greater than zero")
        }
        let descriptor = HeapDescriptor::new();
        descriptor.set_size(size_in_bytes as NSUInteger);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        // descriptor.set_heap_type(MTLHeapType::Placement);
        // descriptor.set_resource_options(MTLResourceOptions::StorageModePrivate);

        let heap = device.device.new_heap(&descriptor);

        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            inner: Arc::new(MetalPoolInner {
                device: device.clone(),
                heap,
                capacity: size_in_bytes as u64,
                id,
            }),
        })
    }

    pub fn id(&self) -> usize {
        self.inner.id
    }

    pub fn device(&self) -> &MetalDevice {
        &self.inner.device
    }

    pub fn capacity(&self) -> u64 {
        self.inner.capacity
    }

    pub fn allocate_buffer(
        &self,
        size: NSUInteger,
        label: &str,
        options: MTLResourceOptions,
    ) -> Result<Arc<Buffer>> {
        if size > self.inner.capacity {
            crate::bail!(
                "pool allocation of {size} bytes exceeds pool capacity {}",
                self.inner.capacity
            )
        }
        let size_align = self
            .inner
            .device
            .device
            .heap_buffer_size_and_align(size, options);
        let align = std::cmp::max(size_align.align, 1);
        let available = self.inner.heap.max_available_size_with_alignment(align);
        if size_align.size > available {
            crate::bail!(
                "pool allocation of {size} bytes exceeds remaining capacity {}",
                available
            )
        }
        let buffer = self
            .inner
            .heap
            .new_buffer(size, options)
            .ok_or_else(|| Error::msg("metal heap allocation returned null"))?;
        buffer.set_label(label);
        println!("allocating {size} with {options:?}");
        Ok(Arc::new(buffer))
    }
}
