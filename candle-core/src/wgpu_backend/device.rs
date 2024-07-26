use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};
use std::hash::Hash;

use candle_wgpu_kernels::EntryPoint;
use rand::SeedableRng;
use tracing::instrument;

use crate::backend::BackendStorage;
use crate::{notImplemented, wrongType, Layout};

#[cfg(feature = "wgpu_debug")]
use super::debug_info::{DebugInfo, Measurements,MInfo, ShaderInfo};

use super::cache::{BindGroupReferenceBase, BindgroupId, BindgroupLayouts, BufferReference, ModelCache};
use super::util::ToU32;
use super::wgpu_functions::{self, create_buffer_init, unary::UnaryOperation, MetaArray};
use super::WgpuStorage;

#[derive(Debug)]
pub (crate) enum MlQueue{
    Dispatch(MlQueueDispatch),
}

#[cfg(feature = "wgpu_debug")]
pub trait ToU64 {
    fn to_u64(self) -> u64;
}

#[cfg(feature = "wgpu_debug")]
macro_rules! impl_to_u64 {
    ($($ty:ty)*) => {
        $(
            impl ToU64 for $ty {
                #[inline]
                fn to_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    }
}
#[cfg(feature = "wgpu_debug")]
impl_to_u64!(u8 u16 u32 u64 usize);

#[derive(Debug, Copy, Clone)]
pub struct OrderedFloat(pub f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &OrderedFloat) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpIsInplaceable{
    pub input1_inplaceable : bool,
    pub input2_inplaceable : bool,
}

impl OpIsInplaceable {
    pub fn new() -> Self {
        Self { input1_inplaceable : false, input2_inplaceable : false }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub (crate) struct PipelineType(pub candle_wgpu_kernels::Pipelines, pub Vec<OrderedFloat>, pub OpIsInplaceable);
pub (crate) type BindGroupReference = BindGroupReferenceBase<Arc<BufferReference>>;

#[derive(Debug)]
pub (crate) enum DispatchedBindgroup {
    BindgroupReference(BindGroupReference),
    CachedBindgroup(BindgroupId),
  }

#[derive(Debug)]
pub (crate) struct MlQueueDispatch{
    pub (crate) x : u32, 
    pub (crate) y : u32, 
    pub (crate) z : u32,
    pub (crate) pipeline : PipelineType,
    pub (crate) bindgroup : DispatchedBindgroup,
    pub (crate) meta : u32,
    pub (crate) workload_size : usize, //the total size needed to calculate. Needed so we do not queue to many operations at once.
    #[cfg(feature = "wgpu_debug")]
    pub (crate) debug : Option<String>,
}

#[derive(Debug)]
pub struct ShaderModuleComputePipelines{
    shader : Arc<wgpu::ShaderModule>,
    pipelines : Mutex<HashMap<PipelineType, Arc<wgpu::ComputePipeline>>>
}

//a struct, where all operations are chunked 
#[derive(Debug)]
pub struct QueueBuffer{
    pub (crate) command_queue : Vec<MlQueue>,
    pub (crate) meta_array : MetaArray,
    pub (crate) current_meta : u32
}

impl QueueBuffer {
    pub fn new() -> Self {
        Self {  command_queue: vec![], meta_array :MetaArray::new(META_BUFFER_SIZE), current_meta : 0 }
    }

    pub (crate) fn add_layout(&mut self, layout : &Layout){
        self.meta_array.add_layout(layout);
    } 

    pub (crate) fn add<T : ToU32>(&mut self, value : T){
        self.meta_array.add(value);
    }

}

#[derive(Debug)]
pub struct WgpuDeviceInner{
    pub device : wgpu::Device, 
    pub device_limits : wgpu::Limits, //we cache the limits here, because device.limit() was relatively slow on the browser

    pub queue : wgpu::Queue,
    pub (crate) shader : Mutex<HashMap<candle_wgpu_kernels::Shaders, ShaderModuleComputePipelines>>,
    pub (crate) rand_state : Mutex<rand::rngs::StdRng>,   

    pub (crate) command_queue : Mutex<QueueBuffer>,
    pub (crate) meta_buffer : wgpu::Buffer, //buffer for storing meta information

    pub (crate) bindgroup_layouts : BindgroupLayouts,

    pub (crate) cache : Mutex<ModelCache>, //if cache is set, all commands are not queued to the gpu, but are cached inside ModelCache, so there can be reused later on
    pub (crate) cached_buffer_counter : AtomicU32,
    pub (crate) cached_bindgroup_counter : AtomicU32,

    pub (crate) cached_bindgroup_reuse_counter : AtomicU32,
    pub (crate) cached_buffer_reuse_counter : AtomicU32,
    pub (crate) cached_buffer_inplace_counter : AtomicU32,
    pub (crate) use_cache : bool,
    #[cfg(feature = "wgpu_debug")]
    pub debug : DebugInfo,
}

#[derive(Debug, Clone)]
pub struct  WgpuDevice {
    pub inner : Arc<WgpuDeviceInner>,
}

impl std::ops::Deref for WgpuDevice{
    type Target = WgpuDeviceInner;

    fn deref(&self) -> &Self::Target {
        return &self.inner;
    }
}

//pub (crate) const META_BUFFER_SIZE : u32 = 65536;
//pub (crate) const META_BUFFER_SIZE : u32 = 2048;
pub (crate) const META_BUFFER_SIZE : u32 = 10*1024*1024; //10mb

pub (crate) const MAX_WORKLOAD_SIZE : u64 = 1024u64*1024*1024*10; //10gb

impl WgpuDevice{
    pub (crate) async fn create(_: usize) -> crate::Result<Self>{
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();

        let mut limits = wgpu::Limits::downlevel_defaults();

        #[cfg(feature = "wgpu_debug")]
        let features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        #[cfg(not(feature = "wgpu_debug"))]
        let features = wgpu::Features::empty();

        let adatper_limits = adapter.limits();
        limits.min_storage_buffer_offset_alignment = adatper_limits.min_storage_buffer_offset_alignment;
        limits.max_storage_buffers_per_shader_stage = 5;
        limits.max_storage_buffer_binding_size = adatper_limits.max_storage_buffer_binding_size; //use as much as possible
        limits.max_buffer_size = adatper_limits.max_buffer_size; //use as much as possible

     
        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                    memory_hints : wgpu::MemoryHints::Performance
                },
                None,
            ).await.map_err(|err| crate::Error::WebGpu(err.to_string().into()))?;
    
        
        #[cfg(feature = "wgpu_debug")]
        let debug_info = super::debug_info::DebugInfo::new(&device);
        
        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: META_BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let device_limits = device.limits();
        let bindgroup_layouts = BindgroupLayouts::new(&device);
        
        Ok(WgpuDevice {
            inner : Arc::new(WgpuDeviceInner{
                device: device,
                device_limits: device_limits,
                queue: queue,
                shader : Mutex::new(HashMap::new()),
                rand_state: Mutex::new(rand::rngs::StdRng::from_entropy()),
                #[cfg(feature = "wgpu_debug")]
                debug : debug_info,
                command_queue: Mutex::new(QueueBuffer::new()),
                meta_buffer : meta_buffer,
                cache : Mutex::new(ModelCache::new()),
                bindgroup_layouts,
                cached_buffer_counter : AtomicU32::new(0),
                cached_bindgroup_counter  : AtomicU32::new(0),
                cached_bindgroup_reuse_counter: AtomicU32::new(0),
    
                cached_buffer_reuse_counter: AtomicU32::new(0),
                cached_buffer_inplace_counter : AtomicU32::new(0),
                use_cache : true
            })
        })
    }

    pub fn print_bindgroup_reuseinfo(&self){
        log::info!("Buffer: created: {}, resued : {}", self.cached_buffer_counter.load(std::sync::atomic::Ordering::Relaxed), self.cached_buffer_reuse_counter.load(std::sync::atomic::Ordering::Relaxed));
        log::info!("Bindgroup: created: {}, resued : {}", self.cached_bindgroup_counter.load(std::sync::atomic::Ordering::Relaxed), self.cached_bindgroup_reuse_counter.load(std::sync::atomic::Ordering::Relaxed));
        log::info!("Bindgroup Inplace used: {}", self.cached_buffer_inplace_counter.load(std::sync::atomic::Ordering::Relaxed));
    }
    pub fn print_bindgroup_reuseinfo2(&self){
        println!("Buffer: created: {}, resued : {}", self.cached_buffer_counter.load(std::sync::atomic::Ordering::Relaxed), self.cached_buffer_reuse_counter.load(std::sync::atomic::Ordering::Relaxed));
        println!("Bindgroup: created: {}, resued : {}", self.cached_bindgroup_counter.load(std::sync::atomic::Ordering::Relaxed), self.cached_bindgroup_reuse_counter.load(std::sync::atomic::Ordering::Relaxed));
        println!("Bindgroup Inplace used: {}", self.cached_buffer_inplace_counter.load(std::sync::atomic::Ordering::Relaxed));
    }
    
    pub fn clear_cache(&self){
        log::info!("Clear cache");
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info_full(&self) -> crate::Result<Measurements>{
        let data = wgpu_functions::read_data_from_gpu_async_buffer::<u64>(self, &self.debug.query_set_buffer).await;
           
        let period = self.queue.get_timestamp_period();
        let mut result = Measurements::new(period);
        let mut last_end_time = 0u64;
        let mut i = 0;
        let mut shader_pipeline2 = self.debug.shader_pipeline.lock().unwrap();
        let shader_pipeline = shader_pipeline2.clone();
        let mut indexes : Vec<_> = shader_pipeline.into_iter().collect();
        indexes.sort_by_key(|f| f.0);
        for p in indexes {
            let start_time = data[(p.0 / 8) as usize];
            let end_time = data[(p.0 / 8) as usize + 1];
            if end_time < start_time {
                panic!("Start Time was after End Time! startTime: {start_time}, endTime:{end_time}, i:{i}");
            }
            if end_time == 0 {
                panic!("End time was 0")
            }
            if start_time == 0 {
                panic!("End time was 0")
            }
            if end_time == start_time {
                panic!("End time == start_time == {}", end_time)
            }
            if start_time < last_end_time {
                panic!("Start Time was before last End Time! startTime: {start_time}, last endTime:{last_end_time}, i:{i}");
            }
            last_end_time = end_time;
            i += 1;
            result.data.push(MInfo::new(p.1.0.to_owned(), start_time, end_time, p.1.1, p.1.2, p.1.3, p.1.4));
        }
        self.debug.counter.store(0u32, std::sync::atomic::Ordering::Relaxed);
        shader_pipeline2.clear();
        Ok(result)
    }

    #[cfg(feature = "wgpu_debug")]
    pub async fn get_debug_info(&self) -> crate::Result<std::collections::HashMap<String, Vec<(u64, u64, u32, u32, u32)>>>{
        let info = self.get_debug_info_full().await?;
        let mut map: std::collections::HashMap<String, Vec<(u64, u64, u32, u32, u32)>> = std::collections::HashMap::new();

        for item in info.data.iter() {
            map.entry(item.label.clone()).or_insert_with(Vec::new).push((item.end_time - item.start_time, item.output_size, item.x, item.y, item.z));
        }
        return Ok(map);
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn get_pipeline_info(&self) -> crate::Result<Vec<ShaderInfo>>{
        use super::debug_info;

        let shaders = self.shader.lock().unwrap();

        return Ok(shaders.iter().map(|(k, v)|{
            let pipelines = v.pipelines.lock().unwrap();
            let s = debug_info::ShaderInfo{    
                name: format!("{:?}", k).to_owned(), 
                pipelines: pipelines.iter().map(|(pk, _)|{
                    return debug_info::PipelineInfo { 
                        name: format!("{:?}", pk.0).to_owned(), 
                        consts : pk.1.iter().map(|f| f.0).collect()
                     }
                }).collect()

            };
            return s;
        }
        ).collect());
    }

    #[instrument]
    fn load_pipeline(device : &wgpu::Device, shader : Arc<wgpu::ShaderModule>, pipeline : &PipelineType, pipeline_layout : &wgpu::PipelineLayout) -> wgpu::ComputePipeline{
        let entry_point = pipeline.0.get_entry_point();
        if pipeline.1.is_empty(){
            return  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(pipeline_layout),
                module: &shader,
                entry_point: entry_point,
                compilation_options : wgpu::PipelineCompilationOptions::default(),
                cache : None
            });
        }
        else{
            let hmap : HashMap<_, _> = pipeline.1.iter().enumerate().map(|(index, value)| (format!("CONSTV_{index}"), value.0)).collect();

            return  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(pipeline_layout),
                module: &shader,
                entry_point: entry_point,
                compilation_options :  wgpu::PipelineCompilationOptions{
                    constants: &hmap,
                    zero_initialize_workgroup_memory: true,
                    vertex_pulling_transform: false,
                },
                cache : None
            });
        }
    }

    #[instrument]
    pub (crate) fn get_pipeline2(&self, pipeline: &PipelineType, pipeline_layout : &wgpu::PipelineLayout) -> crate::Result<Arc<wgpu::ComputePipeline>> {
        let shader = pipeline.0.get_shader();
        let mut shaders = self.shader.lock().unwrap();
        if !shaders.contains_key(&shader){
            let s = wgpu_functions::get_shader(&self.device, shader.load_shader());
            shaders.insert(shader.clone(), ShaderModuleComputePipelines{ shader: Arc::new(s), pipelines: Mutex::new(HashMap::new())});
        }
     
        if let Some(s) = shaders.get(&shader){
            let mut pipelines = s.pipelines.lock().unwrap();
            if !pipelines.contains_key(&pipeline){
                let p = crate::WgpuDevice::load_pipeline(&self.device, s.shader.clone(), pipeline ,pipeline_layout);
                pipelines.insert(pipeline.clone(), Arc::new(p));
            }
            
            if let Some(p) = pipelines.get(&pipeline){
                return Ok(p.clone());
            }
            else{
                panic!("Not expected")
            }
        }
        else{
            panic!("Not expected")
        }
    }
}

impl crate::backend::BackendDevice for WgpuDevice{
    type Storage = WgpuStorage;

    fn new(_: usize) -> crate::Result<Self> {
        return Err(crate::Error::WebGpu("A WgpuDevice must be created using the asynchronous create method".to_owned().into()));
    }

    fn location(&self) -> crate::DeviceLocation {
        return crate::DeviceLocation::Wgpu { gpu_id: 0 }; //TODO: WGPU
    }

    fn same_device(&self, other: &Self) -> bool {
        return self.device.global_id() == other.device.global_id();
    }

    fn zeros_impl(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        let buffer = BufferReference::new(self, shape.elem_count() * 4);
        if shape.elem_count() > 0{
            wgpu_functions::queue_unary_inplace_op(self, buffer.clone(), UnaryOperation::SetZero, 0.0, 0.0,dtype, &Layout::contiguous(shape))?;
        }
        
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
    }

    fn ones_impl(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        let buffer = BufferReference::new(self, shape.elem_count() * 4);
        if shape.elem_count() > 0{
            wgpu_functions::queue_unary_inplace_op(self, buffer.clone(), UnaryOperation::SetOne, 0.0, 0.0,dtype,&Layout::contiguous(shape))?;
        }
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
    }

    unsafe fn alloc_uninit(&self, shape: &crate::Shape, dtype: crate::DType) -> crate::Result<Self::Storage> {
        if dtype == crate::DType::F32 || dtype == crate::DType::U32{
            let buffer = BufferReference::new(self, shape.elem_count() * 4);
            return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
        }
        else{
            wrongType!(alloc_uninit, dtype);
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data : &[T]) -> crate::Result<Self::Storage> {
        let buffer;
        if T::DTYPE == crate::DType::F32{
            let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
            buffer = create_buffer_init(self, &data);
        }
        else if T::DTYPE == crate::DType::U32{
            let data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
            buffer = create_buffer_init(self, &data);
        }
        else{
            // Panic if T is not f32 or u32
            wrongType!(storage_from_slice, T::DTYPE);
        }
        return Ok(WgpuStorage::new(buffer, self.clone(),T::DTYPE));
    }

    fn storage_from_cpu_storage(&self, storage: &crate::CpuStorage) -> crate::Result<Self::Storage> {
        match storage{
            crate::CpuStorage::F32(data) => {
                let buffer = create_buffer_init(self, data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::F32));
            },
            crate::CpuStorage::U32(data) => {
                let buffer = create_buffer_init(self, data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::U32));
            },
            _ =>  wrongType!(storage_from_cpu_storage, storage.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: crate::CpuStorage) -> crate::Result<Self::Storage> {
        match storage{
            crate::CpuStorage::F32(data) => {
                let buffer = create_buffer_init(self, &data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::F32));
            },
            crate::CpuStorage::U32(data) => {
                let buffer = create_buffer_init(self, &data);
                return Ok(WgpuStorage::new(buffer, self.clone(),crate::DType::U32));
            },
            _ =>  wrongType!(storage_from_cpu_storage_owned, storage.dtype()),
        }
    }

    fn rand_uniform(&self, shape: &crate::Shape, dtype: crate::DType, lo: f64, up: f64) -> crate::Result<Self::Storage> {
        let buffer = BufferReference::new(self, shape.elem_count() * 4);
        wgpu_functions::queue_unary_inplace_op(self, buffer.clone(), UnaryOperation::RandUniform, lo as f32, up as f32,dtype,&Layout::contiguous(shape))?;
        return Ok(WgpuStorage::new(buffer, self.clone(), dtype));
    }

    fn rand_normal(&self, shape: &crate::Shape, dtype: crate::DType, mean: f64, std: f64) -> crate::Result<Self::Storage> {
        let buffer = BufferReference::new(self, shape.elem_count() * 4);
        wgpu_functions::queue_unary_inplace_op(self, buffer.clone(), UnaryOperation::RandNormal, mean as  f32, std as f32, dtype,&Layout::contiguous(shape))?;
        return Ok(WgpuStorage::new(buffer, self.clone(),dtype));
    }

    fn set_seed(&self, _: u64) -> crate::Result<()> {
        notImplemented!(set_seed)
    }

    
    #[cfg(target_arch = "wasm32")]
    fn synchronize(&self) -> crate::Result<()> {
        panic!("Synchronize is not possible on wasm. (on_submitted_work_done is currently not implemented in wgpu)");
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn synchronize(&self) -> crate::Result<()> {
        wgpu_functions::synchronize(self)
    }
}