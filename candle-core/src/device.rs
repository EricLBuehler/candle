use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, Result, Shape, Storage, WithDType};

/// A `DeviceLocation` represents a physical device whereas multiple `Device`
/// can live on the same location (typically for cuda devices).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
    Wgpu { gpu_id: usize },
}

/// Cpu, Cuda, or Metal
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    Wgpu(crate::WgpuDevice)
}

pub trait NdArray {
    fn shape(&self) -> Result<Shape>;

    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> NdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: NdArray> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        // This allocates intermediary memory and shouldn't be necessary.
        let storages = self.iter().map(|v| v.to_cpu_storage()).collect::<Vec<_>>();
        CpuStorage::concat(storages.as_slice()).unwrap()
    }
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
    }

    pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
        match self {
            Self::Cuda(d) => Ok(d),
            Self::Cpu => crate::bail!("expected a cuda device, got cpu"),
            Self::Metal(_) => crate::bail!("expected a cuda device, got Metal"),
            Self::Wgpu(_) => crate::bail!("expected a cuda device, got Wgpu"),
        }
    }

    pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
        match self {
            Self::Cuda(_) => crate::bail!("expected a metal device, got cuda"),
            Self::Cpu => crate::bail!("expected a metal device, got cpu"),
            Self::Metal(d) => Ok(d),
            Self::Wgpu(_) => crate::bail!("expected a metal device, got Wgpu"),
        }
    }

    pub fn as_wgpu_device(&self) -> Result<&crate::WgpuDevice> {
        match self {
            Self::Cuda(_) => crate::bail!("expected a wgpu device, got cuda"),
            Self::Cpu => crate::bail!("expected a wgpu device, got cpu"),
            Self::Metal(_) => crate::bail!("expected a wgpu device, got metal"),
            Self::Wgpu(d) => Ok(d),
        }
    }

    pub fn new_cuda_with_stream(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new_with_stream(ordinal)?))
    }

    pub fn new_metal(ordinal: usize) -> Result<Self> {
        Ok(Self::Metal(crate::MetalDevice::new(ordinal)?))
    }

    pub async fn new_wgpu_async(ordinal: usize) -> Result<Self> {
        Ok(Self::Wgpu(crate::WgpuDevice::create(ordinal, crate::WgpuDeviceConfig::default()).await?))
    }

    pub async fn new_wgpu_config_async(ordinal: usize, configuration : crate::WgpuDeviceConfig) -> Result<Self> {
        Ok(Self::Wgpu(crate::WgpuDevice::create(ordinal, configuration).await?))
    }

    ///creates a new wgpu device synchronously. 
    ///If you are targeting wasm32, use the async functions to create a new device.
    #[cfg(all(feature="wgpu", not(target_arch = "wasm32")))]
    pub fn new_wgpu(ordinal: usize) -> Result<Self> {
        pollster::block_on(Device::new_wgpu_async(ordinal))
    }

    ///creates a new wgpu device synchronously. 
    ///If you are targeting wasm32, use the async functions to create a new device.
    #[cfg(all(feature="wgpu", not(target_arch = "wasm32")))]
    pub fn new_wgpu_config(ordinal: usize, configuration : crate::WgpuDeviceConfig) -> Result<Self> {
        pollster::block_on(Device::new_wgpu_config_async(ordinal, configuration))
    }

    #[cfg(not(feature="wgpu"))]
    pub fn new_wgpu(_: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithWgpuSupport)
    }

    #[cfg(not(feature="wgpu"))]
    pub fn new_wgpu_config(_: usize, _ : crate::WgpuDeviceConfig) -> Result<Self> {
        Err(crate::Error::NotCompiledWithWgpuSupport)
    }

    pub fn set_seed(&self, seed: u64) -> Result<()> {
        match self {
            Self::Cpu => CpuDevice.set_seed(seed),
            Self::Cuda(c) => c.set_seed(seed),
            Self::Metal(m) => m.set_seed(seed),
            Self::Wgpu(m) => m.set_seed(seed),
        }
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            (Self::Wgpu(lhs), Self::Wgpu(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => device.location(),
            Device::Metal(device) => device.location(),
            Self::Wgpu(device) => device.location(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }

    pub fn is_wgpu(&self) -> bool {
        matches!(self, Self::Wgpu(_))
    }

    pub fn is_dtype_available(&self, dtype: DType) -> bool{
        match (self, dtype) {
            (Device::Cpu, _) => true,
            (Device::Cuda(_), _) => true,
            (Device::Metal(_), _) => true,
            (Device::Wgpu(_), DType::U32) => true,
            (Device::Wgpu(_), DType::F32) => true,

            (Device::Wgpu(_), DType::U8) => false,
           
            (Device::Wgpu(_dev), DType::I64) => 
            {
                #[cfg(feature="wgpu")]
                return _dev.device_features.contains(wgpu::Features::SHADER_INT64);
                #[cfg(not(feature="wgpu"))]
                return false;
            }
            (Device::Wgpu(_dev), DType::F64) =>  
            {
                #[cfg(feature="wgpu")]
                return _dev.device_features.contains(wgpu::Features::SHADER_F64);
                #[cfg(not(feature="wgpu"))]
                return false;
            },
            (Device::Wgpu(_dev), DType::F16) =>  
            {
                #[cfg(feature="wgpu")]
                return _dev.device_features.contains(wgpu::Features::SHADER_F16);
                #[cfg(not(feature="wgpu"))]
                return false;
            },

            (Device::Wgpu(_), DType::BF16) => false,   
        }
    }

    pub fn supports_bf16(&self) -> bool {
        match self {
            Self::Cuda(_) | Self::Metal(_) => true,
            Self::Cpu | Self::Wgpu(_)=> false,
        }
    }

    /// Return `BF16` for devices that support it, otherwise default to `F32`.
    pub fn bf16_default_to_f32(&self) -> DType {
        if self.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::cuda_is_available() {
            Self::new_cuda(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }

    pub(crate) fn rand_uniform_f64(
        &self,
        lo: f64,
        up: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_uniform(shape, DType::F32, lo, up)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_uniform(shape, dtype, lo, up)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = device.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) fn rand_uniform<T: crate::FloatDType>(
        &self,
        lo: T,
        up: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_uniform_f64(lo.to_f64(), up.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn rand_normal_f64(
        &self,
        mean: f64,
        std: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_normal(shape, DType::F32, mean, std)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_normal(shape, dtype, mean, std)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = device.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) fn rand_normal<T: crate::FloatDType>(
        &self,
        mean: T,
        std: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_normal_f64(mean.to_f64(), std.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.ones_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = device.ones_impl(shape, dtype)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) fn storage_from_slice<D: WithDType>(&self, data: &[D]) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(data.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
            Device::Wgpu(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Wgpu(storage))
            }
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
            Self::Wgpu(d) => d.synchronize(),
        }
    }

    pub async fn synchronize_async(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
            Self::Wgpu(d) => d.synchronize_async().await,
        }
    }
}
