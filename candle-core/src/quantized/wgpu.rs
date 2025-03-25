use super::GgmlDType;
use crate::{backend::{BackendDevice, BackendStorage}, DType, Result, Shape, WgpuDevice, WgpuStorage};
use crate::wgpu_backend::cache::BufferReferenceId;

pub struct QWgpuStorage {
    dtype: GgmlDType,
    storage: WgpuStorage,
}

impl QWgpuStorage {
    pub fn new(dtype: GgmlDType, storage : WgpuStorage) -> Self {
        Self { dtype, storage }
    }
    pub fn buffer(&self) -> &BufferReferenceId {
        self.storage.buffer()
    }
    pub fn zeros(device: &WgpuDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        Ok(QWgpuStorage::new(dtype, device.zeros_impl(&(size / 4,).into(), DType::U32)?))
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &WgpuDevice {
        self.storage.device()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.get_length()
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<WgpuStorage> {
        let dev = self.device();
        let dst = dev.alloc_uninit_size(DType::F32, elem_count);
      
        let mut queue = dev.get_queue();
        queue.add(elem_count);
        let pipeline = match self.dtype(){
            GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_0::Functions::DequantizeBlockToF32),
            GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_1::Functions::DequantizeBlockToF32),
            GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_0::Functions::DequantizeBlockToF32),
            GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_1::Functions::DequantizeBlockToF32),
            GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_0::Functions::DequantizeBlockToF32),
            GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_1::Functions::DequantizeBlockToF32),

            GgmlDType::Q2K => candle_wgpu_kernels::Pipelines::Q2K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q2_k::Functions::DequantizeBlockToF32),
            GgmlDType::Q3K => candle_wgpu_kernels::Pipelines::Q3K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q3_k::Functions::DequantizeBlockToF32),
            GgmlDType::Q4K => candle_wgpu_kernels::Pipelines::Q4K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_k::Functions::DequantizeBlockToF32),
            GgmlDType::Q5K => candle_wgpu_kernels::Pipelines::Q5K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_k::Functions::DequantizeBlockToF32),
            GgmlDType::Q6K => candle_wgpu_kernels::Pipelines::Q6K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q6_k::Functions::DequantizeBlockToF32),
            GgmlDType::Q8K => candle_wgpu_kernels::Pipelines::Q8K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_k::Functions::DequantizeBlockToF32),
            _ => {
                crate::bail!("Dequantize not implemented for {:?}", self.dtype());
            }
        };
        let pipeline = queue.get_pipeline(pipeline);
        let bind_group = dev.create_bind_group_input1(*dst.buffer(), *self.buffer(), DType::F32.into());
        queue.enqueue_64(pipeline, bind_group, (elem_count / self.dtype().block_size()) as u32,elem_count);

        Ok(dst)
    }

    pub fn quantize(&mut self, src: &WgpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu_storage()?;
        let elem_count = src.as_slice::<f32>()?.len();
        let src = crate::Storage::Cpu(src);
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device().alloc_from_slice(DType::U32, &qcpu_storage.data()?)?;
        self.storage = buffer;
        Ok(())
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(WgpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let src_shape = src_shape.dims().to_vec();

        
        let (b,m) = match src_shape.len() {
            3 => (src_shape[0], src_shape[1]),
            2 => (1, src_shape[0]),
            n => crate::bail!("Invalid rank {n} for quantized matmul wgpu"),
        };
        let mut dst_shape = src_shape;
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);

        let dst_shape = Shape::from(dst_shape);
        let dev = storage.device();
        let dst = dev.alloc_uninit_size(DType::F32, dst_shape.elem_count());
        
        let mut input1_stride = layout.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);


        let mut queue = dev.get_queue();
        queue.add(b);
        queue.add(m);
        queue.add(k);
        queue.add(n);

        queue.add(input1_stride_b); //input1_stride_b
        queue.add(layout.start_offset()); //input1_offset
        queue.add(0); //input2_stride_b
        queue.add(0); //input2_ofset
        queue.add(input1_stride_k);
        queue.add(input1_stride_m);

        let pipeline = match self.dtype(){
            GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_0::Functions::MatmulNaiveBlock),
            GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_1::Functions::MatmulNaiveBlock),
            GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_0::Functions::MatmulNaiveBlock),
            GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_1::Functions::MatmulNaiveBlock),
            GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_0::Functions::MatmulNaiveBlock),
            GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_1::Functions::MatmulNaiveBlock),
            GgmlDType::Q2K => candle_wgpu_kernels::Pipelines::Q2K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q2_k::Functions::MatmulNaiveBlock),
            GgmlDType::Q3K => candle_wgpu_kernels::Pipelines::Q3K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q3_k::Functions::MatmulNaiveBlock),
            GgmlDType::Q4K => candle_wgpu_kernels::Pipelines::Q4K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q4_k::Functions::MatmulNaiveBlock),
            GgmlDType::Q5K => candle_wgpu_kernels::Pipelines::Q5K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q5_k::Functions::MatmulNaiveBlock),
            GgmlDType::Q6K => candle_wgpu_kernels::Pipelines::Q6K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q6_k::Functions::MatmulNaiveBlock),
            GgmlDType::Q8K => candle_wgpu_kernels::Pipelines::Q8K(candle_wgpu_kernels::DType::F32, candle_wgpu_kernels::quantized::q8_k::Functions::MatmulNaiveBlock),
            _ => todo!()
        };
        let const_vec = vec![
            (input1_stride_k == 1) as usize,
            (input1_stride_m == 1) as usize,
            (b != 1) as usize,
        ];
        
        let pipeline = queue.get_pipeline_const(pipeline, const_vec);
        let bind_group = dev.create_bind_group_input2(*dst.buffer(), *storage.buffer(),*self.buffer(), DType::F32.into());

        queue.enqueue_workgroups(
            pipeline,
            bind_group,
            (n as u32 + 15) / 16,
            (m as u32+ 15) / 16,
            b as u32,
            k * m * n * b,
        );
    
        Ok((dst, dst_shape))
    }
}

// pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
//     device: &WgpuDevice,
//     data: &[T],
// ) -> Result<QStorage> {
//     let storage = device.alloc_from_slice(DType::F32, data)?;
//     let device = device.clone();
//     Ok(QStorage::Wgpu(QWgpuStorage {
//         dtype: T::DTYPE,
//         storage,
//     }))
// }
