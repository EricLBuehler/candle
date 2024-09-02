use tracing::instrument;

use crate::{backend::BackendStorage, DType, Layout, Shape};

use super::{
    cache::BufferReferenceId, device::WgpuDevice, util::ToU64, wgpu_functions::{self, binary::BinaryOperation, cmp::CmpOperation, matmul::SGEMMParams, read_data_from_gpu_async, reduce::ReduceOperations, unary::UnaryOperation}
};

#[derive(Debug)]
pub struct WgpuStorage {
    pub buffer: BufferReferenceId,
    pub size : u64, 
    pub wgpu_device: WgpuDevice,
    pub dtype: crate::DType,
}

#[instrument(skip(dev, size))]
pub fn create_wgpu_storage<T: ToU64>(dev : &WgpuDevice, dtype : crate::DType, size : T) -> WgpuStorage{
    let size = size.to_u64();
    let buffer;
    {
        let mut cache = dev.cache.lock().unwrap();
        buffer = cache.create_buffer_reference(size, true);
    } 
    return WgpuStorage::new(buffer,dev.clone(), dtype, size);
}

#[instrument(skip(dev, data))]
pub fn create_wgpu_storage_init<T: bytemuck::Pod>(dev : &WgpuDevice, dtype : crate::DType, data : &[T]) -> crate::Result<WgpuStorage>{
    let data : &[u8] = bytemuck::cast_slice(data);
    let size = data.len();
    let buffer;
    {
        if dev.configuration.flush_gpu_before_buffer_init{
            dev.flush_gpu_command()?;
        }
        let mut cache = dev.cache.lock().unwrap();
        buffer = cache.create_buffer_reference_init(dev, data,true);
    } 
    return Ok(WgpuStorage::new(buffer,dev.clone(), dtype, size as u64));
}


impl WgpuStorage {
    pub fn new(buffer: BufferReferenceId, wgpu_device: WgpuDevice, dtype: crate::DType, size : u64) -> Self {
        Self {
            buffer,
            wgpu_device,
            dtype: dtype,
            size
        }
    }

    pub async fn to_cpu_storage_async(&self) -> crate::Result<crate::CpuStorage> {
        match self.dtype {
            crate::DType::U32 => {
                return Ok(crate::CpuStorage::U32(
                    read_data_from_gpu_async(&self.wgpu_device, self.buffer.clone()).await?,
                ))
            }
            crate::DType::F32 => {
                return Ok(crate::CpuStorage::F32(
                    read_data_from_gpu_async(&self.wgpu_device, self.buffer.clone()).await?,
                ))
            }
            crate::DType::U8 => {
                return Ok(
                    crate::CpuStorage::U8(read_data_from_gpu_async(&self.wgpu_device, self.buffer.clone()).await?)
                )
            }
            _ => todo!(),
        }
    }

    pub fn get_length(&self) -> usize {

        return (self.size / 4) as usize; //f32
    }

    fn try_clone_layout(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, layout.shape().elem_count() * 4);
        self.copy_strided_src(&buffer_dest, 0, layout)?;
        return Ok(buffer_dest);
    }

    fn copy_strided_src(
        &self,
        dst: &WgpuStorage,
        dst_offset: usize,
        src_l: &crate::Layout,
    ) -> crate::Result<()> {
        match src_l.contiguous_offsets() {
            Some((start, end)) => {
                let len = end - start;
                let to_copy = ((dst.size as usize / 4) - dst_offset).min(len);
                wgpu_functions::queue_copy(
                    self.device(),
                    dst.buffer,
                    self.buffer.clone(),
                    dst_offset,
                    start,
                    to_copy,
                    self.dtype
                )?;
            }
            None => {
                wgpu_functions::queue_copy_strided(self.device(), dst.buffer, self.buffer.clone(),  self.dtype, src_l, dst_offset as u32)?; 
            }
        }
        return Ok(());
    }

}

impl crate::backend::BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, self.size);
        wgpu_functions::queue_copy(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            0,
            0,
            (self.size / 4) as usize,
            self.dtype
        )?;

        return Ok(buffer_dest);
    }

    fn dtype(&self) -> crate::DType {
        return self.dtype;
    }

    fn device(&self) -> &Self::Device {
        return &self.wgpu_device;
    }

    #[cfg(target_arch = "wasm32")]
    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        panic!("Sync copy to CpuStorage is not allowed for WebGpu device in WebAssembly. First copy the date asynchronously to a CpuStorage"); //panic, so we get a stacktrace and see where we wanted to copy
        //return Err(crate::Error::WebGpu("Sync copy to CpuStorage is not allowed for WebGpu device in WebAssembly. First copy the date asynchronously to a CpuStorage".to_owned().into()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        return pollster::block_on(self.to_cpu_storage_async());
    }

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, layout.shape().elem_count() * 4);
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            UnaryOperation::Affine,
            mul as f32,
            add as f32,
            self.dtype,
            layout,
        )?;
        return Ok(buffer_dest);
    }

    fn powf(&self, layout: &crate::Layout, e: f64) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, layout.shape().elem_count() * 4);
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            UnaryOperation::PowScalar,
            e as f32,
            0.0,
            self.dtype,
            layout,
        )?;
        return Ok(buffer_dest);
    }

    fn elu(&self, layout: &crate::Layout, alpha: f64) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, layout.shape().elem_count() * 4);
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            UnaryOperation::Elu,
            alpha as f32,
            0.0,
            self.dtype,
            layout,
        )?;
        return Ok(buffer_dest);
    }

    fn reduce_op(
        &self,
        reduce_op: crate::op::ReduceOp,
        layout: &crate::Layout,
        reduce_dims: &[usize],
    ) -> crate::Result<Self> {
        let src_dims = layout.dims();
        let mut dst_dims = src_dims.to_vec();
        for &dim in reduce_dims.iter() {
            dst_dims[dim] = 1;
        }
        let dst_shape = Shape::from(dst_dims);
        let mut reduce_dims = reduce_dims.to_vec();

        fn calculate_stride(shape: &[usize]) -> Vec<usize> {
            // Reverse the shape vector and fold over it
            let mut strides = shape
                .iter()
                .rev()
                .scan(1, |state, &dim| {
                    let current_stride = *state;
                    *state *= dim;
                    Some(current_stride)
                })
                .collect::<Vec<usize>>();
            // Reverse the strides to get them in the correct order
            strides.reverse();
            strides
        }
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, dst_shape.elem_count() * 4);

        let op = match reduce_op {
            crate::op::ReduceOp::Sum => ReduceOperations::Sum,
            crate::op::ReduceOp::Min => ReduceOperations::Min,
            crate::op::ReduceOp::Max => ReduceOperations::Max,
            crate::op::ReduceOp::ArgMin => ReduceOperations::ArgMin,
            crate::op::ReduceOp::ArgMax => ReduceOperations::ArgMax,
        };

        // Sort the reduce_dims as they have to be processed from left to right when converting the
        // indexes.
        reduce_dims.sort();
        let mut start_reduce_dim = 0;
        let mut end_reduce_dim = 1;
        let mut current_shape = layout.shape().clone().into_dims();
        let input_stride = calculate_stride(&current_shape[..]);
        let mut current_buffer = None;

        let call_reduce = |output_buffer: BufferReferenceId,
                           output_size: u32,
                           start_reduce_dim: usize,
                           end_reduce_dim: usize,
                           reduce_dims: &Vec<usize>,
                           prev_buffer: BufferReferenceId,
                           current_shape: &Vec<usize>,
                           layout: &Layout|
         -> crate::Result<()> {
            let start_dim = reduce_dims[start_reduce_dim];
            let end_dim = reduce_dims[end_reduce_dim - 1];
            let output_to_start_shape_stride2 = src_dims[(end_dim + 1)..]
                .iter()
                .fold(1, |prev, c| prev * *c)
                as u32;

            let output_to_start_stride1;
            if let Some(index) = current_shape.iter().rposition(|c| *c != 1) {
                output_to_start_stride1 = input_stride[index] as u32;
            } else {
                //All Other Elements have a Shape of 1?
                output_to_start_stride1 = 1 as u32;
            }
            let output_to_start_stride2 =
                src_dims[start_dim..].iter().fold(1, |prev, c| prev * *c) as u32;
            let output_to_start_stride2 =
                output_to_start_stride2 - output_to_start_shape_stride2 * output_to_start_stride1;
            let reduction_length = src_dims[start_dim..(end_dim + 1)]
                .iter()
                .fold(1, |prev, c| prev * *c);
            let stride_reduction = *input_stride[start_dim..(end_dim + 1)].iter().min().unwrap();
            wgpu_functions::queue_reduce_from_buffer_op(
                self.device(),
                output_buffer,
                prev_buffer,
                op,
                self.dtype,
                layout,
                output_size,
                output_to_start_shape_stride2, //Multiply all Shapes after EndDim
                output_to_start_stride1,       //Find Stride of last dimension(that was not reduced)
                output_to_start_stride2, //(Multiply all Shapes from StartDim until end) - output_to_start_shape_stride2 * output_to_start_stride1
                reduction_length as u32,
                stride_reduction as u32, //length of elements to reduce per output
            )?;
            return Ok(());
        };

        loop {
            if end_reduce_dim < reduce_dims.len() {
                if reduce_dims[end_reduce_dim] == reduce_dims[end_reduce_dim - 1] + 1 {
                    //the current end, is handled for the same block
                    end_reduce_dim += 1;
                } else {
                    let start_dim = reduce_dims[start_reduce_dim];
                    let end_dim = reduce_dims[end_reduce_dim - 1];

                    let l = Layout::contiguous(Shape::from_dims(&current_shape));

                    for i in start_dim..(end_dim + 1) {
                        current_shape[i] = 1;
                    }

                    let output_count = current_shape.iter().fold(1, |prev, c| prev * c);
                    

                    let mut cache = self.device().cache.lock().unwrap();
                    let buffer_temp = cache.create_buffer_reference(output_count * 4, false);

                    let (prev_buffer, l) = match current_buffer {
                        Some(buffer) => (buffer, &l),
                        None => (self.buffer.clone(), layout),
                    };

                    call_reduce(
                        buffer_temp.clone(),
                        output_count as u32,
                        start_reduce_dim,
                        end_reduce_dim,
                        &reduce_dims,
                        prev_buffer,
                        &current_shape,
                        l,
                    )?;

                    current_buffer = Some(buffer_temp);

                    start_reduce_dim = end_reduce_dim;
                    end_reduce_dim += 1;
                }
            } else {
                //end was outside of range,
                let start_dim = reduce_dims[start_reduce_dim];
                let end_dim = reduce_dims[end_reduce_dim - 1];

                let l = Layout::contiguous(Shape::from_dims(&current_shape));

                for i in start_dim..(end_dim + 1) {
                    current_shape[i] = 1;
                }
                let (prev_buffer, l) = match current_buffer {
                    Some(buffer) => (buffer, &l),
                    None => (self.buffer.clone(), layout),
                };

                call_reduce(
                    buffer_dest.buffer.clone(),
                    dst_shape.elem_count() as u32,
                    start_reduce_dim,
                    end_reduce_dim,
                    &reduce_dims,
                    prev_buffer,
                    &current_shape,
                    l,
                )?;

                break;
            }
        }
        return Ok(buffer_dest);
    }

    fn cmp(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_size = ((lhs_l.shape().elem_count() + 3) / 4) * 4; //TODO: get next divisible by 4
        let buffer_dest = create_wgpu_storage(self.device(), crate::DType::U8, buffer_size);
       
        let op2 = match op {
            crate::op::CmpOp::Eq => CmpOperation::Eq,
            crate::op::CmpOp::Ne => CmpOperation::Ne,
            crate::op::CmpOp::Le => CmpOperation::Le,
            crate::op::CmpOp::Ge => CmpOperation::Ge,
            crate::op::CmpOp::Lt => CmpOperation::Lt,
            crate::op::CmpOp::Gt => CmpOperation::Gt,
        };

        wgpu_functions::queue_cmp_buffer_from_buffer(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            rhs.buffer.clone(),
            op2,
            self.dtype,
            lhs_l,
            rhs_l,
        )?;
        return Ok(buffer_dest);
    }

    fn to_dtype(&self, layout: &crate::Layout, dtype: crate::DType) -> crate::Result<Self> {
        match (self.dtype, dtype) {
            (DType::F32, DType::F32) => self.try_clone_layout(layout),
            (DType::U32, DType::U32) => self.try_clone_layout(layout),
            (DType::U32, DType::F32) => {
                let buffer_dest = create_wgpu_storage(self.device(), DType::F32, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_u32_to_f32(
                    self.device(),
                    buffer_dest.buffer.clone(),
                    self.buffer.clone(),
                    layout,
                )?;
                Ok(buffer_dest)
            }
            (DType::U8, DType::F32) => {
                let buffer_dest = create_wgpu_storage(self.device(), DType::F32, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_u8_to_f32(
                    self.device(),
                    buffer_dest.buffer.clone(),
                    self.buffer.clone(),
                    layout,
                )?;
                Ok(buffer_dest)
            }
            (DType::F32, DType::U32) => {
                let buffer_dest = create_wgpu_storage(self.device(), DType::U32, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_f32_to_u32(
                    self.device(),
                    buffer_dest.buffer.clone(),
                    self.buffer.clone(),
                    layout,
                )?;
                Ok(buffer_dest)
            },
            (DType::F32, DType::U8) => {
                if !layout.is_contiguous(){
                    panic!("conversion from {:?} to {:?} not suported for non contiguous matrix", self.dtype, dtype);
                }
                let buffer_dest = create_wgpu_storage(self.device(), DType::U8, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_f32_to_u8(
                    self.device(),
                    buffer_dest.buffer.clone(),
                    self.buffer.clone(),
                    layout.start_offset() as u32,
                    layout.shape().elem_count() as u32
                )?;
                Ok(buffer_dest)
            },
            (DType::U32, DType::U8) => {
                if !layout.is_contiguous(){
                    panic!("conversion from {:?} to {:?} not suported for non contiguous matrix", self.dtype, dtype);
                }
                let buffer_dest = create_wgpu_storage(self.device(), DType::U8, layout.shape().elem_count() * 4);
                wgpu_functions::queue_convert_u32_to_u8(
                    self.device(),
                    buffer_dest.buffer.clone(),
                    self.buffer.clone(),
                    layout.start_offset() as u32,
                    layout.shape().elem_count() as u32
                )?;
                Ok(buffer_dest)
            }
            _ => panic!("conversion from {:?} to {:?} not suported on wgpu", self.dtype, dtype),
        }
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, layout.shape().elem_count() * 4);

        let op = match B::NAME {
            "gelu" => UnaryOperation::Gelu,
            "erf" => UnaryOperation::Erf,
            "silu" => UnaryOperation::SiLu,
            "ceil" => UnaryOperation::Ceil,
            "floor" => UnaryOperation::Floor,
            "round" => UnaryOperation::Round,
            "gelu_erf" => UnaryOperation::GeluErf,
            "sign" => UnaryOperation::Sign,
            "abs" => UnaryOperation::Abs,

            "exp" => UnaryOperation::Exp,
            "log" => UnaryOperation::Log,
            "sin" => UnaryOperation::Sin,
            "cos" => UnaryOperation::Cos,
            "neg" => UnaryOperation::Neg,
            "recip" => UnaryOperation::Inverse,
            "sqr" => UnaryOperation::Square,
            "sqrt" => UnaryOperation::Sqrt,
            "tanh" => UnaryOperation::Tanh,
            "relu" => UnaryOperation::Relu,
            "sigmoid" => UnaryOperation::Sigmoid,
            _ => {
                panic!("Operation {} is not supported on wgpu", B::NAME)
            }
        };
        wgpu_functions::queue_unary_from_buffer_op(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            op,
            0.0,
            0.0,
            self.dtype,
            layout,
        )?;
        return Ok(buffer_dest);
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &crate::Layout,
        rhs_layout: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, lhs_layout.shape().elem_count() * 4);

        let op = match B::NAME {
            "add" => BinaryOperation::Add,
            "sub" => BinaryOperation::Minus,
            "mul" => BinaryOperation::Mult,
            "div" => BinaryOperation::Div,
            "minimum" => BinaryOperation::Min,
            "maximum" => BinaryOperation::Max,
            _ => {
                panic!("Operation {} is not supported on wgpu", B::NAME)
            }
        };

        wgpu_functions::queue_binary_buffer_from_buffer(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            rhs.buffer.clone(),
            op,
            self.dtype,
            lhs_layout,
            rhs_layout,
        )?;
        return Ok(buffer_dest);
    }

    fn where_cond(
        &self,
        input_layout : &crate::Layout,
        t: &Self, //true values
        t_layout: &crate::Layout,
        f: &Self, //false values
        f_layout: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), t.dtype, input_layout.shape().elem_count() * 4);

        wgpu_functions::where_cond::queue_where_cond_u32(self.device(), buffer_dest.buffer.clone(), self.buffer.clone(), t.buffer.clone(), f.buffer.clone(), input_layout, t_layout, f_layout, t.dtype)?;
        return Ok(buffer_dest);
    }

    fn conv1d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, (params.b_size * params.c_out * params.l_out()) * 4);

        wgpu_functions::queue_conv1d(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            kernel.buffer.clone(),
            self.dtype,
            params,
            l,
            kernel_l,
        )?;
        return Ok(buffer_dest);
    }

    fn conv_transpose1d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype,  (params.b_size * params.c_out * params.l_out()) * 4);
        wgpu_functions::queue_conv1d_transpose(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            kernel.buffer.clone(),
            self.dtype,
            params,
            l,
            kernel_l,
        )?;
        return Ok(buffer_dest);
    }

    fn conv2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype,   (params.b_size * params.c_out * params.out_h() * params.out_w()) * 4);
        wgpu_functions::queue_conv2d(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            kernel.buffer.clone(),
            self.dtype,
            params,
            l,
            kernel_l,
        )?;
        return Ok(buffer_dest);
    }

    fn conv_transpose2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype,    (params.b_size * params.c_out * params.out_h() * params.out_w()) * 4);
        wgpu_functions::queue_conv2d_transpose(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            kernel.buffer.clone(),
            self.dtype,
            params,
            l,
            kernel_l,
        )?;
        return Ok(buffer_dest);
    }

    fn avg_pool2d(
        &self,
        layout: &crate::Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> crate::Result<Self> {
        
        let (b, c, h, w) = layout.shape().dims4()?;
        let h_out = (h - kernel_size.1) / stride.1 + 1;
        let w_out = (w - kernel_size.0) / stride.0 + 1;

        let buffer_dest = create_wgpu_storage(self.device(), self.dtype,     (b * c * h_out * w_out) * 4);
    
        wgpu_functions::queue_avg_pool2d(self.device(),  buffer_dest.buffer.clone(), self.buffer.clone(),layout, self.dtype(), kernel_size, stride)?;
          
        return Ok(buffer_dest);
    }

    fn max_pool2d(
        &self,
        layout: &crate::Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> crate::Result<Self> {

        let (b, c, h, w) = layout.shape().dims4()?;
        let h_out = (h - kernel_size.1) / stride.1 + 1;
        let w_out = (w - kernel_size.0) / stride.0 + 1;

        let buffer_dest = create_wgpu_storage(self.device(), self.dtype,     (b * c * h_out * w_out) * 4);
    
        wgpu_functions::queue_max_pool2d(self.device(),  buffer_dest.buffer.clone(), self.buffer.clone(),layout, self.dtype(), kernel_size, stride)?;
          
        return Ok(buffer_dest);
    }

    fn upsample_nearest1d(&self, layout: &crate::Layout, target_size: usize) -> crate::Result<Self> {
        let (b, c, _) = layout.shape().dims3()?;

        let buffer_dest = create_wgpu_storage(self.device(), self.dtype,     (b * c * target_size) * 4);
    
        wgpu_functions::queue_upsample1d(self.device(),  buffer_dest.buffer.clone(), self.buffer.clone(),layout, self.dtype(), target_size)?;
          
        return Ok(buffer_dest);
    }

    fn upsample_nearest2d(&self, layout: &crate::Layout, target_size_y: usize, target_size_x: usize) -> crate::Result<Self> {
        let (b, c, _, _) = layout.shape().dims4()?;

        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, (b * c * target_size_x * target_size_y) * 4);
    
        wgpu_functions::queue_upsample2d(self.device(),  buffer_dest.buffer.clone(), self.buffer.clone(),layout, self.dtype(), (target_size_y, target_size_x))?;
          
        return Ok(buffer_dest);
    }

    fn gather(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        d: usize,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, (indexes_l.shape().elem_count()) * 4);

        wgpu_functions::queue_gather(self.device(),  buffer_dest.buffer.clone(), self.buffer.clone(),indexes.buffer.clone(), self.dtype(), l, indexes_l, d)?;
          
        return Ok(buffer_dest);
    }

    fn scatter_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, (l.shape().elem_count()) * 4);

        self.copy_strided_src( &buffer_dest, 0, l)?;

        wgpu_functions::queue_scatter_add_inplace(self.device(),  buffer_dest.buffer.clone(),indexes.buffer.clone(), source.buffer.clone(), self.dtype(), &Layout::contiguous(l.shape().clone()), indexes_l, source_l, d)?;
          
        return Ok(buffer_dest);
    }

    fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
        d: usize,
    ) -> crate::Result<Self> {
        let mut new_shape = lhs_l.shape().clone().into_dims();
        new_shape[d] = rhs_l.shape().elem_count();
        let new_shape = Shape::from_dims(&new_shape[..]);

        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, (new_shape.elem_count()) * 4);

        wgpu_functions::queue_index_select(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            rhs.buffer.clone(),
            self.dtype,
            lhs_l,
            rhs_l,
            d,
        )?;
        return Ok(buffer_dest);
    }

    fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> crate::Result<Self> {
        
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, (l.shape().elem_count()) * 4);

        self.copy_strided_src( &buffer_dest, 0, l)?;

        wgpu_functions::queue_index_add_inplace(self.device(),  buffer_dest.buffer.clone(),indexes.buffer.clone(), source.buffer.clone(), self.dtype(), &Layout::contiguous(l.shape().clone()), indexes_l, source_l, d)?;
          
        return Ok(buffer_dest);
    }

    fn matmul(
        &self,
        rhs: &Self,
        (batching, m, n, k): (usize, usize, usize, usize),
        layout1: &crate::Layout,
        layout2: &crate::Layout,
    ) -> crate::Result<Self> {
        let buffer_dest = create_wgpu_storage(self.device(), self.dtype, batching * (m * n) * 4);

        wgpu_functions::queue_matmul_buffer(
            self.device(),
            buffer_dest.buffer.clone(),
            self.buffer.clone(),
            rhs.buffer.clone(),
            SGEMMParams::new(batching, m, k, n),
            0,
            layout1,
            layout2,
            self.dtype,
        )?;
        return Ok(buffer_dest);
    }

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &crate::Layout,
    ) -> crate::Result<()> {
        return self.copy_strided_src(&dst, dst_offset, src_l);
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> crate::Result<()> {
        wgpu_functions::queue_copy2d(
            self.device(),
            dst.buffer.clone(),
            self.buffer.clone(),
            self.dtype,
            d1 as u32,
            d2 as u32,
            src_stride1 as u32,
            dst_stride1 as u32,
            src_offset as u32,
            dst_offset as u32
        )?;
        Ok(())
    }
}


impl Drop for WgpuStorage{
    fn drop(&mut self) {
        let mut cache = self.device().cache.lock().unwrap();
        cache.buffer_reference.queue_for_deletion(&self. buffer);
        // if let Some(reference) = cache.buffer_reference.get_mut(&self.buffer){
        //     reference.set_referenced_by_candle_storage(false);
        // }
    }
}