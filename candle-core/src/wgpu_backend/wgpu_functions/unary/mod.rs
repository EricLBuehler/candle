use rand::RngCore;
use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input0, create_bind_group_input1, enqueue, MatrixLayout};

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaUnary {
    input_layout: MatrixLayout,
    operation: u32,
    scalar1: f32,
    scalar2: f32,
    seed: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaUnaryContiguous {
    offset: u32,
    length: u32,
    operation: u32,
    scalar1: f32, //optionally scalar value
    scalar2: f32,
    seed: u32,
}

#[derive(Copy, Clone, Debug)]
//#[allow(dead_code)]
pub enum UnaryOperation {
    SetZero = 0,
    SetOne = 1,
    IncOne = 2,
    DecOne = 3,
    Identity = 4,
    Square = 5,
    Affine = 6,
    Abs = 7,
    Acos = 8,
    Acosh = 9,
    Asin = 10,
    Asinh = 11,
    Atan = 12,
    Atanh = 13,
    Ceil = 14,
    Cos = 15,
    Cosh = 16,
    Deg = 17,
    Exp = 21,
    Floor = 22,
    Fract = 23,
    InverseSqrt = 24,
    Log = 25,
    Log2 = 26,
    Rad = 27,
    Sign = 28,
    Sin = 29,
    Sinh = 31,
    Sqrt = 32,
    Tan = 33,
    Tanh = 34,
    Trunc = 35,
    BinaryStep = 36,
    Sigmoid = 37,
    Relu = 38,
    Softplus = 39,
    LeakyRelu = 40,
    SiLu = 41,
    Gassian = 42,

    Neg = 45,
    Inverse = 46,
    RandNormal = 47,
    RandUniform = 48,
    Gelu = 49,
    Round = 50,

    Elu = 52,
    AddScalar = 101,
    MultScalar = 102,
    MinusScalar = 103,
    DivScalar = 104,
    MaxScalar = 105,
    MinScalar = 106,
    PowScalar = 107,
}

pub fn queue_unary_inplace_op(
    dev: &WgpuDevice,
    buffer: &Buffer,
    op: UnaryOperation,
    scalar1: f32,
    scalar2: f32,
    dtype: crate::DType,
    layout: crate::Layout,
) -> crate::Result<()> {
    let meta = MetaUnary {
        operation: op as u32,
        scalar1,
        scalar2,
        input_layout: MatrixLayout::from_layout(&layout),
        seed: dev.rand_state.lock().unwrap().next_u32(),
    };
    let pipeline = dev.get_pipeline(super::Shader::Unary(dtype), Pipelines::UnaryInplace)?;

    let bind_group = create_bind_group_input0(dev, pipeline.clone(), meta, buffer);
    enqueue(
        dev,
        pipeline,
        bind_group,
        layout.shape().elem_count() as u32,
        &format!("unary inplace op:{:?}, dtype:{:?}", op, Pipelines::UnaryInplace),
    );
    return Ok(());
}

pub fn queue_unary_from_buffer_op(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    op: UnaryOperation,
    scalar1: f32,
    scalar2: f32,
    dtype: crate::DType,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    if input_layout.is_contiguous() {
        let meta = MetaUnaryContiguous {
            operation: op as u32,
            scalar1,
            scalar2,
            seed: dev.rand_state.lock().unwrap().next_u32(),
            offset: input_layout.start_offset() as u32,
            length: input_layout.shape().elem_count() as u32,
        };

        let pipeline = dev.get_pipeline(super::Shader::Unary(dtype), Pipelines::UnaryFromBufferContiguous)?;

        let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
        enqueue(
            dev,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            &format!(
                "unary op:{:?}, dtype:{:?}, pipeline:{:?}",
                op, dtype, Pipelines::UnaryFromBufferContiguous
            ),
        );
    } else {
        let meta = MetaUnary {
            operation: op as u32,
            scalar1,
            scalar2,
            input_layout: MatrixLayout::from_layout(&input_layout),
            seed: dev.rand_state.lock().unwrap().next_u32(),
        };

        let pipeline = dev.get_pipeline(super::Shader::Unary(dtype), Pipelines::UnaryFromBuffer)?;
        let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
        enqueue(
            dev,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            &format!(
                "unary op:{:?}, dtype:{:?}, pipeline:{:?}",
                op, dtype, Pipelines::UnaryFromBuffer
            ),
        );
    }
    return Ok(());
}
