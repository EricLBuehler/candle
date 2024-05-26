use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, enqueue, MatrixLayout};


#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaConvert {
    input1_layout: MatrixLayout,
}



pub fn queue_convert_u32_to_f32(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let meta = MetaConvert {
        input1_layout: MatrixLayout::from_layout(&input_layout),
    };

    let pipeline = dev.get_pipeline(super::Shader::Convert(crate::DType::U32), Pipelines::ConvertU32ToF32)?;
    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
    enqueue(
        dev,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        &format!("u32_to_f32"),
    );
    return Ok(());
}

pub fn queue_convert_f32_to_u32(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let meta = MetaConvert {
        input1_layout: MatrixLayout::from_layout(&input_layout),
    };

    let pipeline = dev.get_pipeline(super::Shader::Convert(crate::DType::F32), Pipelines::ConvertF32ToU32)?;

    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
    enqueue(
        dev,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        &format!("f32_to_u32"),
    );
    return Ok(());
}