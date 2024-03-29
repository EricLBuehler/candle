use crate::{shape::Dim, Error, Result, Shape, Tensor};

impl Tensor {
    /// Concatenates two or more tensors along a particular dimension.
    ///
    /// All tensors must of the same rank, and the output will have
    /// the same rank
    ///
    /// ```rust
    /// # use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = Tensor::cat(&[&a, &b], 0)?;
    /// assert_eq!(c.shape().dims(), &[4, 3]);
    ///
    /// let c = Tensor::cat(&[&a, &b], 1)?;
    /// assert_eq!(c.shape().dims(), &[2, 6]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn cat<A: AsRef<Tensor>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let dim = dim.to_index(arg0.shape(), "cat")?;
        for arg in args {
            arg.as_ref().check_dim(dim, "cat")?;
        }
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg0.rank() != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: arg0.rank(),
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx != dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
        }
        let all_contiguous = args.iter().all(|v| v.as_ref().is_contiguous());
        if all_contiguous {
            Self::cat_contiguous(args, dim)
        } else if dim == 0 {
            Self::cat0(args)
        } else {
            let args: Vec<Tensor> = args
                .iter()
                .map(|a| a.as_ref().transpose(0, dim))
                .collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    fn cat0<A: AsRef<Tensor>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
                if dim_idx != 0 && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
            let next_offset = offsets.last().unwrap() + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = crate::op::BackpropOp::new(args, |args| crate::op::Op::Cat(args, 0));
        let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage()
                .copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(crate::tensor::from_storage(storage, shape, op, false))
    }

    fn cat_contiguous<A: AsRef<Tensor>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            Err(Error::OpRequiresAtLeastOneTensor { op: "cat" }.bt())?
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[dim] = 0;
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                }
                .bt())?
            }
            if arg.device().location() != device.location() {
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                }
                .bt())?
            }
            if rank != arg.rank() {
                Err(Error::UnexpectedNumberOfDims {
                    expected: rank,
                    got: arg.rank(),
                    shape: arg.shape().clone(),
                }
                .bt())?
            }
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == dim {
                    cat_dims[dim] += v2;
                }
                if dim_idx != dim && v1 != v2 {
                    Err(Error::ShapeMismatchCat {
                        dim: dim_idx,
                        first_shape: arg0.shape().clone(),
                        n: arg_idx + 1,
                        nth_shape: arg.shape().clone(),
                    }
                    .bt())?
                }
            }
        }
        let cat_target_dim_len = cat_dims[dim];
        let block_size: usize = cat_dims.iter().skip(1 + dim).product();
        let shape = Shape::from(cat_dims);
        let op = crate::op::BackpropOp::new(args, |args| crate::op::Op::Cat(args, dim));
        let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
        let mut dst_o = 0;
        for arg in args.iter() {
            let arg = arg.as_ref();
            let arg_dims = arg.shape().dims();
            let d1: usize = arg_dims.iter().take(dim).product();
            let d2 = block_size * arg_dims[dim];
            let dst_s = block_size * cat_target_dim_len;
            let src_o = arg.layout().start_offset();
            arg.storage().copy2d(
                &mut storage,
                d1,
                d2,
                /* src_s */ d2,
                dst_s,
                src_o,
                dst_o,
            )?;
            dst_o += d2;
        }
        Ok(crate::tensor::from_storage(storage, shape, op, false))
    }
}