#[derive(Clone, Debug)]
pub enum TensorPool {
    #[cfg(feature = "metal")]
    Metal(crate::metal_backend::MetalTensorPool),
}

impl TensorPool {
    #[cfg(feature = "metal")]
    pub fn from_metal(pool: crate::metal_backend::MetalTensorPool) -> Self {
        Self::Metal(pool)
    }

    #[cfg(feature = "metal")]
    pub fn as_metal(&self) -> Option<&crate::metal_backend::MetalTensorPool> {
        match self {
            Self::Metal(pool) => Some(pool),
        }
    }

    #[cfg(feature = "metal")]
    pub fn into_metal(self) -> crate::metal_backend::MetalTensorPool {
        match self {
            Self::Metal(pool) => pool,
        }
    }
}
