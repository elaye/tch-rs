//! The different kind of elements supported in Torch.

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

/// The different kind of elements that a Tensor can hold.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
}

impl Kind {
    pub(super) fn c_int(self) -> libc::c_int {
        match self {
            Kind::Uint8 => 0,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 3,
            Kind::Int64 => 4,
            Kind::Half => 5,
            Kind::Float => 6,
            Kind::Double => 7,
            Kind::ComplexHalf => 8,
            Kind::ComplexFloat => 9,
            Kind::ComplexDouble => 10,
            Kind::Bool => 11,
        }
    }

    pub(super) fn of_c_int(v: libc::c_int) -> Kind {
        match v {
            0 => Kind::Uint8,
            1 => Kind::Int8,
            2 => Kind::Int16,
            3 => Kind::Int,
            4 => Kind::Int64,
            5 => Kind::Half,
            6 => Kind::Float,
            7 => Kind::Double,
            8 => Kind::ComplexHalf,
            9 => Kind::ComplexFloat,
            10 => Kind::ComplexDouble,
            11 => Kind::Bool,
            _ => panic!("unexpected kind {}", v),
        }
    }

    pub fn elt_size_in_bytes(self) -> usize {
        match self {
            Kind::Uint8 => 1,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 4,
            Kind::Int64 => 8,
            Kind::Half => 2,
            Kind::Float => 4,
            Kind::Double => 8,
            Kind::ComplexHalf => 4,
            Kind::ComplexFloat => 8,
            Kind::ComplexDouble => 16,
            Kind::Bool => 1,
        }
    }
}

pub const FLOAT_CPU: (Kind, crate::Device) = (Kind::Float, crate::Device::Cpu);
pub const DOUBLE_CPU: (Kind, crate::Device) = (Kind::Double, crate::Device::Cpu);
pub const INT64_CPU: (Kind, crate::Device) = (Kind::Int64, crate::Device::Cpu);

pub const FLOAT_CUDA: (Kind, crate::Device) = (Kind::Float, crate::Device::Cuda(0));
pub const DOUBLE_CUDA: (Kind, crate::Device) = (Kind::Double, crate::Device::Cuda(0));
pub const INT64_CUDA: (Kind, crate::Device) = (Kind::Int64, crate::Device::Cuda(0));

pub trait T {
    const KIND: Kind;
}

impl T for u8 {
    const KIND: Kind = Kind::Uint8;
}

impl T for i8 {
    const KIND: Kind = Kind::Int8;
}

impl T for i16 {
    const KIND: Kind = Kind::Int16;
}

impl T for i32 {
    const KIND: Kind = Kind::Int;
}

impl T for i64 {
    const KIND: Kind = Kind::Int64;
}

impl T for f32 {
    const KIND: Kind = Kind::Float;
}

impl T for f64 {
    const KIND: Kind = Kind::Double;
}

impl T for bool {
    const KIND: Kind = Kind::Bool;
}
