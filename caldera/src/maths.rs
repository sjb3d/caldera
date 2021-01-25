use std::convert::TryInto;
use std::ops::Mul;
pub use ultraviolet::*;

pub trait AsSigned {
    type Output;
    fn as_signed(&self) -> Self::Output;
}

impl AsSigned for UVec2 {
    type Output = IVec2;
    fn as_signed(&self) -> Self::Output {
        IVec2::new(self.x as i32, self.y as i32)
    }
}

pub trait AsFloat {
    type Output;
    fn as_float(&self) -> Self::Output;
}

impl AsFloat for UVec2 {
    type Output = Vec2;
    fn as_float(&self) -> Self::Output {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

pub trait DivRoundUp {
    fn div_round_up(&self, divisor: u32) -> Self;
}

impl DivRoundUp for UVec2 {
    fn div_round_up(&self, divisor: u32) -> Self {
        (*self + Self::broadcast(divisor - 1)) / divisor
    }
}

pub trait IntoTransposedTransform {
    fn into_transposed_transform(&self) -> [f32; 12];
}

impl IntoTransposedTransform for Similarity3 {
    fn into_transposed_transform(&self) -> [f32; 12] {
        self.into_homogeneous_matrix().transposed().as_slice()[..12]
            .try_into()
            .unwrap()
    }
}

pub struct Scale2Offset2 {
    pub scale: Vec2,
    pub offset: Vec2,
}

impl Scale2Offset2 {
    pub fn new(scale: Vec2, offset: Vec2) -> Self {
        Self { scale, offset }
    }

    pub fn into_homogeneous_matrix(&self) -> Mat3 {
        Mat3::new(
            Vec3::new(self.scale.x, 0.0, 0.0),
            Vec3::new(0.0, self.scale.y, 0.0),
            self.offset.into_homogeneous_point(),
        )
    }

    pub fn inversed(&self) -> Self {
        // y = a*x + b => x = (y - b)/a
        let scale_rcp = Vec2::broadcast(1.0) / self.scale;
        Scale2Offset2 {
            scale: scale_rcp,
            offset: -self.offset * scale_rcp,
        }
    }
}

impl Mul for Scale2Offset2 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Scale2Offset2) -> Self::Output {
        // a(b(v)) = a.s*(b.s*v + b.o) + a.o
        Scale2Offset2 {
            scale: self.scale * rhs.scale,
            offset: self.scale.mul_add(rhs.offset, self.offset),
        }
    }
}

impl Mul<Vec2> for Scale2Offset2 {
    type Output = Vec2;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Vec2) -> Self::Output {
        self.scale.mul_add(rhs, self.offset)
    }
}
