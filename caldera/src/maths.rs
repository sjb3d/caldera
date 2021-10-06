use bytemuck::{Pod, Zeroable};
use std::ops::Mul;

pub use std::f32::consts::PI;
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

pub trait AsUnsigned {
    type Output;
    fn as_unsigned(&self) -> Self::Output;
}

impl AsUnsigned for IVec3 {
    type Output = UVec3;
    fn as_unsigned(&self) -> Self::Output {
        UVec3::new(self.x as u32, self.y as u32, self.z as u32)
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

pub trait MapMut {
    type Component;
    fn map_mut<F>(&self, f: F) -> Self
    where
        F: FnMut(Self::Component) -> Self::Component;
}

impl MapMut for UVec3 {
    type Component = u32;
    fn map_mut<F>(&self, mut f: F) -> Self
    where
        F: FnMut(Self::Component) -> Self::Component,
    {
        UVec3::new(f(self.x), f(self.y), f(self.z))
    }
}

pub trait IsNan {
    fn is_nan(&self) -> bool;
}

impl IsNan for Vec3 {
    fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

pub trait Saturated {
    fn saturated(&self) -> Self;
}

impl Saturated for Vec3 {
    fn saturated(&self) -> Self {
        self.clamped(Vec3::zero(), Vec3::one())
    }
}

pub trait DivRoundUp {
    fn div_round_up(&self, divisor: u32) -> Self;
}

impl DivRoundUp for u32 {
    fn div_round_up(&self, divisor: u32) -> Self {
        (*self + (divisor - 1)) / divisor
    }
}
impl DivRoundUp for UVec2 {
    fn div_round_up(&self, divisor: u32) -> Self {
        (*self + Self::broadcast(divisor - 1)) / divisor
    }
}

pub trait TransformVec3 {
    fn transform_vec3(&self, vec: Vec3) -> Vec3;
}

impl TransformVec3 for Isometry3 {
    fn transform_vec3(&self, vec: Vec3) -> Vec3 {
        self.rotation * vec
    }
}

impl TransformVec3 for Similarity3 {
    fn transform_vec3(&self, vec: Vec3) -> Vec3 {
        self.rotation * (self.scale * vec)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct Transform3 {
    pub cols: [Vec3; 4],
}

impl Transform3 {
    pub fn identity() -> Self {
        Transform3::new(Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z(), Vec3::zero())
    }

    pub fn new(col0: Vec3, col1: Vec3, col2: Vec3, col3: Vec3) -> Self {
        Self {
            cols: [col0, col1, col2, col3],
        }
    }

    pub fn extract_rotation(&self) -> Rotor3 {
        self.into_similarity().rotation
    }

    pub fn extract_translation(&self) -> Vec3 {
        self.cols[3]
    }

    pub fn into_similarity(self) -> Similarity3 {
        let scale = self.cols[0].cross(self.cols[1]).dot(self.cols[2]).powf(1.0 / 3.0);
        let rotation = Mat3::new(self.cols[0] / scale, self.cols[1] / scale, self.cols[2] / scale).into_rotor3();
        let translation = self.cols[3];
        Similarity3::new(translation, rotation, scale)
    }

    pub fn transposed(&self) -> TransposedTransform3 {
        TransposedTransform3::new(
            Vec4::new(self.cols[0].x, self.cols[1].x, self.cols[2].x, self.cols[3].x),
            Vec4::new(self.cols[0].y, self.cols[1].y, self.cols[2].y, self.cols[3].y),
            Vec4::new(self.cols[0].z, self.cols[1].z, self.cols[2].z, self.cols[3].z),
        )
    }
}

impl Default for Transform3 {
    fn default() -> Self {
        Transform3::identity()
    }
}

impl Mul for Transform3 {
    type Output = Transform3;

    fn mul(self, rhs: Transform3) -> Self::Output {
        Transform3::new(
            self.transform_vec3(rhs.cols[0]),
            self.transform_vec3(rhs.cols[1]),
            self.transform_vec3(rhs.cols[2]),
            self * rhs.cols[3],
        )
    }
}

impl Mul<Vec3> for Transform3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        self.cols[0] * rhs.x + self.cols[1] * rhs.y + self.cols[2] * rhs.z + self.cols[3]
    }
}

impl TransformVec3 for Transform3 {
    fn transform_vec3(&self, vec: Vec3) -> Vec3 {
        self.cols[0] * vec.x + self.cols[1] * vec.y + self.cols[2] * vec.z
    }
}

pub trait IntoTransform {
    fn into_transform(self) -> Transform3;
}

impl IntoTransform for Mat4 {
    fn into_transform(self) -> Transform3 {
        Transform3::new(
            self.cols[0].truncated(),
            self.cols[1].truncated(),
            self.cols[2].truncated(),
            self.cols[3].truncated(),
        )
    }
}
impl IntoTransform for Isometry3 {
    fn into_transform(self) -> Transform3 {
        self.into_homogeneous_matrix().into_transform()
    }
}
impl IntoTransform for Similarity3 {
    fn into_transform(self) -> Transform3 {
        self.into_homogeneous_matrix().into_transform()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct TransposedTransform3 {
    pub cols: [Vec4; 3],
}

impl TransposedTransform3 {
    fn new(col0: Vec4, col1: Vec4, col2: Vec4) -> Self {
        Self {
            cols: [col0, col1, col2],
        }
    }
}

pub trait FromPackedUnorm<T> {
    fn from_packed_unorm(p: T) -> Self;
}

impl FromPackedUnorm<u32> for Vec4 {
    fn from_packed_unorm(p: u32) -> Self {
        let x = ((p & 0xff) as f32) / 255.0;
        let y = (((p >> 8) & 0xff) as f32) / 255.0;
        let z = (((p >> 16) & 0xff) as f32) / 255.0;
        let w = (((p >> 24) & 0xff) as f32) / 255.0;
        Vec4::new(x, y, z, w)
    }
}

pub trait IntoPackedUnorm<T> {
    fn into_packed_unorm(self) -> T;
}

impl IntoPackedUnorm<u16> for Vec2 {
    fn into_packed_unorm(self) -> u16 {
        let x = (self.x.max(0.0).min(1.0) * 255.0).round() as u8 as u16;
        let y = (self.y.max(0.0).min(1.0) * 255.0).round() as u8 as u16;
        x | (y << 8)
    }
}

impl IntoPackedUnorm<u32> for Vec2 {
    fn into_packed_unorm(self) -> u32 {
        let x = (self.x.max(0.0).min(1.0) * 65535.0).round() as u16 as u32;
        let y = (self.y.max(0.0).min(1.0) * 65535.0).round() as u16 as u32;
        x | (y << 16)
    }
}

impl IntoPackedUnorm<u32> for Vec3 {
    fn into_packed_unorm(self) -> u32 {
        let x = (self.x.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        let y = (self.y.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        let z = (self.z.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        x | (y << 8) | (z << 16)
    }
}

impl IntoPackedUnorm<u32> for Vec4 {
    fn into_packed_unorm(self) -> u32 {
        let x = (self.x.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        let y = (self.y.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        let z = (self.z.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        let w = (self.w.max(0.0).min(1.0) * 255.0).round() as u8 as u32;
        x | (y << 8) | (z << 16) | (w << 24)
    }
}

pub trait IntoPackedSnorm<T> {
    fn into_packed_snorm(self) -> T;
}

impl IntoPackedSnorm<u32> for Vec2 {
    fn into_packed_snorm(self) -> u32 {
        let x = (self.x.max(-1.0).min(1.0) * 32767.0).round() as i16 as u16 as u32;
        let y = (self.y.max(-1.0).min(1.0) * 32767.0).round() as i16 as u16 as u32;
        x | (y << 16)
    }
}

pub trait SumElements {
    type Output;
    fn sum_elements(&self) -> Self::Output;
}

impl SumElements for Vec3 {
    type Output = f32;
    fn sum_elements(&self) -> Self::Output {
        self.x + self.y + self.z
    }
}

pub trait CopySign {
    fn copysign(&self, other: Self) -> Self;
}

impl CopySign for Vec2 {
    fn copysign(&self, other: Self) -> Self {
        Self::new(self.x.copysign(other.x), self.y.copysign(other.y))
    }
}

pub trait VecToOct {
    fn into_oct(self) -> Vec2;
}

impl VecToOct for Vec3 {
    fn into_oct(self) -> Vec2 {
        let p = self.xy() / self.abs().sum_elements();
        if self.z > 0.0 {
            p
        } else {
            (Vec2::broadcast(1.0) - Vec2::new(p.y, p.x).abs()).copysign(p)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Scale2Offset2 {
    pub scale: Vec2,
    pub offset: Vec2,
}

impl Scale2Offset2 {
    pub fn new(scale: Vec2, offset: Vec2) -> Self {
        Self { scale, offset }
    }

    pub fn into_homogeneous_matrix(self) -> Mat3 {
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
