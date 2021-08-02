use crate::maths::*;

fn lerp_mat3(a: &Mat3, b: &Mat3, t: f32) -> Mat3 {
    (*a) * (1.0 - t) + (*b) * t
}

pub trait Gamma {
    fn into_linear(self) -> Self;
    fn into_gamma(self) -> Self;
}

impl Gamma for Vec3 {
    fn into_linear(self) -> Self {
        self.map(|x: f32| {
            if x < 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        })
    }

    fn into_gamma(self) -> Self {
        self.map(|x: f32| {
            if x < 0.0031308 {
                x * 12.92
            } else {
                (x.powf(1.0 / 2.4) * 1.055) - 0.055
            }
        })
    }
}

// assumes Rec709 primaries
pub trait Luminance {
    fn luminance(&self) -> f32;
}

impl Luminance for Vec3 {
    #[allow(clippy::excessive_precision)]
    fn luminance(&self) -> f32 {
        self.dot(Vec3::new(0.2126729, 0.7151522, 0.0721750))
    }
}

#[allow(clippy::excessive_precision)]
pub const fn xyz_from_rec709_matrix() -> Mat3 {
    // reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    Mat3::new(
        Vec3::new(0.4124564, 0.2126729, 0.0193339),
        Vec3::new(0.3575761, 0.7151522, 0.1191920),
        Vec3::new(0.1804375, 0.0721750, 0.9503041),
    )
}
#[allow(clippy::excessive_precision)]
pub const fn rec709_from_xyz_matrix() -> Mat3 {
    // reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    Mat3::new(
        Vec3::new(3.2404542, -0.9692660, 0.0556434),
        Vec3::new(-1.5371385, 1.8760108, -0.2040259),
        Vec3::new(-0.4985314, 0.0415560, 1.0572252),
    )
}

#[allow(clippy::excessive_precision)]
pub const fn ap1_from_xyz_matrix() -> Mat3 {
    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    Mat3::new(
        Vec3::new(1.6410233797, -0.6636628587, 0.0117218943),
        Vec3::new(-0.3248032942, 1.6153315917, -0.0082844420),
        Vec3::new(-0.2364246952, 0.0167563477, 0.9883948585),
    )
}
#[allow(clippy::excessive_precision)]
pub const fn xyz_from_ap1_matrix() -> Mat3 {
    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    Mat3::new(
        Vec3::new(0.6624541811, 0.2722287168, -0.0055746495),
        Vec3::new(0.1340042065, 0.6740817658, 0.0040607335),
        Vec3::new(0.1561876870, 0.0536895174, 1.0103391003),
    )
}

#[allow(clippy::excessive_precision)]
pub fn derive_aces_fit_matrices() {
    let xyz_from_rec709 = xyz_from_rec709_matrix();
    let d60_from_d65 = chromatic_adaptation_matrix(bradford_lms_from_xyz_matrix(), WhitePoint::D60, WhitePoint::D65);
    let ap1_from_xyz = ap1_from_xyz_matrix();

    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/rrt/RRT.ctl (RRT_SAT_MAT)
    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl (ODT_SAT_MAT)
    let luma_from_ap1_vec = Vec3::new(0.2722287168, 0.6740817658, 0.0536895174);
    let luma_from_ap1 = Mat3::new(luma_from_ap1_vec, luma_from_ap1_vec, luma_from_ap1_vec).transposed();
    let rrt_sat_factor = 0.96;
    let rrt_sat = lerp_mat3(&luma_from_ap1, &Mat3::identity(), rrt_sat_factor);
    let odt_sat_factor = 0.93;
    let odt_sat = lerp_mat3(&luma_from_ap1, &Mat3::identity(), odt_sat_factor);

    let acescg_from_rec709 = ap1_from_xyz * d60_from_d65 * xyz_from_rec709;
    let rec709_from_acescg = acescg_from_rec709.inversed();
    println!("acescg_from_rec709 = {:#?}", acescg_from_rec709);
    println!("rec709_from_acescg = {:#?}", rec709_from_acescg);

    // expected to match https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    let fit_from_rec709 = rrt_sat * acescg_from_rec709;
    let rec709_from_fit = rec709_from_acescg * odt_sat;
    println!("fit_from_rec709 = {:#?}", fit_from_rec709);
    println!("rec709_from_fit = {:#?}", rec709_from_fit);
}

pub trait ToXYZ {
    type Output;
    fn to_xyz(&self) -> Self::Output;
}

impl ToXYZ for Vec2 {
    type Output = Vec3;
    fn to_xyz(&self) -> Self::Output {
        Vec3::new(self.x / self.y, 1.0, (1.0 - self.x - self.y) / self.y)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WhitePoint {
    D60,
    D65,
    E,
    Custom { chroma: Vec2 },
}

impl Default for WhitePoint {
    fn default() -> Self {
        Self::D65
    }
}

impl WhitePoint {
    fn to_chroma(self) -> Vec2 {
        match self {
            Self::D60 => Vec2::new(0.32168, 0.33767),
            Self::D65 => Vec2::new(0.31270, 0.32900),
            Self::E => Vec2::new(0.3333, 0.3333),
            Self::Custom { chroma } => chroma,
        }
    }
}

#[allow(clippy::excessive_precision)]
pub const fn bradford_lms_from_xyz_matrix() -> Mat3 {
    Mat3::new(
        Vec3::new(0.8951000, -0.7502000, 0.0389000),
        Vec3::new(0.2664000, 1.7135000, -0.0685000),
        Vec3::new(-0.1614000, 0.0367000, 1.0296000),
    )
}

pub fn chromatic_adaptation_matrix(lms_from_xyz: Mat3, dst: WhitePoint, src: WhitePoint) -> Mat3 {
    let dst_lms = lms_from_xyz * dst.to_chroma().to_xyz();
    let src_lms = lms_from_xyz * src.to_chroma().to_xyz();
    lms_from_xyz.inversed() * Mat3::from_nonuniform_scale(dst_lms / src_lms) * lms_from_xyz
}
