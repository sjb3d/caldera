use caldera::*;

fn lerp_mat3(a: &Mat3, b: &Mat3, t: f32) -> Mat3 {
    (*a) * (1.0 - t) + (*b) * t
}

pub fn derive_matrices() {
    // reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    let xyz_from_rec709 = Mat3::new(
        Vec3::new(0.4124564, 0.2126729, 0.0193339),
        Vec3::new(0.3575761, 0.7151522, 0.1191920),
        Vec3::new(0.1804375, 0.0721750, 0.9503041),
    );

    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    let d60_from_d65 = Mat3::new(
        Vec3::new(0.987224, -0.00759836, 0.00307257),
        Vec3::new(-0.00611327, 1.00186, -0.00509595),
        Vec3::new(0.0159533, 0.00533002, 1.08168),
    )
    .inversed();

    // reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
    let ap1_from_xyz = Mat3::new(
        Vec3::new(1.6410233797, -0.6636628587, 0.0117218943),
        Vec3::new(-0.3248032942, 1.6153315917, -0.0082844420),
        Vec3::new(-0.2364246952, 0.0167563477, 0.9883948585),
    );

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
