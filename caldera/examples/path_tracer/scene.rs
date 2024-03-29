use crate::prelude::*;
use bytemuck::Contiguous;
use caldera::prelude::*;
use ply_rs::{parser, ply};
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::Arc,
};
use strum::{EnumString, EnumVariantNames};

#[derive(Debug, Default)]
pub struct Transform {
    pub world_from_local: Similarity3,
}

impl Transform {
    pub fn new(world_from_local: Similarity3) -> Self {
        Self { world_from_local }
    }
}

#[derive(Debug)]
pub enum Geometry {
    TriangleMesh {
        positions: Vec<Vec3>,
        normals: Option<Vec<Vec3>>,
        uvs: Option<Vec<Vec2>>,
        indices: Vec<UVec3>,
        min: Vec3,
        max: Vec3,
        area: f32,
    },
    Quad {
        local_from_quad: Similarity3,
        size: Vec2,
    },
    Disc {
        local_from_disc: Similarity3,
        radius: f32,
    },
    Sphere {
        centre: Vec3,
        radius: f32,
    },
    #[allow(dead_code)]
    Mandelbulb {
        local_from_bulb: Similarity3,
    },
}

pub const MANDELBULB_RADIUS: f32 = 1.1;

#[derive(Debug)]
pub enum Reflectance {
    Checkerboard(Vec3), // HACK! TODO: proper shaders
    Constant(Vec3),
    Texture(PathBuf),
}

#[derive(Debug, Clone, Copy, Contiguous, Eq, PartialEq)]
#[repr(u32)]
pub enum Conductor {
    Aluminium,
    AluminiumAntimonide,
    Chromium,
    Copper,
    Iron,
    Lithium,
    Gold,
    Silver,
    TitaniumNitride,
    Tungsten,
    Vanadium,
    VanadiumNitride,
    Custom, // eta=2, k=0, TODO: proper custom spectra
}

#[derive(Debug, Clone, Copy)]
pub enum Surface {
    None,
    Diffuse,
    Mirror,
    SmoothDielectric,
    RoughDielectric { roughness: f32 },
    SmoothPlastic,
    RoughPlastic { roughness: f32 },
    RoughConductor { conductor: Conductor, roughness: f32 },
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, Contiguous, Eq, PartialEq, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum Illuminant {
    E,
    CornellBox,
    D65,
    F10,
}

impl Illuminant {
    pub fn white_point(&self) -> WhitePoint {
        match self {
            Self::E => WhitePoint::E,
            Self::CornellBox => unimplemented!(),
            Self::D65 => WhitePoint::D65,
            Self::F10 => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Emission {
    pub illuminant: Illuminant,
    pub intensity: Vec3,
}

impl Emission {
    pub fn new_uniform(intensity: Vec3) -> Self {
        Self {
            illuminant: Illuminant::E,
            intensity,
        }
    }
}

/*
    Eventually this would be some kind of graph/script that we can
    runtime compile to a GLSL callable shader.  The graph would read
    interpolated data from the geometry (e.g. texture coordinates)
    and uniform data from the instance (e.g. textures, constants)
    and produces a closure for the BRDF (and emitter if present).

    For now we just enumerate some fixed options for the result of
    this shader:

      * Reflectance is a constant colour or a texture read

      * Pick from a fixed set of surface models (some also have
        a roughness parameter.

      * Geometry can optionally emit a fixed colour (quads/spheres
        only for now).
*/
#[derive(Debug)]
pub struct Material {
    pub reflectance: Reflectance,
    pub surface: Surface,
    pub emission: Option<Emission>,
}

#[derive(Debug)]
pub struct Instance {
    pub transform_ref: TransformRef,
    pub geometry_ref: GeometryRef,
    pub material_ref: MaterialRef,
}

#[derive(Debug)]
pub enum Light {
    Dome {
        emission: Emission,
    },
    SolidAngle {
        emission: Emission,
        direction_ws: Vec3,
        solid_angle: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Camera {
    Pinhole {
        world_from_camera: Similarity3,
        fov_y: f32,
    },
    ThinLens {
        world_from_camera: Similarity3,
        fov_y: f32,
        aperture_radius: f32,
        focus_distance: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransformRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeometryRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstanceRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LightRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CameraRef(pub u32);

impl Instance {
    pub fn new(transform_ref: TransformRef, geometry_ref: GeometryRef, material_ref: MaterialRef) -> Self {
        Self {
            transform_ref,
            geometry_ref,
            material_ref,
        }
    }
}

pub type SharedScene = Arc<Scene>;

#[derive(Debug, Default)]
pub struct Scene {
    pub transforms: Vec<Transform>,
    pub geometries: Vec<Geometry>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
    pub lights: Vec<Light>,
    pub cameras: Vec<Camera>,
}

impl Scene {
    pub fn add_transform(&mut self, transform: Transform) -> TransformRef {
        let index = self.transforms.len();
        self.transforms.push(transform);
        TransformRef(index as u32)
    }

    pub fn add_geometry(&mut self, geometry: Geometry) -> GeometryRef {
        let index = self.geometries.len();
        self.geometries.push(geometry);
        GeometryRef(index as u32)
    }

    pub fn add_material(&mut self, material: Material) -> MaterialRef {
        let index = self.materials.len();
        self.materials.push(material);
        MaterialRef(index as u32)
    }

    pub fn add_instance(&mut self, instance: Instance) -> InstanceRef {
        let index = self.instances.len();
        self.instances.push(instance);
        InstanceRef(index as u32)
    }

    pub fn add_light(&mut self, light: Light) -> LightRef {
        let index = self.lights.len();
        self.lights.push(light);
        LightRef(index as u32)
    }

    pub fn add_camera(&mut self, camera: Camera) -> CameraRef {
        let index = self.cameras.len();
        self.cameras.push(camera);
        CameraRef(index as u32)
    }

    pub fn geometry_ref_iter(&self) -> impl Iterator<Item = GeometryRef> {
        (0..self.geometries.len()).map(|i| GeometryRef(i as u32))
    }

    pub fn material_ref_iter(&self) -> impl Iterator<Item = MaterialRef> {
        (0..self.materials.len()).map(|i| MaterialRef(i as u32))
    }

    pub fn instance_ref_iter(&self) -> impl Iterator<Item = InstanceRef> {
        (0..self.instances.len()).map(|i| InstanceRef(i as u32))
    }

    pub fn camera_ref_iter(&self) -> impl Iterator<Item = CameraRef> {
        (0..self.cameras.len()).map(|i| CameraRef(i as u32))
    }

    pub fn transform(&self, r: TransformRef) -> &Transform {
        self.transforms.get(r.0 as usize).unwrap()
    }

    pub fn geometry(&self, r: GeometryRef) -> &Geometry {
        self.geometries.get(r.0 as usize).unwrap()
    }

    pub fn material(&self, r: MaterialRef) -> &Material {
        self.materials.get(r.0 as usize).unwrap()
    }

    pub fn instance(&self, r: InstanceRef) -> &Instance {
        self.instances.get(r.0 as usize).unwrap()
    }

    pub fn camera(&self, r: CameraRef) -> &Camera {
        self.cameras.get(r.0 as usize).unwrap()
    }

    pub fn bake_unique_geometry(&mut self) {
        let mut instance_counts = vec![0u32; self.geometries.len()];
        for instance in self.instances.iter() {
            instance_counts[instance.geometry_ref.0 as usize] += 1;
        }
        let identity_ref = self.add_transform(Transform::default());
        for instance in self.instances.iter_mut() {
            if instance_counts[instance.geometry_ref.0 as usize] == 1 {
                let world_from_local = self
                    .transforms
                    .get(instance.transform_ref.0 as usize)
                    .unwrap()
                    .world_from_local;
                match self.geometries.get_mut(instance.geometry_ref.0 as usize).unwrap() {
                    Geometry::TriangleMesh { positions, normals, .. } => {
                        for pos in positions.iter_mut() {
                            *pos = world_from_local * *pos;
                        }
                        if let Some(normals) = normals {
                            for normal in normals.iter_mut() {
                                *normal = world_from_local.transform_vec3(*normal).normalized();
                            }
                        }
                    }
                    Geometry::Quad { local_from_quad, .. } => {
                        *local_from_quad = world_from_local * *local_from_quad;
                    }
                    Geometry::Disc { local_from_disc, .. } => {
                        *local_from_disc = world_from_local * *local_from_disc;
                    }
                    Geometry::Sphere { centre, radius } => {
                        *centre = world_from_local * *centre;
                        *radius *= world_from_local.scale.abs();
                    }
                    Geometry::Mandelbulb { local_from_bulb } => {
                        *local_from_bulb = world_from_local * *local_from_bulb;
                    }
                }
                instance.transform_ref = identity_ref;
            }
        }
    }
}

pub struct TriangleMeshBuilder {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<UVec3>,
    pub area: f32,
}

impl TriangleMeshBuilder {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            area: 0.0,
        }
    }

    pub fn with_quad(mut self, v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        let base = UVec3::broadcast(self.positions.len() as u32);
        let normal_vec0 = (v2 - v1).cross(v0 - v1);
        let normal_vec1 = (v0 - v3).cross(v2 - v3);
        let normal = (normal_vec0 + normal_vec1).normalized();
        self.positions.push(v0);
        self.positions.push(v1);
        self.positions.push(v2);
        self.positions.push(v3);
        self.normals.push(normal);
        self.normals.push(normal);
        self.normals.push(normal);
        self.normals.push(normal);
        self.uvs.push(Vec2::new(0.0, 0.0));
        self.uvs.push(Vec2::new(1.0, 0.0));
        self.uvs.push(Vec2::new(1.0, 1.0));
        self.uvs.push(Vec2::new(0.0, 1.0));
        self.indices.push(base + UVec3::new(0, 1, 2));
        self.indices.push(base + UVec3::new(2, 3, 0));
        self.area += 0.5 * (normal_vec0.mag() + normal_vec1.mag());
        self
    }

    pub fn build(self) -> Geometry {
        let mut min = Vec3::broadcast(f32::INFINITY);
        let mut max = Vec3::broadcast(-f32::INFINITY);
        for pos in self.positions.iter() {
            min = min.min_by_component(*pos);
            max = max.max_by_component(*pos);
        }
        Geometry::TriangleMesh {
            positions: self.positions,
            normals: Some(self.normals),
            uvs: Some(self.uvs),
            indices: self.indices,
            min,
            max,
            area: self.area,
        }
    }
}

macro_rules! spectrum_samples {
    ($(($w:literal, $v:literal)),+) => { [ $( ($w, $v), )+ ] }
}

// reference: https://www.graphics.cornell.edu/online/box/data.html
#[rustfmt::skip]
const CORNELL_BOX_WHITE_SAMPLES: &[(f32, f32)] = &spectrum_samples!(
    (400.0, 0.343),(404.0, 0.445),(408.0, 0.551),(412.0, 0.624),(416.0, 0.665),
    (420.0, 0.687),(424.0, 0.708),(428.0, 0.723),(432.0, 0.715),(436.0, 0.710),
    (440.0, 0.745),(444.0, 0.758),(448.0, 0.739),(452.0, 0.767),(456.0, 0.777),
    (460.0, 0.765),(464.0, 0.751),(468.0, 0.745),(472.0, 0.748),(476.0, 0.729),
    (480.0, 0.745),(484.0, 0.757),(488.0, 0.753),(492.0, 0.750),(496.0, 0.746),
    (500.0, 0.747),(504.0, 0.735),(508.0, 0.732),(512.0, 0.739),(516.0, 0.734),
    (520.0, 0.725),(524.0, 0.721),(528.0, 0.733),(532.0, 0.725),(536.0, 0.732),
    (540.0, 0.743),(544.0, 0.744),(548.0, 0.748),(552.0, 0.728),(556.0, 0.716),
    (560.0, 0.733),(564.0, 0.726),(568.0, 0.713),(572.0, 0.740),(576.0, 0.754),
    (580.0, 0.764),(584.0, 0.752),(588.0, 0.736),(592.0, 0.734),(596.0, 0.741),
    (600.0, 0.740),(604.0, 0.732),(608.0, 0.745),(612.0, 0.755),(616.0, 0.751),
    (620.0, 0.744),(624.0, 0.731),(628.0, 0.733),(632.0, 0.744),(636.0, 0.731),
    (640.0, 0.712),(644.0, 0.708),(648.0, 0.729),(652.0, 0.730),(656.0, 0.727),
    (660.0, 0.707),(664.0, 0.703),(668.0, 0.729),(672.0, 0.750),(676.0, 0.760),
    (680.0, 0.751),(684.0, 0.739),(688.0, 0.724),(692.0, 0.730),(696.0, 0.740),
    (700.0, 0.737)
);
#[rustfmt::skip]
const CORNELL_BOX_GREEN_SAMPLES: &[(f32, f32)] = &spectrum_samples!(
    (400.0, 0.092),(404.0, 0.096),(408.0, 0.098),(412.0, 0.097),(416.0, 0.098),
    (420.0, 0.095),(424.0, 0.095),(428.0, 0.097),(432.0, 0.095),(436.0, 0.094),
    (440.0, 0.097),(444.0, 0.098),(448.0, 0.096),(452.0, 0.101),(456.0, 0.103),
    (460.0, 0.104),(464.0, 0.107),(468.0, 0.109),(472.0, 0.112),(476.0, 0.115),
    (480.0, 0.125),(484.0, 0.140),(488.0, 0.160),(492.0, 0.187),(496.0, 0.229),
    (500.0, 0.285),(504.0, 0.343),(508.0, 0.390),(512.0, 0.435),(516.0, 0.464),
    (520.0, 0.472),(524.0, 0.476),(528.0, 0.481),(532.0, 0.462),(536.0, 0.447),
    (540.0, 0.441),(544.0, 0.426),(548.0, 0.406),(552.0, 0.373),(556.0, 0.347),
    (560.0, 0.337),(564.0, 0.314),(568.0, 0.285),(572.0, 0.277),(576.0, 0.266),
    (580.0, 0.250),(584.0, 0.230),(588.0, 0.207),(592.0, 0.186),(596.0, 0.171),
    (600.0, 0.160),(604.0, 0.148),(608.0, 0.141),(612.0, 0.136),(616.0, 0.130),
    (620.0, 0.126),(624.0, 0.123),(628.0, 0.121),(632.0, 0.122),(636.0, 0.119),
    (640.0, 0.114),(644.0, 0.115),(648.0, 0.117),(652.0, 0.117),(656.0, 0.118),
    (660.0, 0.120),(664.0, 0.122),(668.0, 0.128),(672.0, 0.132),(676.0, 0.139),
    (680.0, 0.144),(684.0, 0.146),(688.0, 0.150),(692.0, 0.152),(696.0, 0.157),
    (700.0, 0.159)
);
#[rustfmt::skip]
const CORNELL_BOX_RED_SAMPLES: &[(f32, f32)] = &spectrum_samples!(
    (400.0, 0.040),(404.0, 0.046),(408.0, 0.048),(412.0, 0.053),(416.0, 0.049),
    (420.0, 0.050),(424.0, 0.053),(428.0, 0.055),(432.0, 0.057),(436.0, 0.056),
    (440.0, 0.059),(444.0, 0.057),(448.0, 0.061),(452.0, 0.061),(456.0, 0.060),
    (460.0, 0.062),(464.0, 0.062),(468.0, 0.062),(472.0, 0.061),(476.0, 0.062),
    (480.0, 0.060),(484.0, 0.059),(488.0, 0.057),(492.0, 0.058),(496.0, 0.058),
    (500.0, 0.058),(504.0, 0.056),(508.0, 0.055),(512.0, 0.056),(516.0, 0.059),
    (520.0, 0.057),(524.0, 0.055),(528.0, 0.059),(532.0, 0.059),(536.0, 0.058),
    (540.0, 0.059),(544.0, 0.061),(548.0, 0.061),(552.0, 0.063),(556.0, 0.063),
    (560.0, 0.067),(564.0, 0.068),(568.0, 0.072),(572.0, 0.080),(576.0, 0.090),
    (580.0, 0.099),(584.0, 0.124),(588.0, 0.154),(592.0, 0.192),(596.0, 0.255),
    (600.0, 0.287),(604.0, 0.349),(608.0, 0.402),(612.0, 0.443),(616.0, 0.487),
    (620.0, 0.513),(624.0, 0.558),(628.0, 0.584),(632.0, 0.620),(636.0, 0.606),
    (640.0, 0.609),(644.0, 0.651),(648.0, 0.612),(652.0, 0.610),(656.0, 0.650),
    (660.0, 0.638),(664.0, 0.627),(668.0, 0.620),(672.0, 0.630),(676.0, 0.628),
    (680.0, 0.642),(684.0, 0.639),(688.0, 0.657),(692.0, 0.639),(696.0, 0.635),
    (700.0, 0.642)
);
#[rustfmt::skip]
pub const CORNELL_BOX_ILLUMINANT: RegularlySampledIlluminant = RegularlySampledIlluminant {
    samples: &[0.0, 8.0, 15.6, 18.4, 0.0],
    sample_norm: 1.0,
    wavelength_base: 400.0,
    wavelength_step_size: 100.0,
};

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum CornellBoxVariant {
    Original,
    DomeLight,
    Conductor,
}

#[allow(clippy::excessive_precision)]
pub fn create_cornell_box_scene(variant: &CornellBoxVariant) -> Scene {
    let mut scene = Scene::default();

    let floor = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.5528, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.5592),
                Vec3::new(0.5496, 0.0, 0.5592),
            )
            .build(),
    );
    let ceiling = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.556, 0.5488, 0.0),
                Vec3::new(0.556, 0.5488, 0.5592),
                Vec3::new(0.0, 0.5488, 0.5592),
                Vec3::new(0.0, 0.5488, 0.0),
            )
            .build(),
    );
    let grey_wall = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.5496, 0.0, 0.5592),
                Vec3::new(0.0, 0.0, 0.5592),
                Vec3::new(0.0, 0.5488, 0.5592),
                Vec3::new(0.556, 0.5488, 0.5592),
            )
            .build(),
    );
    let red_wall = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.5528, 0.0, 0.0),
                Vec3::new(0.5496, 0.0, 0.5592),
                Vec3::new(0.556, 0.5488, 0.5592),
                Vec3::new(0.556, 0.5488, 0.0),
            )
            .build(),
    );
    let green_wall = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.0, 0.0, 0.5592),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.5488, 0.0),
                Vec3::new(0.0, 0.5488, 0.5592),
            )
            .build(),
    );
    let short_block = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.130, 0.165, 0.065),
                Vec3::new(0.082, 0.165, 0.225),
                Vec3::new(0.240, 0.165, 0.272),
                Vec3::new(0.290, 0.165, 0.114),
            )
            .with_quad(
                Vec3::new(0.290, 0.0, 0.114),
                Vec3::new(0.290, 0.165, 0.114),
                Vec3::new(0.240, 0.165, 0.272),
                Vec3::new(0.240, 0.0, 0.272),
            )
            .with_quad(
                Vec3::new(0.130, 0.0, 0.065),
                Vec3::new(0.130, 0.165, 0.065),
                Vec3::new(0.290, 0.165, 0.114),
                Vec3::new(0.290, 0.0, 0.114),
            )
            .with_quad(
                Vec3::new(0.082, 0.0, 0.225),
                Vec3::new(0.082, 0.165, 0.225),
                Vec3::new(0.130, 0.165, 0.065),
                Vec3::new(0.130, 0.0, 0.065),
            )
            .with_quad(
                Vec3::new(0.240, 0.0, 0.272),
                Vec3::new(0.240, 0.165, 0.272),
                Vec3::new(0.082, 0.165, 0.225),
                Vec3::new(0.082, 0.0, 0.225),
            )
            .build(),
    );
    let tall_block = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.423, 0.330, 0.247),
                Vec3::new(0.265, 0.330, 0.296),
                Vec3::new(0.314, 0.330, 0.456),
                Vec3::new(0.472, 0.330, 0.406),
            )
            .with_quad(
                Vec3::new(0.423, 0.0, 0.247),
                Vec3::new(0.423, 0.330, 0.247),
                Vec3::new(0.472, 0.330, 0.406),
                Vec3::new(0.472, 0.0, 0.406),
            )
            .with_quad(
                Vec3::new(0.472, 0.0, 0.406),
                Vec3::new(0.472, 0.330, 0.406),
                Vec3::new(0.314, 0.330, 0.456),
                Vec3::new(0.314, 0.0, 0.456),
            )
            .with_quad(
                Vec3::new(0.314, 0.0, 0.456),
                Vec3::new(0.314, 0.330, 0.456),
                Vec3::new(0.265, 0.330, 0.296),
                Vec3::new(0.265, 0.0, 0.296),
            )
            .with_quad(
                Vec3::new(0.265, 0.0, 0.296),
                Vec3::new(0.265, 0.330, 0.296),
                Vec3::new(0.423, 0.330, 0.247),
                Vec3::new(0.423, 0.0, 0.247),
            )
            .build(),
    );

    let rgb_from_xyz = rec709_from_xyz_matrix();

    let white_reflectance = rgb_from_xyz
        * xyz_from_spectral_reflectance_sweep(
            CORNELL_BOX_WHITE_SAMPLES.iter().copied().into_sweep(),
            D65_ILLUMINANT.iter().into_sweep(),
        );
    let red_reflectance = rgb_from_xyz
        * xyz_from_spectral_reflectance_sweep(
            CORNELL_BOX_RED_SAMPLES.iter().copied().into_sweep(),
            D65_ILLUMINANT.iter().into_sweep(),
        );
    let green_reflectance = rgb_from_xyz
        * xyz_from_spectral_reflectance_sweep(
            CORNELL_BOX_GREEN_SAMPLES.iter().copied().into_sweep(),
            D65_ILLUMINANT.iter().into_sweep(),
        );

    let white_material = scene.add_material(Material {
        reflectance: Reflectance::Constant(white_reflectance),
        surface: Surface::Diffuse,
        emission: None,
    });
    let red_material = scene.add_material(Material {
        reflectance: Reflectance::Constant(red_reflectance),
        surface: Surface::Diffuse,
        emission: None,
    });
    let green_material = scene.add_material(Material {
        reflectance: Reflectance::Constant(green_reflectance),
        surface: Surface::Diffuse,
        emission: None,
    });
    let tall_block_material = match variant {
        CornellBoxVariant::DomeLight => scene.add_material(Material {
            reflectance: Reflectance::Constant(Vec3::broadcast(1.0)),
            surface: Surface::Mirror,
            emission: None,
        }),
        _ => white_material,
    };

    let identity = scene.add_transform(Transform::default());

    scene.add_instance(Instance::new(identity, floor, white_material));
    scene.add_instance(Instance::new(identity, ceiling, white_material));
    scene.add_instance(Instance::new(identity, grey_wall, white_material));
    scene.add_instance(Instance::new(identity, red_wall, red_material));
    scene.add_instance(Instance::new(identity, green_wall, green_material));
    if !matches!(variant, CornellBoxVariant::Conductor) {
        scene.add_instance(Instance::new(identity, short_block, white_material));
        scene.add_instance(Instance::new(identity, tall_block, tall_block_material));
    }

    match variant {
        CornellBoxVariant::DomeLight => {
            scene.add_light(Light::Dome {
                emission: Emission::new_uniform(Vec3::new(0.4, 0.6, 0.8) * 0.8),
            });
            let solid_angle = PI / 4096.0;
            scene.add_light(Light::SolidAngle {
                emission: Emission::new_uniform(Vec3::new(1.0, 0.8, 0.6) * 2.0 / solid_angle),
                direction_ws: Vec3::new(-1.0, 8.0, -5.0).normalized(),
                solid_angle,
            });
        }
        CornellBoxVariant::Conductor => {
            let light_x = 0.45;
            let light_z = 0.1;
            let r_a = 0.05;
            let r_b = 0.0005;
            let power = 0.005;
            for i in 0..4 {
                let r = r_a + (r_b - r_a) * ((i as f32) / 3.0).powf(0.5);
                let sphere = scene.add_geometry(Geometry::Sphere {
                    centre: Vec3::new(light_x, 0.1 + 0.1 * (i as f32), light_z),
                    radius: r,
                });
                let material = scene.add_material(Material {
                    reflectance: Reflectance::Constant(Vec3::zero()),
                    surface: Surface::None,
                    emission: Some(Emission {
                        illuminant: Illuminant::CornellBox,
                        intensity: Vec3::broadcast(power / (4.0 * PI * r * r)),
                    }),
                });
                scene.add_instance(Instance::new(identity, sphere, material));
            }

            let camera_x = 0.278;
            let camera_z = -0.8;
            let roughness = [0.05, 0.1, 0.25, 0.5];
            for (i, roughness) in roughness.iter().copied().enumerate() {
                let x = 0.35 - 0.11 * (i as f32).powf(0.85);
                let y = 0.5488 / 2.0;
                let z = 0.25 - 0.06 * (i as f32).powf(1.2);

                let look_dir = Vec3::new(camera_x - x, 0.0, camera_z - z).normalized();
                let light_dir = Vec3::new(light_x - x, 0.0, light_z - z).normalized();
                let half_dir = (look_dir + light_dir).normalized();
                let rotation = Rotor3::from_rotation_between(Vec3::unit_z(), half_dir);

                let quad = scene.add_geometry(Geometry::Quad {
                    local_from_quad: Similarity3 {
                        translation: Vec3::new(x, y, z),
                        rotation,
                        scale: 1.0,
                    },
                    size: Vec2::new(0.1, 0.5488 * 0.9),
                });
                let material = scene.add_material(Material {
                    reflectance: Reflectance::Constant(Vec3::broadcast(0.8)),
                    surface: Surface::RoughConductor {
                        conductor: Conductor::Aluminium,
                        roughness,
                    },
                    emission: None,
                });
                scene.add_instance(Instance::new(identity, quad, material));
            }
        }
        _ => {
            let light_x0 = 0.213;
            let light_x1 = 0.343;
            let light_z0 = 0.227;
            let light_z1 = 0.332;
            let light_y = 0.5488 - 0.0001;
            let light_geometry = scene.add_geometry(Geometry::Quad {
                local_from_quad: Similarity3::new(
                    Vec3::new(0.5 * (light_x1 + light_x0), light_y, 0.5 * (light_z1 + light_z0)),
                    Rotor3::from_rotation_yz(0.5 * PI),
                    1.0,
                ),
                size: Vec2::new(light_x1 - light_x0, light_z1 - light_z0),
            });
            let light_material = scene.add_material(Material {
                reflectance: Reflectance::Constant(Vec3::broadcast(0.78)),
                surface: Surface::Diffuse,
                emission: Some(Emission {
                    illuminant: Illuminant::CornellBox,
                    intensity: Vec3::broadcast(1.0),
                }),
            });
            scene.add_instance(Instance::new(identity, light_geometry, light_material));
        }
    }

    scene.add_camera(Camera::Pinhole {
        world_from_camera: Similarity3::new(Vec3::new(0.278, 0.273, -0.8), Rotor3::identity(), 0.25),
        fov_y: 2.0 * (0.025f32 / 2.0).atan2(0.035),
    });

    scene
}

#[derive(Clone, Copy)]
struct PlyVertex {
    pos: Vec3,
}

#[derive(Clone, Copy)]
struct PlyFace {
    indices: UVec3,
}

impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        Self { pos: Vec3::zero() }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.pos.x = v,
            ("y", ply::Property::Float(v)) => self.pos.y = v,
            ("z", ply::Property::Float(v)) => self.pos.z = v,
            _ => {}
        }
    }
}

impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        Self { indices: UVec3::zero() }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListInt(v)) => {
                assert_eq!(v.len(), 3);
                for (dst, src) in self.indices.as_mut_slice().iter_mut().zip(v.iter()) {
                    *dst = *src as u32;
                }
            }
            _ => {}
        }
    }
}

pub fn load_ply(filename: &Path) -> Geometry {
    println!("loading {:?}", filename);
    let mut f = BufReader::new(File::open(filename).unwrap());

    let vertex_parser = parser::Parser::<PlyVertex>::new();
    let face_parser = parser::Parser::<PlyFace>::new();
    let header = vertex_parser.read_header(&mut f).unwrap();
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for (_key, element) in header.elements.iter() {
        match element.name.as_ref() {
            "vertex" => {
                vertices = vertex_parser
                    .read_payload_for_element(&mut f, element, &header)
                    .unwrap();
            }
            "face" => {
                faces = face_parser.read_payload_for_element(&mut f, element, &header).unwrap();
            }
            _ => {}
        }
    }

    let mut min = Vec3::broadcast(f32::INFINITY);
    let mut max = Vec3::broadcast(-f32::INFINITY);
    for v in vertices.iter() {
        min = min.min_by_component(v.pos);
        max = max.max_by_component(v.pos);
    }

    let mut normals = vec![Vec3::zero(); vertices.len()];
    let mut area = 0.0;
    for src in faces.iter() {
        let v0 = vertices[src.indices[0] as usize].pos;
        let v1 = vertices[src.indices[1] as usize].pos;
        let v2 = vertices[src.indices[2] as usize].pos;
        area += (0.5 * (v2 - v1).cross(v0 - v1).mag()) as f64;
        let normal = (v2 - v1).cross(v0 - v1).normalized();
        if !normal.is_nan() {
            // TODO: weight by angle at vertex?
            normals[src.indices[0] as usize] += normal;
            normals[src.indices[1] as usize] += normal;
            normals[src.indices[2] as usize] += normal;
        }
    }
    for n in normals.iter_mut() {
        let u = n.normalized();
        if !u.is_nan() {
            *n = u;
        }
    }
    let uvs = vec![Vec2::zero(); normals.len()];

    Geometry::TriangleMesh {
        positions: vertices.drain(..).map(|v| v.pos).collect(),
        normals: Some(normals),
        uvs: Some(uvs),
        indices: faces.drain(..).map(|f| f.indices).collect(),
        min,
        max,
        area: area as f32,
    }
}

pub fn create_material_test_scene(ply_filename: &Path, surfaces: &[Surface], illuminant: Illuminant) -> Scene {
    let mut scene = Scene::default();

    let eps = 0.001;
    let wall_distance = 4.0 - eps;
    let floor_size = 8.0 - eps;
    let floor_geometry = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(-floor_size, eps, wall_distance),
                Vec3::new(floor_size, eps, wall_distance),
                Vec3::new(floor_size, eps, -floor_size),
                Vec3::new(-floor_size, eps, -floor_size),
            )
            .with_quad(
                Vec3::new(-floor_size, eps, wall_distance),
                Vec3::new(-floor_size, floor_size, wall_distance),
                Vec3::new(floor_size, floor_size, wall_distance),
                Vec3::new(floor_size, eps, wall_distance),
            )
            .build(),
    );
    let floor_material = scene.add_material(Material {
        reflectance: Reflectance::Checkerboard(Vec3::broadcast(0.8)),
        surface: Surface::Diffuse,
        emission: None,
    });
    let identity = scene.add_transform(Transform::default());
    scene.add_instance(Instance::new(identity, floor_geometry, floor_material));

    let object_mesh = load_ply(ply_filename);
    let (centre, half_extent) = match object_mesh {
        Geometry::TriangleMesh { min, max, .. } => (0.5 * (max + min), 0.5 * (max - min)),
        _ => panic!("expected a triangle mesh"),
    };

    let object_geometry = scene.add_geometry(object_mesh);
    let max_half_extent = half_extent.component_max();
    let y_offset = 1.01 * half_extent.y / max_half_extent;
    let spacing = 1.5;
    for (i, surface) in surfaces.iter().enumerate() {
        let object_transform = scene.add_transform(Transform {
            world_from_local: Similarity3::new(
                Vec3::new(
                    ((i as f32) - (surfaces.len() as f32 - 1.0) * 0.5) * spacing,
                    y_offset,
                    0.0,
                ),
                Rotor3::from_rotation_xz(0.75 * PI),
                1.0 / max_half_extent,
            ) * Similarity3::new(-centre, Rotor3::identity(), 1.0),
        });
        let object_material = scene.add_material(Material {
            reflectance: Reflectance::Constant(Vec3::one()),
            surface: *surface,
            emission: None,
        });
        scene.add_instance(Instance::new(object_transform, object_geometry, object_material));
    }

    let light1_geometry = scene.add_geometry(Geometry::Quad {
        local_from_quad: Similarity3::new(Vec3::new(0.0, 6.0, 0.0), Rotor3::from_rotation_yz(0.5 * PI), 1.0),
        size: Vec2::new(5.0, 5.0),
    });
    let light2_geometry = scene.add_geometry(Geometry::Quad {
        local_from_quad: Similarity3::new(Vec3::new(-6.0, 3.0, 0.0), Rotor3::from_rotation_xz(-0.5 * PI), 1.0),
        size: Vec2::new(5.0, 5.0),
    });
    let light3_geometry = scene.add_geometry(Geometry::Quad {
        local_from_quad: Similarity3::new(Vec3::new(4.5, 3.0, -4.5), Rotor3::from_rotation_xz(0.25 * PI), 1.0),
        size: Vec2::new(5.0, 5.0),
    });
    let light_material = scene.add_material(Material {
        reflectance: Reflectance::Constant(Vec3::zero()),
        surface: Surface::None,
        emission: Some(Emission {
            illuminant,
            intensity: Vec3::broadcast(1.0),
        }),
    });
    scene.add_instance(Instance::new(identity, light1_geometry, light_material));
    scene.add_instance(Instance::new(identity, light2_geometry, light_material));
    scene.add_instance(Instance::new(identity, light3_geometry, light_material));
    scene.add_light(Light::Dome {
        emission: Emission::new_uniform(Vec3::broadcast(0.05)),
    });

    let camera_orientation = Rotor3::from_rotation_xz(-PI / 16.0) * Rotor3::from_rotation_yz(PI / 8.0);
    let distance = 5.5 + 1.0 * ((surfaces.len() - 1) as f32);
    scene.add_camera(Camera::Pinhole {
        world_from_camera: Similarity3::new(
            Vec3::new(0.0, y_offset, 0.0) + camera_orientation * Vec3::new(0.0, 0.0, -distance),
            camera_orientation,
            1.0,
        ),
        fov_y: PI / 8.0,
    });

    scene
}
