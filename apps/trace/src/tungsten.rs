use crate::scene;
use bytemuck::{Pod, Zeroable};
use caldera::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs::File, io::BufReader, io::Read};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
enum ScalarOrVec3 {
    Scalar(f32),
    Vec3([f32; 3]),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum TextureOrValue {
    Value(ScalarOrVec3),
    Texture(String),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BsdfType {
    Null,
    Lambert,
    Mirror,
    Plastic,
    RoughPlastic,
    Dielectric,
    RoughDielectric,
    Conductor,
    RoughConductor,
    Transparency,
    #[serde(rename = "thinsheet")]
    ThinSheet,
    OrenNayar,
    LambertianFiber,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Distribution {
    Beckmann,
    #[serde(rename = "ggx")]
    GGX,
}

#[derive(Debug, Serialize, Deserialize)]
struct Bsdf {
    name: Option<String>,
    #[serde(rename = "type")]
    bsdf_type: BsdfType,
    albedo: TextureOrValue,
    distribution: Option<Distribution>,
    roughness: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrimitiveTransform {
    position: Option<[f32; 3]>,
    rotation: Option<[f32; 3]>,
    scale: Option<ScalarOrVec3>,
}

impl ScalarOrVec3 {
    fn into_vec3(self) -> Vec3 {
        match self {
            ScalarOrVec3::Scalar(s) => Vec3::broadcast(s),
            ScalarOrVec3::Vec3(v) => v.into(),
        }
    }

    fn ungamma_colour(self) -> Self {
        match self {
            ScalarOrVec3::Scalar(s) => ScalarOrVec3::Scalar(s),
            ScalarOrVec3::Vec3(v) => ScalarOrVec3::Vec3(Vec3::from(v).into_linear().into()),
        }
    }
}

impl PrimitiveTransform {
    fn decompose(&self) -> (Similarity3, Option<Vec3>) {
        let translation = self.position.map(Vec3::from).unwrap_or_else(Vec3::zero);
        let rotation = self
            .rotation
            .map(|r| {
                let r = Vec3::from(r) * PI / 180.0;
                Rotor3::from_euler_angles(r.z, r.x, r.y)
            })
            .unwrap_or_else(Rotor3::identity);
        let (scale, vector_scale) = self
            .scale
            .as_ref()
            .map(|s| match s {
                ScalarOrVec3::Scalar(s) => (*s, None),
                ScalarOrVec3::Vec3(v) => (1.0, Some(v.into())),
            })
            .unwrap_or((1.0, None));
        (Similarity3::new(translation, rotation, scale), vector_scale)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PrimitiveType {
    Mesh,
    Quad,
    Sphere,
    InfiniteSphereCap,
    InfiniteSphere,
    Skydome,
    Curves,
    #[serde(rename = "disk")]
    Disc,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum BsdfRef {
    Named(String),
    Inline(Bsdf),
}

#[derive(Debug, Serialize, Deserialize)]
struct Primitive {
    transform: PrimitiveTransform,
    #[serde(rename = "type")]
    primitive_type: PrimitiveType,
    file: Option<String>,
    power: Option<ScalarOrVec3>,
    emission: Option<TextureOrValue>,
    bsdf: Option<BsdfRef>,
    cap_angle: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CameraTransform {
    position: [f32; 3],
    look_at: [f32; 3],
    up: [f32; 3],
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum Resolution {
    Square(f32),
    Rectangular(f32, f32),
}

impl Resolution {
    fn aspect_ratio(&self) -> f32 {
        match self {
            Resolution::Square(_) => 1.0,
            Resolution::Rectangular(width, height) => width / height,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Camera {
    transform: CameraTransform,
    resolution: Resolution,
    fov: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Scene {
    bsdfs: Vec<Bsdf>,
    primitives: Vec<Primitive>,
    camera: Camera,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Triangle {
    indices: IVec3,
    mat: i32,
}

fn load_mesh<P: AsRef<Path>>(path: P, extra_scale: Option<Vec3>) -> scene::Geometry {
    let mut reader = BufReader::new(File::open(path).unwrap());

    let mut vertex_count = 0u64;
    reader.read_exact(bytemuck::bytes_of_mut(&mut vertex_count)).unwrap();

    let mut vertices = vec![Vertex::zeroed(); vertex_count as usize];
    reader
        .read_exact(bytemuck::cast_slice_mut(vertices.as_mut_slice()))
        .unwrap();

    let mut triangle_count = 0u64;
    reader.read_exact(bytemuck::bytes_of_mut(&mut triangle_count)).unwrap();

    let mut triangles = vec![Triangle::zeroed(); triangle_count as usize];
    reader
        .read_exact(bytemuck::cast_slice_mut(triangles.as_mut_slice()))
        .unwrap();

    if let Some(extra_scale) = extra_scale {
        for v in vertices.iter_mut() {
            v.pos *= extra_scale;
        }
    }

    let mut min = Vec3::broadcast(f32::INFINITY);
    let mut max = Vec3::broadcast(-f32::INFINITY);
    for v in vertices.iter() {
        min = min.min_by_component(v.pos);
        max = max.max_by_component(v.pos);
    }

    scene::Geometry::TriangleMesh {
        positions: vertices.iter().map(|v| v.pos).collect(),
        uvs: vertices.iter().map(|v| Vec2::new(v.uv.x, 1.0 - v.uv.y)).collect(),
        indices: triangles.drain(..).map(|t| t.indices.as_unsigned()).collect(),
        min,
        max,
    }
}

pub fn load_scene<P: AsRef<Path>>(path: P) -> scene::Scene {
    let reader = BufReader::new(File::open(&path).unwrap());

    let scene: Scene = serde_json::from_reader(reader).unwrap();

    let load_shader = |bsdf_ref: &BsdfRef| {
        let bsdf = match bsdf_ref {
            BsdfRef::Inline(bsdf) => bsdf,
            BsdfRef::Named(name) => scene
                .bsdfs
                .iter()
                .find(|bsdf| bsdf.name.as_ref() == Some(name))
                .unwrap(),
        };
        let mut reflectance = match &bsdf.albedo {
            TextureOrValue::Value(value) => scene::Reflectance::Constant(value.ungamma_colour().into_vec3()),
            TextureOrValue::Texture(filename) => scene::Reflectance::Texture(path.as_ref().with_file_name(filename)),
        };
        let surface = match bsdf.bsdf_type {
            BsdfType::Null => {
                reflectance = scene::Reflectance::Constant(Vec3::zero());
                scene::Surface::Diffuse
            }
            BsdfType::Lambert => scene::Surface::Diffuse,
            BsdfType::Mirror => scene::Surface::Mirror,
            BsdfType::Conductor => scene::Surface::Conductor { roughness: 0.0 },
            BsdfType::RoughConductor => scene::Surface::Conductor {
                roughness: bsdf.roughness.unwrap().sqrt(),
            },
            BsdfType::Plastic => scene::Surface::Plastic { roughness: 0.0 },
            BsdfType::RoughPlastic => scene::Surface::Plastic {
                roughness: bsdf.roughness.unwrap().sqrt(),
            },
            _ => scene::Surface::Diffuse,
        };
        scene::Shader {
            reflectance,
            surface,
            emission: None,
        }
    };

    let mut output = scene::Scene::default();
    for primitive in scene.primitives.iter() {
        match primitive.primitive_type {
            PrimitiveType::Quad => {
                let (world_from_local, extra_scale) = primitive.transform.decompose();

                let size = extra_scale
                    .map(|v| Vec2::new(v.x, v.z))
                    .unwrap_or_else(|| Vec2::broadcast(1.0));
                let mesh = scene::Geometry::Quad {
                    local_from_quad: Similarity3::new(Vec3::zero(), Rotor3::from_rotation_yz(-PI / 2.0), 1.0),
                    size,
                };

                let mut shader = load_shader(primitive.bsdf.as_ref().unwrap());
                if let Some(s) = primitive.power.as_ref() {
                    let area = size.x * size.y * world_from_local.scale.abs() * world_from_local.scale.abs();
                    shader.emission = Some(s.into_vec3() / (area * PI));
                }
                if let Some(TextureOrValue::Value(v)) = &primitive.emission {
                    shader.emission = Some(v.into_vec3());
                }

                let geometry_ref = output.add_geometry(mesh);
                let transform_ref = output.add_transform(scene::Transform::new(world_from_local));
                let shader_ref = output.add_shader(shader);
                output.add_instance(scene::Instance::new(transform_ref, geometry_ref, shader_ref));
            }
            PrimitiveType::Mesh => {
                let (world_from_local, extra_scale) = primitive.transform.decompose();

                let mesh = load_mesh(
                    path.as_ref().with_file_name(primitive.file.as_ref().unwrap()),
                    extra_scale,
                );
                let shader = load_shader(&primitive.bsdf.as_ref().unwrap());

                let geometry_ref = output.add_geometry(mesh);
                let transform_ref = output.add_transform(scene::Transform::new(world_from_local));
                let shader_ref = output.add_shader(shader);
                output.add_instance(scene::Instance::new(transform_ref, geometry_ref, shader_ref));
            }
            PrimitiveType::Sphere => {
                let (world_from_local, extra_scale) = primitive.transform.decompose();
                if extra_scale.is_some() {
                    unimplemented!();
                }

                let centre = world_from_local.translation;
                let radius = world_from_local.scale.abs();

                let mut shader = load_shader(&primitive.bsdf.as_ref().unwrap());
                if let Some(s) = primitive.power.as_ref() {
                    let area = 4.0 * PI * radius * radius;
                    shader.emission = Some(s.into_vec3() / (area * PI));
                }

                let geometry_ref = output.add_geometry(scene::Geometry::Sphere { centre, radius });
                let transform_ref = output.add_transform(scene::Transform::new(world_from_local));
                let shader_ref = output.add_shader(shader);
                output.add_instance(scene::Instance::new(transform_ref, geometry_ref, shader_ref));
            }
            PrimitiveType::InfiniteSphereCap => {
                let (world_from_local, _extra_scale) = primitive.transform.decompose();

                let theta = primitive.cap_angle.unwrap() * PI / 180.0;
                let solid_angle = 2.0 * PI * (1.0 - theta.cos());

                let emission = primitive.power.as_ref().map(|p| p.into_vec3() / solid_angle).unwrap();

                let direction_ws = world_from_local.rotation * Vec3::unit_y();

                output.add_light(scene::Light::SolidAngle {
                    emission,
                    direction_ws,
                    solid_angle,
                });
            }
            PrimitiveType::InfiniteSphere => {
                let emission = match primitive.emission.as_ref().unwrap() {
                    TextureOrValue::Value(v) => v.into_vec3(),
                    TextureOrValue::Texture(_) => {
                        println!("TODO: InfiniteSphere texture!");
                        Vec3::new(0.2, 0.3, 0.4)
                    }
                };
                output.add_light(scene::Light::Dome { emission });
            }
            PrimitiveType::Skydome => {
                println!("TODO: convert Skydome geometry!");
                output.add_light(scene::Light::Dome {
                    emission: Vec3::new(0.2, 0.3, 0.4),
                });
            }
            PrimitiveType::Curves => {
                println!("TODO: curves!");
            }
            PrimitiveType::Disc => {
                println!("TODO: disc!");
            }
        }
    }
    {
        let camera = &scene.camera;
        let world_from_local = {
            let isometry = Mat4::look_at(
                camera.transform.position.into(),
                camera.transform.look_at.into(),
                camera.transform.up.into(),
            )
            .into_isometry()
            .inversed();
            Similarity3::new(
                isometry.translation,
                isometry.rotation * Rotor3::from_rotation_xz(PI),
                1.0,
            )
        };
        let transform_ref = output.add_transform(scene::Transform::new(world_from_local));
        let aspect_ratio = camera.resolution.aspect_ratio();
        output.add_camera(scene::Camera {
            transform_ref,
            fov_y: camera.fov * (PI / 180.0) / aspect_ratio,
        });
    }
    output.bake_unique_geometry();
    output
}