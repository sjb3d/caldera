use crate::scene;
use bytemuck::{Pod, Zeroable};
use caldera::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs::File, io::BufReader, io::Read};

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum ScalarOrVec3 {
    Scalar(f32),
    Vec3([f32; 3]),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum Albedo {
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
    RoughConductor,
    Transparency,
    #[serde(rename = "thinsheet")]
    ThinSheet,
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
    albedo: Albedo,
    distribution: Option<Distribution>,
    roughness: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrimitiveTransform {
    position: Option<[f32; 3]>,
    rotation: Option<[f32; 3]>,
    scale: Option<ScalarOrVec3>,
}

impl PrimitiveTransform {
    fn into_similarity3(self) -> (Similarity3, Option<Vec3>) {
        let translation = self.position.map(Vec3::from).unwrap_or_else(Vec3::zero);
        let rotation = self
            .rotation
            .map(|r| Rotor3::from_euler_angles(r[0], r[1], r[2]))
            .unwrap_or_else(Rotor3::identity);
        let (scale, vector_scale) = self
            .scale
            .map(|s| match s {
                ScalarOrVec3::Scalar(s) => (s, None),
                ScalarOrVec3::Vec3(v) => (1.0, Some(v.into())),
            })
            .unwrap_or((1.0, None));
        (
            Similarity3 {
                translation,
                rotation,
                scale,
            },
            vector_scale,
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PrimitiveType {
    Mesh,
    Quad,
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
    power: Option<f32>,
    emission: Option<ScalarOrVec3>,
    bsdf: BsdfRef,
}

#[derive(Debug, Serialize, Deserialize)]
struct CameraTransform {
    position: [f32; 3],
    look_at: [f32; 3],
    up: [f32; 3],
}

#[derive(Debug, Serialize, Deserialize)]
struct Camera {
    transform: CameraTransform,
    resolution: [f32; 2],
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

fn load_mesh<P: AsRef<Path>>(path: P) -> scene::Geometry {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);

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
    let file = File::open(&path).unwrap();
    let reader = BufReader::new(file);

    let scene: Scene = serde_json::from_reader(reader).unwrap();

    let mut output = scene::Scene::default();
    for primitive in scene.primitives {
        match primitive.primitive_type {
            PrimitiveType::Quad => {}
            PrimitiveType::Mesh => {
                let (world_from_local, extra_scale) = primitive.transform.into_similarity3();
                if extra_scale.is_some() {
                    unimplemented!();
                }

                let mesh = load_mesh(path.as_ref().with_file_name(primitive.file.as_ref().unwrap()));

                let geometry_ref = output.add_geometry(mesh);
                let transform_ref = output.add_transform(scene::Transform::new(world_from_local));

                let bsdf = match &primitive.bsdf {
                    BsdfRef::Inline(bsdf) => bsdf,
                    BsdfRef::Named(name) => scene
                        .bsdfs
                        .iter()
                        .find(|bsdf| bsdf.name.as_ref() == Some(name))
                        .unwrap(),
                };
                let reflectance = match &bsdf.albedo {
                    Albedo::Value(value) => scene::Reflectance::Constant(match value {
                        ScalarOrVec3::Scalar(s) => Vec3::broadcast(*s),
                        ScalarOrVec3::Vec3(v) => v.into(),
                    }),
                    Albedo::Texture(filename) => scene::Reflectance::Texture(path.as_ref().with_file_name(filename)),
                };
                let shader_ref = output.add_shader(scene::Shader {
                    reflectance,
                    surface: scene::Surface::Diffuse,
                    emission: None,
                });

                output.add_instance(scene::Instance::new(transform_ref, geometry_ref, shader_ref));
            }
        }
    }
    output.add_light(scene::Light::Dome {
        emission: Vec3::new(0.2, 0.3, 0.4),
    });
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
        let aspect_ratio = camera.resolution[0] / camera.resolution[1];
        output.add_camera(scene::Camera {
            transform_ref,
            fov_y: camera.fov * (PI / 180.0) / aspect_ratio,
        });
    }
    output.bake_unique_geometry();
    output
}
