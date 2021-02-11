use crate::scene;
use caldera::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

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
    scale: Option<ScalarOrVec3>,
    rotation: Option<[f32; 3]>,
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
    fov: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Scene {
    bsdfs: Vec<Bsdf>,
    primitives: Vec<Primitive>,
    camera: Camera,
}

pub fn load_scene<P: AsRef<Path>>(path: P) -> scene::Scene {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let scene: Scene = serde_json::from_reader(reader).unwrap();
    println!("{:#?}", scene);

    scene::Scene::default()
}
