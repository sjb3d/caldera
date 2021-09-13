use crate::cluster::Mesh;
use caldera::prelude::*;
use ply_rs::{parser, ply};
use std::{fs::File, io::BufReader, path::Path};

pub fn load_ply_mesh(file_name: &Path) -> Mesh {
    let vertex_parser = parser::Parser::<PlyVertex>::new();
    let face_parser = parser::Parser::<PlyFace>::new();

    let mut f = BufReader::new(File::open(file_name).unwrap());
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
            _ => panic!("unexpected element {:?}", element),
        }
    }

    let mut normals = vec![Vec3::zero(); vertices.len()];
    let mut face_normals = Vec::new();
    for src in faces.iter() {
        let v0 = vertices[src.indices[0] as usize].pos;
        let v1 = vertices[src.indices[1] as usize].pos;
        let v2 = vertices[src.indices[2] as usize].pos;
        let face_normal = (v2 - v1).cross(v0 - v1).normalized();
        face_normals.push(if face_normal.is_nan() {
            Vec3::zero()
        } else {
            // TODO: weight by angle at vertex?
            normals[src.indices[0] as usize] += face_normal;
            normals[src.indices[1] as usize] += face_normal;
            normals[src.indices[2] as usize] += face_normal;
            face_normal
        });
    }
    for n in normals.iter_mut() {
        let u = n.normalized();
        if !u.is_nan() {
            *n = u;
        }
    }

    Mesh {
        positions: vertices.drain(..).map(|v| v.pos).collect(),
        normals,
        triangles: faces.drain(..).map(|f| f.indices).collect(),
        face_normals,
    }
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
            (k, _) => panic!("unknown key {}", k),
        }
    }
}
