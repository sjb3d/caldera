use crate::scene::*;
use caldera::*;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_till1},
    character::complete::{char, digit1, multispace0, multispace1},
    combinator::{all_consuming, map, map_res},
    multi::many1,
    number::complete::float,
    sequence::{delimited, preceded, terminated, tuple},
    IResult,
};
use std::collections::HashMap;

enum Element<'a> {
    Transform {
        name: &'a str,
        transform: Transform,
    },
    Mesh {
        name: &'a str,
        positions: Vec<Vec3>,
        indices: Vec<UVec3>,
    },
    Instance {
        transform: &'a str,
        geometry: &'a str,
    },
    Camera {
        transform: &'a str,
        fov_y: f32,
    },
}

fn transform_matrix(i: &str) -> IResult<&str, Transform> {
    map(
        tuple((
            float,
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
        )),
        |v| {
            let col0 = Vec3::new(v.0, v.4, v.8);
            let col1 = Vec3::new(v.1, v.5, v.9);
            let col2 = Vec3::new(v.2, v.6, v.10);
            let col3 = Vec3::new(v.3, v.7, v.11);
            let scale = col0.cross(col1).dot(col2).powf(1.0 / 3.0);
            let rotation = Mat3::new(col0 / scale, col1 / scale, col2 / scale).into_rotor3();
            let translation = col3;
            Transform(Similarity3::new(translation, rotation, scale))
        },
    )(i)
}

fn quoted_name(i: &str) -> IResult<&str, &str> {
    delimited(char('"'), take_till1(|c| c == '"'), char('"'))(i)
}

fn element_transform(i: &str) -> IResult<&str, Element> {
    let (i, name) = quoted_name(i)?;
    let (i, transform) = preceded(multispace0, transform_matrix)(i)?;
    Ok((i, Element::Transform { name, transform }))
}

fn vec3(i: &str) -> IResult<&str, Vec3> {
    map(
        tuple((float, preceded(multispace1, float), preceded(multispace1, float))),
        |(x, y, z)| Vec3::new(x, y, z),
    )(i)
}

fn uint(i: &str) -> IResult<&str, u32> {
    map_res(digit1, str::parse::<u32>)(i)
}

fn uvec3(i: &str) -> IResult<&str, UVec3> {
    map(
        tuple((uint, preceded(multispace1, uint), preceded(multispace1, uint))),
        |(x, y, z)| UVec3::new(x, y, z),
    )(i)
}

fn element_mesh(i: &str) -> IResult<&str, Element> {
    let (i, name) = quoted_name(i)?;
    let (i, positions) = delimited(
        preceded(multispace0, char('{')),
        many1(preceded(multispace0, vec3)),
        preceded(multispace0, char('}')),
    )(i)?;
    let (i, indices) = delimited(
        preceded(multispace0, char('{')),
        many1(preceded(multispace0, uvec3)),
        preceded(multispace0, char('}')),
    )(i)?;
    Ok((
        i,
        Element::Mesh {
            name,
            positions,
            indices,
        },
    ))
}

fn element_camera(i: &str) -> IResult<&str, Element> {
    let (i, transform) = quoted_name(i)?;
    let (i, fov_y) = preceded(multispace0, float)(i)?;
    Ok((i, Element::Camera { transform, fov_y }))
}

fn element_instance(i: &str) -> IResult<&str, Element> {
    let (i, transform) = quoted_name(i)?;
    let (i, geometry) = preceded(multispace0, quoted_name)(i)?;
    Ok((i, Element::Instance { transform, geometry }))
}

fn element(i: &str) -> IResult<&str, Element> {
    alt((
        preceded(tag("transform"), preceded(multispace0, element_transform)),
        preceded(tag("mesh"), preceded(multispace0, element_mesh)),
        preceded(tag("instance"), preceded(multispace0, element_instance)),
        preceded(tag("camera"), preceded(multispace0, element_camera)),
    ))(i)
}

pub fn parse_scene(i: &str) -> Scene {
    let mut elements = all_consuming(terminated(many1(preceded(multispace0, element)), multispace0))(i)
        .unwrap()
        .1;

    let mut scene = Scene::default();
    let mut transform_refs = HashMap::new();
    let mut geometry_refs = HashMap::new();

    scene.add_light(Light {
        emission: Vec3::new(0.5, 0.6, 0.7),
    });
    let shader_ref = scene.add_shader(ShaderBuilder::new_diffuse(Vec3::broadcast(0.8)).build());

    for element in elements.drain(..) {
        match element {
            Element::Transform { name, transform } => {
                let transform_ref = scene.add_transform(transform);
                transform_refs.insert(name, transform_ref);
            }
            Element::Mesh {
                name,
                positions,
                indices,
            } => {
                let geometry_ref = scene.add_geometry(Geometry::TriangleMesh { positions, indices });
                geometry_refs.insert(name, geometry_ref);
            }
            Element::Instance { transform, geometry } => {
                scene.add_instance(Instance {
                    transform_ref: *transform_refs.get(transform).unwrap(),
                    geometry_ref: *geometry_refs.get(geometry).unwrap(),
                    shader_ref,
                });
            }
            Element::Camera { transform, fov_y } => {
                scene.add_camera(Camera {
                    transform_ref: *transform_refs.get(transform).unwrap(),
                    fov_y,
                });
            }
        }
    }

    scene
}
