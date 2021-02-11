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
        world_from_local: Similarity3,
    },
    Mesh {
        name: &'a str,
        positions: Vec<Vec3>,
        indices: Vec<UVec3>,
    },
    Instance {
        transform_ref: &'a str,
        geometry_ref: &'a str,
        surface: Surface,
        reflectance: Vec3,
    },
    Camera {
        transform_ref: &'a str,
        fov_y: f32,
    },
}

fn rotor3_from_quaternion(q: Vec4) -> Rotor3 {
    Rotor3::new(q.x, Bivec3::new(-q.w, q.z, -q.y))
}

fn vec4(i: &str) -> IResult<&str, Vec4> {
    map(
        tuple((
            float,
            preceded(multispace1, float),
            preceded(multispace1, float),
            preceded(multispace1, float),
        )),
        |(x, y, z, w)| Vec4::new(x, y, z, w),
    )(i)
}

fn similarity3(i: &str) -> IResult<&str, Similarity3> {
    map(
        tuple((vec3, preceded(multispace1, vec4), preceded(multispace1, float))),
        |(translation, rotation, scale)| Similarity3::new(translation, rotor3_from_quaternion(rotation), scale),
    )(i)
}

fn quoted_name(i: &str) -> IResult<&str, &str> {
    delimited(char('"'), take_till1(|c| c == '"'), char('"'))(i)
}

fn element_transform(i: &str) -> IResult<&str, Element> {
    let (i, name) = quoted_name(i)?;
    let (i, world_from_local) = preceded(multispace0, similarity3)(i)?;
    Ok((i, Element::Transform { name, world_from_local }))
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
    let (i, transform_ref) = quoted_name(i)?;
    let (i, fov_y) = preceded(multispace0, float)(i)?;
    Ok((i, Element::Camera { transform_ref, fov_y }))
}

fn surface(i: &str) -> IResult<&str, Surface> {
    alt((
        map(tag("diffuse"), |_| Surface::Diffuse),
        map(tag("mirror"), |_| Surface::Mirror),
    ))(i)
}

fn element_instance(i: &str) -> IResult<&str, Element> {
    let (i, transform_ref) = quoted_name(i)?;
    let (i, geometry_ref) = preceded(multispace0, quoted_name)(i)?;
    let (i, surface) = preceded(multispace0, surface)(i)?;
    let (i, reflectance) = preceded(multispace1, vec3)(i)?;
    Ok((
        i,
        Element::Instance {
            transform_ref,
            geometry_ref,
            surface,
            reflectance,
        },
    ))
}

fn element(i: &str) -> IResult<&str, Element> {
    alt((
        preceded(tag("transform"), preceded(multispace0, element_transform)),
        preceded(tag("mesh"), preceded(multispace0, element_mesh)),
        preceded(tag("instance"), preceded(multispace0, element_instance)),
        preceded(tag("camera"), preceded(multispace0, element_camera)),
    ))(i)
}

pub fn load_export(i: &str) -> Scene {
    let mut elements = all_consuming(terminated(many1(preceded(multispace0, element)), multispace0))(i)
        .unwrap()
        .1;

    let mut scene = Scene::default();
    let mut transform_refs = HashMap::new();
    let mut geometry_refs = HashMap::new();

    scene.add_light(Light::Dome {
        emission: Vec3::new(0.5, 0.6, 0.7),
    });

    for element in elements.drain(..) {
        match element {
            Element::Transform { name, world_from_local } => {
                let transform_ref = scene.add_transform(Transform::new(world_from_local));
                if transform_refs.insert(name, transform_ref).is_some() {
                    panic!("multiple transforms with name \"{}\"", name);
                }
            }
            Element::Mesh {
                name,
                positions,
                indices,
            } => {
                let mut min = Vec3::broadcast(f32::INFINITY);
                let mut max = Vec3::broadcast(-f32::INFINITY);
                for pos in positions.iter() {
                    min = min.min_by_component(*pos);
                    max = max.max_by_component(*pos);
                }
                let geometry_ref = scene.add_geometry(Geometry::TriangleMesh {
                    positions,
                    uvs: Vec::new(),
                    indices,
                    min,
                    max,
                });
                if geometry_refs.insert(name, geometry_ref).is_some() {
                    panic!("multiple geometry with name \"{}\"", name);
                }
            }
            Element::Instance {
                transform_ref,
                geometry_ref,
                surface,
                reflectance,
            } => {
                let shader_ref = scene.add_shader(Shader {
                    reflectance: Reflectance::Constant(reflectance),
                    surface,
                    emission: None,
                });
                scene.add_instance(Instance::new(
                    *transform_refs.get(transform_ref).unwrap(),
                    *geometry_refs.get(geometry_ref).unwrap(),
                    shader_ref,
                ));
            }
            Element::Camera { transform_ref, fov_y } => {
                scene.add_camera(Camera {
                    transform_ref: *transform_refs.get(transform_ref).unwrap(),
                    fov_y,
                });
            }
        }
    }

    scene.bake_unique_geometry();
    scene
}
