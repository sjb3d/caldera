use bitvec::prelude::*;
use caldera::prelude::*;

pub const MAX_VERTICES_PER_CLUSTER: usize = 64;
pub const MAX_TRIANGLES_PER_CLUSTER: usize = 124;

pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub triangles: Vec<UVec3>,
}

#[derive(Debug)]
pub struct Cluster {
    pub mesh_vertices: Vec<u32>,
    pub triangles: Vec<UVec3>,
}

impl Cluster {
    fn new() -> Self {
        Self {
            mesh_vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}

struct TriangleListPerVertex {
    offset_per_vertex: Vec<u32>,
    triangle_indices: Vec<u32>,
}

impl TriangleListPerVertex {
    fn new(mesh: &Mesh) -> Self {
        let vertex_count = mesh.positions.len();
        let triangle_count = mesh.triangles.len();
        let corner_count = 3 * triangle_count;

        let mut offset_per_vertex = vec![0u32; vertex_count + 1];
        for triangle in mesh.triangles.iter() {
            for &vertex in triangle.as_slice() {
                offset_per_vertex[vertex as usize] += 1;
            }
        }

        let mut postfix_sum = 0u32;
        for offset in offset_per_vertex.iter_mut() {
            postfix_sum += *offset;
            *offset = postfix_sum;
        }

        let mut triangle_indices = vec![u32::MAX; corner_count];
        for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
            for &vertex in triangle.as_slice() {
                let offset_slot = &mut offset_per_vertex[vertex as usize];
                let offset = *offset_slot - 1;
                triangle_indices[offset as usize] = triangle_index as u32;
                *offset_slot = offset;
            }
        }

        Self {
            offset_per_vertex,
            triangle_indices,
        }
    }

    fn triangle_indices_for_vertex(&self, vertex: u32) -> &[u32] {
        let begin = self.offset_per_vertex[vertex as usize] as usize;
        let end = self.offset_per_vertex[vertex as usize + 1] as usize;
        &self.triangle_indices[begin..end]
    }
}

struct ClusterVertexRemap(Vec<u8>);

impl ClusterVertexRemap {
    fn new(mesh: &Mesh) -> Self {
        Self(vec![u8::MAX; mesh.positions.len()])
    }

    fn get_or_insert(&mut self, mesh_vertex: u32, cluster: &mut Cluster) -> u32 {
        let mut cluster_vertex = self.0[mesh_vertex as usize];
        if cluster_vertex == u8::MAX {
            cluster_vertex = cluster.mesh_vertices.len() as u8;
            cluster.mesh_vertices.push(mesh_vertex);
            self.0[mesh_vertex as usize] = cluster_vertex as u8;
        }
        cluster_vertex as u32
    }

    fn contains(&self, mesh_vertex: u32) -> bool {
        self.0[mesh_vertex as usize] != u8::MAX
    }

    fn reset(&mut self, cluster: &Cluster) {
        for &mesh_vertex in &cluster.mesh_vertices {
            self.0[mesh_vertex as usize] = u8::MAX;
        }
    }
}

struct ClusterBuilder<'m> {
    mesh: &'m Mesh,
    triangle_list_per_vertex: TriangleListPerVertex,
    available_triangles: BitVec,
    vertex_remap: ClusterVertexRemap,
}

impl<'m> ClusterBuilder<'m> {
    fn new(mesh: &'m Mesh) -> Self {
        Self {
            mesh,
            triangle_list_per_vertex: TriangleListPerVertex::new(mesh),
            available_triangles: bitvec![1; mesh.triangles.len()],
            vertex_remap: ClusterVertexRemap::new(mesh),
        }
    }

    fn add_next_available_triangle(&mut self, cluster: &mut Cluster) -> bool {
        assert!(cluster.mesh_vertices.len() + 3 <= MAX_VERTICES_PER_CLUSTER);
        assert!(cluster.triangles.len() + 1 <= MAX_TRIANGLES_PER_CLUSTER);
        if let Some(triangle_index) = self.available_triangles.first_one() {
            let triangle = self.mesh.triangles[triangle_index]
                .map_mut(|mesh_vertex| self.vertex_remap.get_or_insert(mesh_vertex, cluster));
            cluster.triangles.push(triangle);
            self.available_triangles.set(triangle_index, false);
            true
        } else {
            false
        }
    }

    fn find_best_adjacent_triangle(&self, cluster: &Cluster) -> Option<u32> {
        // early out if full
        if cluster.mesh_vertices.len() == MAX_VERTICES_PER_CLUSTER
            || cluster.triangles.len() == MAX_TRIANGLES_PER_CLUSTER
        {
            return None;
        }

        // HACK: just return the first one for now
        // TODO: build score using effect on cluster position and normal bounds
        cluster
            .mesh_vertices
            .iter()
            .flat_map(|&vertex| self.triangle_list_per_vertex.triangle_indices_for_vertex(vertex))
            .copied()
            .filter(|&triangle_index| self.available_triangles[triangle_index as usize])
            .filter(|&triangle_index| {
                let new_vertex_count = self.mesh.triangles[triangle_index as usize]
                    .as_slice()
                    .iter()
                    .copied()
                    .filter(|&vertex| !self.vertex_remap.contains(vertex))
                    .count();
                cluster.mesh_vertices.len() + new_vertex_count <= MAX_VERTICES_PER_CLUSTER
            })
            .next()
    }

    fn build(mut self) -> Vec<Cluster> {
        let mut clusters = Vec::new();
        loop {
            let mut cluster = Cluster::new();
            if !self.add_next_available_triangle(&mut cluster) {
                break;
            }
            while let Some(triangle_index) = self.find_best_adjacent_triangle(&cluster) {
                let triangle = self.mesh.triangles[triangle_index as usize]
                    .map_mut(|mesh_vertex| self.vertex_remap.get_or_insert(mesh_vertex, &mut cluster));
                cluster.triangles.push(triangle);
                self.available_triangles.set(triangle_index as usize, false);
            }
            self.vertex_remap.reset(&cluster);
            clusters.push(cluster);
        }
        clusters
    }
}

pub fn build_clusters(mut mesh: Mesh) -> (Mesh, Vec<Cluster>) {
    let mut clusters = ClusterBuilder::new(&mesh).build();

    let mut new_vertex_from_old = vec![u32::MAX; mesh.positions.len()];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    mesh.triangles.clear();
    for cluster in clusters.iter_mut() {
        for mesh_vertex in cluster.mesh_vertices.iter_mut() {
            let old_vertex = *mesh_vertex;
            let mut new_vertex = new_vertex_from_old[old_vertex as usize];
            if new_vertex == u32::MAX {
                new_vertex = positions.len() as u32;
                new_vertex_from_old[old_vertex as usize] = new_vertex;
                positions.push(mesh.positions[old_vertex as usize]);
                normals.push(mesh.normals[old_vertex as usize]);
            }
            *mesh_vertex = new_vertex;
        }
        mesh.triangles.extend(
            cluster
                .triangles
                .iter()
                .map(|&triangle| triangle.map_mut(|cluster_vertex| cluster.mesh_vertices[cluster_vertex as usize])),
        );
    }
    mesh.positions = positions;
    mesh.normals = normals;

    (mesh, clusters)
}
