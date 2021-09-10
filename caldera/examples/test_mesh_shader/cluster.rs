use bitvec::prelude::*;
use caldera::prelude::*;

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

struct VertexRemap(Vec<u8>);

impl VertexRemap {
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
    vertex_remap: VertexRemap,
}

impl<'m> ClusterBuilder<'m> {
    const MAX_VERTICES: usize = 64;
    const MAX_TRIANGLES: usize = 126;

    fn new(mesh: &'m Mesh) -> Self {
        Self {
            mesh,
            triangle_list_per_vertex: TriangleListPerVertex::new(mesh),
            available_triangles: bitvec![1; mesh.triangles.len()],
            vertex_remap: VertexRemap::new(mesh),
        }
    }

    fn add_next_available_triangle(&mut self, cluster: &mut Cluster) -> bool {
        assert!(cluster.mesh_vertices.len() + 3 <= Self::MAX_VERTICES);
        assert!(cluster.triangles.len() + 1 <= Self::MAX_TRIANGLES);
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
        // HACK: if we cannot add 2 new vertices then just bail
        // TODO: minimal check, check budget vs each candidate triangle
        if cluster.mesh_vertices.len() + 2 > Self::MAX_VERTICES || cluster.triangles.len() + 1 > Self::MAX_TRIANGLES {
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

pub fn build_clusters(mesh: &Mesh) -> Vec<Cluster> {
    ClusterBuilder::new(mesh).build()
}
