use crate::context::*;
use imgui::Ui;
use spark::vk;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Alloc {
    pub mem: vk::DeviceMemory,
    pub offset: vk::DeviceSize,
}

struct Chunk {
    context: Arc<Context>,
    memory_type_index: u32,
    mem: vk::DeviceMemory,
    size: u32,
    offset: u32,
}

impl Chunk {
    pub fn new(context: &Arc<Context>, memory_type_index: u32, size: u32) -> Self {
        let mem = {
            let memory_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: vk::DeviceSize::from(size),
                memory_type_index,
                ..Default::default()
            };
            unsafe { context.device.allocate_memory(&memory_allocate_info, None) }.unwrap()
        };
        Self {
            context: Arc::clone(&context),
            memory_type_index,
            mem,
            size,
            offset: 0,
        }
    }

    pub fn allocate(&mut self, mem_req: &vk::MemoryRequirements) -> Option<Alloc> {
        let alignment_mask = (mem_req.alignment as u32) - 1;
        let size = mem_req.size as u32;
        let offset = (self.offset + alignment_mask) & !alignment_mask;
        let next_offset = offset + size;
        if next_offset <= self.size {
            self.offset = next_offset;
            Some(Alloc {
                mem: self.mem,
                offset: vk::DeviceSize::from(offset),
            })
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe {
            self.context.device.free_memory(Some(self.mem), None);
        }
    }
}

pub struct Allocator {
    context: Arc<Context>,
    chunks: Vec<Chunk>,
    chunk_size: u32,
}

impl Allocator {
    pub fn new(context: &Arc<Context>, chunk_size: u32) -> Self {
        Self {
            context: Arc::clone(&context),
            chunks: Vec::new(),
            chunk_size,
        }
    }

    pub fn allocate(
        &mut self,
        mem_req: &vk::MemoryRequirements,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Alloc {
        let memory_type_index = self
            .context
            .get_memory_type_index(mem_req.memory_type_bits, memory_property_flags)
            .unwrap();
        for chunk in self.chunks.iter_mut() {
            if chunk.memory_type_index == memory_type_index {
                if let Some(alloc) = chunk.allocate(mem_req) {
                    return alloc;
                }
            }
        }
        let mut alloc_size = self.chunk_size;
        if (alloc_size as vk::DeviceSize) < mem_req.size {
            alloc_size = mem_req.size.next_power_of_two() as u32;
            println!("allocator: adding large chunk size {} MB", alloc_size / (1024 * 1024));
        }
        let mut chunk = Chunk::new(&self.context, memory_type_index, alloc_size);
        let alloc = chunk.allocate(mem_req);
        self.chunks.push(chunk);
        alloc.unwrap()
    }

    pub fn reset(&mut self) {
        for chunk in self.chunks.iter_mut() {
            chunk.reset();
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui, name: &str) {
        ui.text(name);
        ui.next_column();
        ui.text(format!(
            "{} MB",
            self.chunks.iter().map(|chunk| chunk.size as usize).sum::<usize>() / (1024 * 1024)
        ));
        ui.next_column();
    }
}
