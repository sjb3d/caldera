#[derive(Debug, Clone, Copy)]
struct HeapBlock {
    begin: u32,
    end: u32,
}

#[derive(Debug)]
pub(crate) struct HeapAllocator {
    sorted_free_list: Vec<HeapBlock>,
    alloc_list: Vec<HeapBlock>,
}

impl HeapAllocator {
    pub fn new(size: u32) -> Self {
        Self {
            sorted_free_list: vec![HeapBlock { begin: 0, end: size }],
            alloc_list: Vec::new(),
        }
    }

    pub fn alloc(&mut self, size: u32, align: u32) -> Option<u32> {
        let align_mask = align - 1;
        for (index, block) in self.sorted_free_list.iter().enumerate() {
            let aligned = (block.begin + align_mask) & !align_mask;
            let alloc = HeapBlock {
                begin: aligned,
                end: aligned + size,
            };
            if alloc.end <= block.end {
                let block = block.clone();
                self.sorted_free_list.remove(index);
                if alloc.end != block.end {
                    self.sorted_free_list.insert(
                        index,
                        HeapBlock {
                            begin: alloc.end,
                            end: block.end,
                        },
                    );
                }
                if alloc.begin != block.begin {
                    self.sorted_free_list.insert(
                        index,
                        HeapBlock {
                            begin: block.begin,
                            end: alloc.begin,
                        },
                    )
                }
                self.alloc_list.push(alloc);
                return Some(alloc.begin);
            }
        }
        None
    }

    pub fn free(&mut self, alloc: u32) {
        let mut alloc = {
            let remove_index = self
                .alloc_list
                .iter()
                .enumerate()
                .find_map(|(index, block)| (block.begin == alloc).then(|| index))
                .expect("failed to find allocation");
            self.alloc_list.swap_remove(remove_index)
        };

        let mut insert_index = self
            .sorted_free_list
            .iter()
            .enumerate()
            .find_map(
                |(index, block)| {
                    if alloc.end <= block.begin {
                        Some(index)
                    } else {
                        None
                    }
                },
            )
            .unwrap_or(self.sorted_free_list.len());

        if let Some(next) = self.sorted_free_list.get(insert_index) {
            if alloc.end == next.begin {
                alloc.end = next.end;
                self.sorted_free_list.remove(insert_index);
            }
        }

        let prev_index = insert_index.wrapping_sub(1);
        if let Some(prev) = self.sorted_free_list.get(prev_index) {
            if prev.end == alloc.begin {
                alloc.begin = prev.begin;
                self.sorted_free_list.remove(prev_index);
                insert_index = prev_index;
            }
        }

        self.sorted_free_list.insert(insert_index, alloc);
    }
}
