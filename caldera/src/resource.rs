use bitvec::prelude::*;
use std::num;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Generation(num::NonZeroU16);

impl Generation {
    pub(crate) fn advance(&mut self) {
        self.0 = num::NonZeroU16::new(self.0.get().wrapping_add(1)).unwrap_or_else(Self::first_generation)
    }

    pub(crate) fn first_generation() -> num::NonZeroU16 {
        unsafe { num::NonZeroU16::new_unchecked(1) }
    }
}

impl Default for Generation {
    fn default() -> Self {
        Self(Self::first_generation())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ResourceHandle {
    generation: Generation,
    index: u16,
}

pub(crate) struct ResourceEntry<T> {
    generation: Generation,
    data: Option<T>,
}

impl<T> Default for ResourceEntry<T> {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            data: None,
        }
    }
}

pub(crate) struct ResourceVec<T> {
    active: BitVec,
    entries: Vec<ResourceEntry<T>>,
}

impl<T> ResourceVec<T> {
    pub(crate) fn new() -> Self {
        Self {
            active: BitVec::new(),
            entries: Vec::new(),
        }
    }

    pub(crate) fn allocate(&mut self, data: T) -> ResourceHandle {
        let index = if let Some(index) = self.active.iter_zeros().next() {
            assert_eq!(self.active.get_mut(index).unwrap().replace(true), false);
            index
        } else {
            let index = self.active.len();
            self.active.push(true);
            self.entries.push(ResourceEntry::default());
            index
        };
        let entry = self.entries.get_mut(index).unwrap();
        entry.generation.advance();
        entry.data = Some(data);
        ResourceHandle {
            generation: entry.generation,
            index: index as u16,
        }
    }

    pub(crate) fn free(&mut self, handle: ResourceHandle) {
        let index = handle.index as usize;
        assert_eq!(self.active.get_mut(index).unwrap().replace(false), true);
        self.entries.get_mut(index).unwrap().data.take().unwrap();
    }

    pub(crate) fn get(&self, handle: ResourceHandle) -> Option<&T> {
        self.entries.get(handle.index as usize).and_then(|e| {
            if e.generation == handle.generation {
                e.data.as_ref()
            } else {
                None
            }
        })
    }

    pub(crate) fn active_count(&self) -> usize {
        self.active.count_ones()
    }

    pub(crate) fn get_mut(&mut self, handle: ResourceHandle) -> Option<&mut T> {
        self.entries.get_mut(handle.index as usize).and_then(|e| {
            if e.generation == handle.generation {
                e.data.as_mut()
            } else {
                None
            }
        })
    }
}
