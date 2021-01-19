use std::num;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Generation(num::NonZeroU16);

impl Generation {
    pub(crate) fn advance(&mut self) {
        self.0 = num::NonZeroU16::new(self.0.get().wrapping_add(1)).unwrap_or(Self::first_generation())
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

#[derive(Clone, Copy, PartialEq, Eq)]
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

pub(crate) struct ResourceArray<T> {
    entries: Box<[ResourceEntry<T>]>,
}

impl<T> ResourceArray<T> {
    pub(crate) fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        for _i in 0..capacity {
            entries.push(Default::default());
        }
        Self {
            entries: entries.into_boxed_slice(),
        }
    }

    pub(crate) fn allocate(&mut self, data: T) -> Option<ResourceHandle> {
        // TODO: maintain bitmask of free slots
        self.entries
            .iter_mut()
            .enumerate()
            .find(|(_, e)| e.data.is_none())
            .map(|(i, e)| {
                e.generation.advance();
                e.data = Some(data);
                ResourceHandle {
                    generation: e.generation,
                    index: i as u16,
                }
            })
    }

    pub(crate) fn free(&mut self, handle: ResourceHandle) {
        self.entries
            .get_mut(handle.index as usize)
            .unwrap()
            .data
            .take()
            .unwrap();
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
