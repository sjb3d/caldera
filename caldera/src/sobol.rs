struct SobolIterator {
    v: [u32; 1 + Self::MAX_INDEX_BITS],
    x: u32,
    i: u32,
}

impl SobolIterator {
    const MAX_INDEX_BITS: usize = 16;
    const MAX_INDEX: u32 = ((1 << Self::MAX_INDEX_BITS) - 1) as u32;

    fn new(a: u32, m: &[u32]) -> Self {
        let mut v = [0u32; 1 + Self::MAX_INDEX_BITS];
        for (i, (v, m)) in v.iter_mut().zip(m.iter()).enumerate() {
            *v = m << (31 - i);
        }
        let s = m.len();
        for i in s..=Self::MAX_INDEX_BITS {
            let j = i - s;
            v[i] = v[j] ^ (v[j] >> s);
            for k in 0..(s - 1) {
                if ((a >> k) & 1) != 0 {
                    v[i] ^= v[j + 1 + k];
                }
            }
        }
        Self { v, x: 0, i: u32::MAX }
    }

    fn advance(&mut self) -> u32 {
        // compute the next value in gray code order
        self.i = self.i.wrapping_add(1);
        self.x ^= self.v[self.i.trailing_ones() as usize];
        self.x
    }
}

impl Iterator for SobolIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        (self.i != Self::MAX_INDEX).then(|| self.advance())
    }
}

pub fn sobol(d: u32) -> impl Iterator<Item = u32> {
    // direction numbers from new-joe-kuo-6.21201, see https://web.maths.unsw.edu.au/~fkuo/sobol/
    let (a, m): (_, &[_]) = match d {
        0 => (0, &[1; 32]),
        1 => (0, &[1]),
        2 => (1, &[1, 3]),
        3 => (1, &[1, 3, 1]),
        4 => (2, &[1, 1, 1]),
        5 => (1, &[1, 1, 3, 3]),
        6 => (4, &[1, 3, 5, 13]),
        7 => (2, &[1, 1, 5, 5, 17]),
        _ => unimplemented!(),
    };
    SobolIterator::new(a, m)
}
