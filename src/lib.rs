//! Implementation of Sinsemilla outside the circuit.

#![no_std]

// We require `alloc` for now.
#[macro_use]
extern crate alloc;

use alloc::vec::Vec;

use group::{Curve, Wnaf};
use pasta_curves::{
    arithmetic::{CurveAffine, CurveExt},
    pallas,
};
use subtle::CtOption;

mod addition;
use self::addition::IncompletePoint;
mod sinsemilla_s;
pub use sinsemilla_s::SINSEMILLA_S;

/// Number of bits of each message piece in $\mathsf{SinsemillaHashToPoint}$
pub const K: usize = 10;

/// $\frac{1}{2^K}$
pub const INV_TWO_POW_K: [u8; 32] = [
    1, 0, 192, 196, 160, 229, 70, 82, 221, 165, 74, 202, 85, 7, 62, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 240, 63,
];

/// The largest integer such that $2^c \leq (r_P - 1) / 2$, where $r_P$ is the order
/// of Pallas.
pub const C: usize = 253;

// Sinsemilla Q generators

/// SWU hash-to-curve personalization for Sinsemilla $Q$ generators.
pub const Q_PERSONALIZATION: &str = "z.cash:SinsemillaQ";

// Sinsemilla S generators

/// SWU hash-to-curve personalization for Sinsemilla $S$ generators.
pub const S_PERSONALIZATION: &str = "z.cash:SinsemillaS";

/// Converts a little-endian [`K`]-bit string into an integer.
pub fn lebs2ip_k(bits: [bool; K]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0u32, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

/// Coordinate extractor for Pallas.
///
/// Defined in [Zcash Protocol Spec § 5.4.9.7: Coordinate Extractor for Pallas][concreteextractorpallas].
///
/// [concreteextractorpallas]: https://zips.z.cash/protocol/nu5.pdf#concreteextractorpallas
fn extract_p_bottom(point: CtOption<pallas::Point>) -> CtOption<pallas::Base> {
    point.map(|p| {
        p.to_affine()
            .coordinates()
            .map(|c| *c.x())
            .unwrap_or_else(pallas::Base::zero)
    })
}

/// Pads the given iterator (which MUST have length $\leq K * C$) with zero-bits to a
/// multiple of $K$ bits.
struct Pad<I: Iterator<Item = bool>> {
    /// The iterator we are padding.
    inner: I,
    /// The measured length of the inner iterator.
    ///
    /// This starts as a lower bound, and will be accurate once `padding_left.is_some()`.
    len: usize,
    /// The amount of padding that remains to be emitted.
    padding_left: Option<usize>,
}

impl<I: Iterator<Item = bool>> Pad<I> {
    fn new(inner: I) -> Self {
        Pad {
            inner,
            len: 0,
            padding_left: None,
        }
    }
}

impl<I: Iterator<Item = bool>> Iterator for Pad<I> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have identified the required padding, the inner iterator has ended,
            // and we will never poll it again.
            if let Some(n) = self.padding_left.as_mut() {
                if *n == 0 {
                    // Either we already emitted all necessary padding, or there was no
                    // padding required.
                    break None;
                } else {
                    // Emit the next padding bit.
                    *n -= 1;
                    break Some(false);
                }
            } else if let Some(ret) = self.inner.next() {
                // We haven't reached the end of the inner iterator yet.
                self.len += 1;
                assert!(self.len <= K * C);
                break Some(ret);
            } else {
                // Inner iterator just ended, so we now know its length.
                let rem = self.len % K;
                if rem > 0 {
                    // The inner iterator requires padding in the range [1,K).
                    self.padding_left = Some(K - rem);
                } else {
                    // No padding required.
                    self.padding_left = Some(0);
                }
            }
        }
    }
}

/// A domain in which $\mathsf{SinsemillaHashToPoint}$ and $\mathsf{SinsemillaHash}$ can
/// be used.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct HashDomain {
    Q: pallas::Point,
}

impl HashDomain {
    /// Constructs a new `HashDomain` with a specific prefix string.
    pub fn new(domain: &str) -> Self {
        HashDomain {
            Q: pallas::Point::hash_to_curve(Q_PERSONALIZATION)(domain.as_bytes()),
        }
    }

    /// $\mathsf{SinsemillaHashToPoint}$ from [§ 5.4.1.9][concretesinsemillahash].
    ///
    /// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
    pub fn hash_to_point(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Point> {
        self.hash_to_point_with_progress(msg, &mut || {})
    }

    /// [`HashDomain::hash_to_point`], calling `progress` after each [`K`]-bit piece of
    /// the message, for a caller on a slow device that must report progress while it
    /// runs. With `computed-generators` each piece is a hash-to-curve, so one
    /// NoteCommit takes ~0.7 s on a Cortex-M33.
    pub fn hash_to_point_with_progress(
        &self,
        msg: impl Iterator<Item = bool>,
        progress: &mut dyn FnMut(),
    ) -> CtOption<pallas::Point> {
        self.hash_to_point_inner(msg, progress).into()
    }

    #[allow(non_snake_case)]
    fn hash_to_point_inner(
        &self,
        msg: impl Iterator<Item = bool>,
        progress: &mut dyn FnMut(),
    ) -> IncompletePoint {
        let padded: Vec<_> = Pad::new(msg).collect();
        #[cfg(feature = "computed-generators")]
        let hasher = pallas::Point::hash_to_curve(S_PERSONALIZATION);

        padded
            .chunks(K)
            .fold(IncompletePoint::from(self.Q), |acc, chunk| {
                let index = lebs2ip_k(chunk.try_into().expect("correct length"));
                #[cfg(not(feature = "computed-generators"))]
                let (S_x, S_y) = SINSEMILLA_S[index as usize];
                #[cfg(not(feature = "computed-generators"))]
                let S_chunk = pallas::Affine::from_xy(S_x, S_y).unwrap();
                #[cfg(feature = "computed-generators")]
                let S_chunk = IncompletePoint::from(hasher(&index.to_le_bytes()));
                let acc = (acc + S_chunk) + acc;
                progress();
                acc
            })
    }

    /// $\mathsf{SinsemillaHash}$ from [§ 5.4.1.9][concretesinsemillahash].
    ///
    /// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
    ///
    /// # Panics
    ///
    /// This panics if the message length is greater than [`K`] * [`C`]
    pub fn hash(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Base> {
        extract_p_bottom(self.hash_to_point(msg))
    }

    /// Constructs a new `HashDomain` from a given `Q`.
    ///
    /// This is only for testing use.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn from_Q(Q: pallas::Point) -> Self {
        HashDomain { Q }
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn Q(&self) -> pallas::Point {
        self.Q
    }
}

/// A domain in which $\mathsf{SinsemillaCommit}$ and $\mathsf{SinsemillaShortCommit}$ can
/// be used.
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct CommitDomain {
    M: HashDomain,
    R: pallas::Point,
}

impl CommitDomain {
    /// Constructs a new `CommitDomain` with a specific prefix string.
    pub fn new(domain: &str) -> Self {
        let m_prefix = format!("{}-M", domain);
        let r_prefix = format!("{}-r", domain);
        let hasher_r = pallas::Point::hash_to_curve(&r_prefix);
        CommitDomain {
            M: HashDomain::new(&m_prefix),
            R: hasher_r(&[]),
        }
    }

    /// $\mathsf{SinsemillaCommit}$ from [§ 5.4.8.4][concretesinsemillacommit].
    ///
    /// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
    #[allow(non_snake_case)]
    pub fn commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Point> {
        self.commit_with_progress(msg, r, &mut || {})
    }

    /// [`CommitDomain::commit`], calling `progress` as
    /// [`HashDomain::hash_to_point_with_progress`] does.
    pub fn commit_with_progress(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
        progress: &mut dyn FnMut(),
    ) -> CtOption<pallas::Point> {
        // We use complete addition for the blinding factor.
        CtOption::<pallas::Point>::from(self.M.hash_to_point_inner(msg, progress))
            .map(|p| p + Wnaf::new().scalar(r).base(self.R))
    }

    /// $\mathsf{SinsemillaShortCommit}$ from [§ 5.4.8.4][concretesinsemillacommit].
    ///
    /// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
    pub fn short_commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Base> {
        self.short_commit_with_progress(msg, r, &mut || {})
    }

    /// [`CommitDomain::short_commit`], calling `progress` as
    /// [`HashDomain::hash_to_point_with_progress`] does.
    pub fn short_commit_with_progress(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
        progress: &mut dyn FnMut(),
    ) -> CtOption<pallas::Base> {
        extract_p_bottom(self.commit_with_progress(msg, r, progress))
    }

    /// Returns the Sinsemilla $R$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn R(&self) -> pallas::Point {
        self.R
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn Q(&self) -> pallas::Point {
        self.M.Q
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{Pad, K};
    use pasta_curves::{arithmetic::CurveExt, pallas};

    #[test]
    fn pad() {
        assert_eq!(
            Pad::new([].iter().cloned()).collect::<Vec<_>>(),
            vec![false; 0]
        );
        assert_eq!(
            Pad::new([true].iter().cloned()).collect::<Vec<_>>(),
            vec![true, false, false, false, false, false, false, false, false, false]
        );
        assert_eq!(
            Pad::new([true, true].iter().cloned()).collect::<Vec<_>>(),
            vec![true, true, false, false, false, false, false, false, false, false]
        );
        assert_eq!(
            Pad::new([true, true, true].iter().cloned()).collect::<Vec<_>>(),
            vec![true, true, true, false, false, false, false, false, false, false]
        );
        assert_eq!(
            Pad::new(
                [true, true, false, true, false, true, false, true, false, true]
                    .iter()
                    .cloned()
            )
            .collect::<Vec<_>>(),
            vec![true, true, false, true, false, true, false, true, false, true]
        );
        assert_eq!(
            Pad::new(
                [true, true, false, true, false, true, false, true, false, true, true]
                    .iter()
                    .cloned()
            )
            .collect::<Vec<_>>(),
            vec![
                true, true, false, true, false, true, false, true, false, true, true, false, false,
                false, false, false, false, false, false, false
            ]
        );
    }

    #[test]
    fn progress_is_reported_after_each_piece() {
        use super::{CommitDomain, HashDomain};

        // 21 bits: two full pieces and a padded third.
        let msg = [true, false, true].repeat(7);
        let hash = HashDomain::new("z.cash:test-progress");
        let mut calls = 0;
        let with_progress =
            hash.hash_to_point_with_progress(msg.iter().copied(), &mut || calls += 1);
        assert_eq!(calls, 3);
        assert_eq!(
            Option::<pallas::Point>::from(with_progress),
            Option::from(hash.hash_to_point(msg.iter().copied())),
        );

        let commit = CommitDomain::new("z.cash:test-progress");
        let r = pallas::Scalar::from(7);
        let mut calls = 0;
        let with_progress =
            commit.short_commit_with_progress(msg.iter().copied(), &r, &mut || calls += 1);
        assert_eq!(calls, 3);
        assert_eq!(
            Option::<pallas::Base>::from(with_progress),
            Option::from(commit.short_commit(msg.iter().copied(), &r)),
        );
    }

    #[test]
    fn sinsemilla_s() {
        use super::sinsemilla_s::SINSEMILLA_S;
        use group::Curve;
        use pasta_curves::arithmetic::CurveAffine;

        let hasher = pallas::Point::hash_to_curve(super::S_PERSONALIZATION);

        for j in 0..(1u32 << K) {
            let computed = {
                let point = hasher(&j.to_le_bytes()).to_affine().coordinates().unwrap();
                (*point.x(), *point.y())
            };
            let actual = SINSEMILLA_S[j as usize];
            assert_eq!(computed, actual);
        }
    }

    #[cfg(feature = "computed-generators")]
    #[test]
    fn computed_generators() {
        use super::{lebs2ip_k, HashDomain, IncompletePoint, SINSEMILLA_S};
        use pasta_curves::arithmetic::CurveAffine;

        let domain = HashDomain::new("z.cash:Sinsemilla-computed-generators-test");

        for j in 0..(1u32 << K) {
            let msg = (0..K).map(|i| (j & (1 << i)) != 0);
            let actual: Option<pallas::Point> = domain.hash_to_point(msg).into();

            let (s_x, s_y) = SINSEMILLA_S[j as usize];
            let s = pallas::Affine::from_xy(s_x, s_y).unwrap();
            let q = IncompletePoint::from(domain.Q);
            let expected: subtle::CtOption<pallas::Point> = ((q + s) + q).into();

            assert_eq!(actual, expected.into());
        }

        for msg in [
            vec![],
            vec![true],
            vec![
                true, false, true, true, false, false, true, false, true, false, true,
            ],
        ] {
            let actual: Option<pallas::Point> = domain.hash_to_point(msg.iter().copied()).into();
            let padded: Vec<_> = Pad::new(msg.into_iter()).collect();
            let expected: subtle::CtOption<pallas::Point> = padded
                .chunks(K)
                .fold(IncompletePoint::from(domain.Q), |acc, chunk| {
                    let index = lebs2ip_k(chunk.try_into().expect("correct length"));
                    let (s_x, s_y) = SINSEMILLA_S[index as usize];
                    let s = pallas::Affine::from_xy(s_x, s_y).unwrap();
                    (acc + s) + acc
                })
                .into();

            assert_eq!(actual, expected.into());
        }
    }
}
