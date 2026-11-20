//! A from-scratch HNSW (Hierarchical Navigable Small World) approximate-nearest-
//! neighbor index. In-memory, operating on `(MemoryId, vector)` pairs.
//!
//! The graph has multiple layers: higher layers are sparse "express lanes" for
//! coarse navigation, layer 0 holds every node. Search greedily descends from a
//! single entry point through the upper layers, then explores layer 0 with a
//! beam of width `ef`. Crate-internal; the store wires it into `search`.

// Consumed by the store in the integration task; unreachable in a non-test build
// until then.
#![allow(dead_code)]

use crate::index::cosine;
use crate::types::MemoryId;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

const M: usize = 16; // max neighbors per node on layers > 0
const M0: usize = 32; // max neighbors on layer 0 (denser)
const EF_CONSTRUCTION: usize = 100; // beam width while inserting
const MAX_LEVEL: usize = 16; // safety cap on randomized levels

fn level_mult() -> f64 {
    1.0 / (M as f64).ln()
}

/// Deterministic xorshift64 PRNG — avoids a `rand` dependency and keeps layer
/// assignment reproducible in tests.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        // Guard against the all-zero state, which xorshift can't escape.
        Xorshift(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A float in (0, 1).
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
}

struct Node {
    id: MemoryId,
    vector: Vec<f32>,
    /// `neighbors[layer]` = indices (into `nodes`) this node links to on `layer`.
    neighbors: Vec<Vec<u32>>,
}

pub(crate) struct HnswIndex {
    nodes: Vec<Node>,
    /// Maps a `MemoryId` to its node index, so `insert` is idempotent: inserting
    /// an id that already exists is a no-op. This keeps the index consistent even
    /// if a concurrent lazy-build and an incremental `add` both try to insert the
    /// same row (the build reads committed storage; the add re-inserts).
    ids: HashMap<MemoryId, u32>,
    entry: Option<u32>,
    top_layer: usize,
    rng: Xorshift,
}

impl HnswIndex {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            ids: HashMap::new(),
            entry: None,
            top_layer: 0,
            rng: Xorshift::new(0x2545F4914F6CDD1D),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    fn distance(&self, query: &[f32], idx: u32) -> f32 {
        // Cosine distance: 0 = identical direction, up to 2 = opposite.
        1.0 - cosine(query, &self.nodes[idx as usize].vector)
    }

    fn random_level(&mut self) -> usize {
        let r = self.rng.next_f64();
        ((-r.ln()) * level_mult()).floor() as usize
        // (caller-side cap applied in `insert`)
    }

    /// Insert a vector. Idempotency/dedup is the store's job; this just adds a node.
    pub(crate) fn insert(&mut self, id: MemoryId, vector: Vec<f32>) {
        // Idempotent: never add a second node for an id already in the index.
        if self.ids.contains_key(&id) {
            return;
        }
        let level = self.random_level().min(MAX_LEVEL);
        // Node indices are u32; the store rebuilds the index per process and caps
        // well below 4B nodes in practice, so this cast can't realistically wrap.
        let new_idx = self.nodes.len() as u32;
        self.ids.insert(id, new_idx);
        self.nodes.push(Node {
            id,
            vector,
            neighbors: vec![Vec::new(); level + 1],
        });

        let entry = match self.entry {
            None => {
                self.entry = Some(new_idx);
                self.top_layer = level;
                return;
            }
            Some(e) => e,
        };

        let query = self.nodes[new_idx as usize].vector.clone();
        let mut ep = entry;

        // Coarse descent through layers above the new node's top level.
        if self.top_layer > level {
            for layer in ((level + 1)..=self.top_layer).rev() {
                if let Some(&(_, nearest)) = self.search_layer(&query, &[ep], layer, 1).first() {
                    ep = nearest;
                }
            }
        }

        // Connect on each layer the new node participates in.
        let start_layer = level.min(self.top_layer);
        for layer in (0..=start_layer).rev() {
            let candidates = self.search_layer(&query, &[ep], layer, EF_CONSTRUCTION);
            let max_degree = if layer == 0 { M0 } else { M };
            let selected = self.select_neighbors_heuristic(&candidates, max_degree);

            for &neighbor in &selected {
                self.nodes[new_idx as usize].neighbors[layer].push(neighbor);
                self.nodes[neighbor as usize].neighbors[layer].push(new_idx);
                self.prune(neighbor, layer, max_degree);
            }
            if let Some(&(_, nearest)) = candidates.first() {
                ep = nearest;
            }
        }

        if level > self.top_layer {
            self.top_layer = level;
            self.entry = Some(new_idx);
        }
    }

    /// Keep a node's degree on a layer within `max_degree` using the same diversity
    /// heuristic as insertion, so a far-but-important link (e.g. to an outlier) is
    /// not dropped merely for being the farthest.
    fn prune(&mut self, node: u32, layer: usize, max_degree: usize) {
        if self.nodes[node as usize].neighbors[layer].len() <= max_degree {
            return;
        }
        let anchor = self.nodes[node as usize].vector.clone();
        let mut scored: Vec<(f32, u32)> = self.nodes[node as usize].neighbors[layer]
            .iter()
            .map(|&n| (self.distance(&anchor, n), n))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        self.nodes[node as usize].neighbors[layer] =
            self.select_neighbors_heuristic(&scored, max_degree);
    }

    /// HNSW neighbor-selection heuristic: from `candidates` (ascending distance to
    /// `base`), keep one only if it is closer to `base` than to every already-kept
    /// neighbor. This retains diverse, long-range links so outlier nodes keep
    /// in-edges and stay reachable — the naive "keep the M nearest" leaves outliers
    /// orphaned. Pads with the nearest leftovers so degree/connectivity stays high.
    fn select_neighbors_heuristic(&self, candidates: &[(f32, u32)], m: usize) -> Vec<u32> {
        let mut kept: Vec<u32> = Vec::new();
        for &(dist_to_base, cand) in candidates {
            if kept.len() >= m {
                break;
            }
            let cand_vec = &self.nodes[cand as usize].vector;
            let diverse = kept
                .iter()
                .all(|&k| dist_to_base < 1.0 - cosine(cand_vec, &self.nodes[k as usize].vector));
            if diverse {
                kept.push(cand);
            }
        }
        if kept.len() < m {
            for &(_, cand) in candidates {
                if kept.len() >= m {
                    break;
                }
                if !kept.contains(&cand) {
                    kept.push(cand);
                }
            }
        }
        kept
    }

    /// Greedy beam search on a single layer. Returns `(distance, idx)` ascending
    /// by distance, at most `ef` entries.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        layer: usize,
        ef: usize,
    ) -> Vec<(f32, u32)> {
        let mut visited: HashSet<u32> = HashSet::new();
        // Explore the closest unexpanded candidate first (min-heap via Reverse).
        let mut frontier: BinaryHeap<Reverse<Ranked>> = BinaryHeap::new();
        // Keep the best `ef` found so far; peek() is the farthest (max-heap).
        let mut best: BinaryHeap<Ranked> = BinaryHeap::new();

        for &ep in entry_points {
            let d = self.distance(query, ep);
            if visited.insert(ep) {
                frontier.push(Reverse(Ranked { dist: d, idx: ep }));
                best.push(Ranked { dist: d, idx: ep });
            }
        }
        while best.len() > ef {
            best.pop();
        }

        while let Some(Reverse(current)) = frontier.pop() {
            let farthest = best.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if current.dist > farthest && best.len() >= ef {
                break;
            }
            let neighbors = self.nodes[current.idx as usize]
                .neighbors
                .get(layer)
                .cloned()
                .unwrap_or_default();
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    let d = self.distance(query, neighbor);
                    let farthest = best.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                    if d < farthest || best.len() < ef {
                        frontier.push(Reverse(Ranked {
                            dist: d,
                            idx: neighbor,
                        }));
                        best.push(Ranked {
                            dist: d,
                            idx: neighbor,
                        });
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = best.into_iter().map(|r| (r.dist, r.idx)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }

    /// Approximate `k` nearest `MemoryId`s to `query`, using beam width `ef`.
    pub(crate) fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<MemoryId> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let mut ep = entry;
        for layer in (1..=self.top_layer).rev() {
            if let Some(&(_, nearest)) = self.search_layer(query, &[ep], layer, 1).first() {
                ep = nearest;
            }
        }
        self.search_layer(query, &[ep], 0, ef.max(k))
            .into_iter()
            .take(k)
            .map(|(_, idx)| self.nodes[idx as usize].id)
            .collect()
    }
}

/// A `(distance, idx)` pair with a total order (distance, then idx) so it can go
/// in a `BinaryHeap` despite `f32` not being `Ord`.
#[derive(Clone, Copy)]
struct Ranked {
    dist: f32,
    idx: u32,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.idx.cmp(&other.idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mid(n: u64) -> MemoryId {
        MemoryId::new(n)
    }

    #[test]
    fn empty_search_is_empty() {
        let idx = HnswIndex::new();
        assert!(idx.search(&[1.0, 0.0], 5, 16).is_empty());
    }

    #[test]
    fn single_insert_is_found() {
        let mut idx = HnswIndex::new();
        idx.insert(mid(1), vec![1.0, 0.0]);
        assert_eq!(idx.search(&[1.0, 0.0], 1, 16), vec![mid(1)]);
    }

    #[test]
    fn insert_is_idempotent_on_repeated_id() {
        let mut idx = HnswIndex::new();
        idx.insert(mid(1), vec![1.0, 0.0]);
        idx.insert(mid(1), vec![1.0, 0.0]); // same id again -> no-op
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.search(&[1.0, 0.0], 5, 16), vec![mid(1)]);
    }

    #[test]
    fn returns_nearest_first_on_small_set() {
        let mut idx = HnswIndex::new();
        idx.insert(mid(1), vec![1.0, 0.0]);
        idx.insert(mid(2), vec![0.0, 1.0]);
        idx.insert(mid(3), vec![0.9, 0.1]);
        let res = idx.search(&[1.0, 0.0], 3, 32);
        assert_eq!(res[0], mid(1)); // exact match closest
        assert_eq!(res[1], mid(3)); // then the near-aligned one
    }

    #[test]
    fn recall_at_10_matches_brute_force() {
        let mut rng = Xorshift::new(42);
        let dim = 16;
        let n = 500usize;
        let mut idx = HnswIndex::new();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_f64() as f32 - 0.5).collect();
            idx.insert(mid(i as u64), v.clone());
            vectors.push(v);
        }

        let mut hits = 0;
        for qi in 0..20 {
            let query = &vectors[qi * 10];
            let true_nearest = (0..n)
                .min_by(|&a, &b| {
                    (1.0 - cosine(query, &vectors[a]))
                        .total_cmp(&(1.0 - cosine(query, &vectors[b])))
                })
                .unwrap();
            if idx
                .search(query, 10, 64)
                .contains(&mid(true_nearest as u64))
            {
                hits += 1;
            }
        }
        assert!(hits >= 18, "recall@10 too low: {hits}/20");
    }

    #[test]
    fn self_recall_is_near_perfect() {
        // Querying an inserted vector should return that exact node — a direct check
        // that nodes stay reachable (no orphaning). The diversity heuristic keeps
        // this near-100% on a realistic spread distribution.
        let mut rng = Xorshift::new(99);
        let dim = 24;
        let n = 1500usize;
        let mut idx = HnswIndex::new();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_f64() as f32 - 0.5).collect();
            idx.insert(mid(i as u64), v.clone());
            vectors.push(v);
        }

        let sampled: Vec<usize> = (0..n).step_by(30).collect();
        let found = sampled
            .iter()
            .filter(|&&i| idx.search(&vectors[i], 1, 64).first() == Some(&mid(i as u64)))
            .count();
        assert!(
            found >= sampled.len() - 1,
            "self-recall@1 too low: {found}/{}",
            sampled.len()
        );
    }
}
