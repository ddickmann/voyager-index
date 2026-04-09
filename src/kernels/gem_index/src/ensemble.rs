//! Multi-index ensemble for modality-specific codebooks with shared routing
//! and reciprocal rank fusion (RRF).

use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop};
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

use crate::graph::{build_graph, GemGraph};
use crate::search::beam_search;

/// Per-modality segment: codebook + codes + graph.
pub struct ModalitySegment {
    pub codebook: TwoStageCodebook,
    pub flat_codes: FlatDocCodes,
    pub doc_profiles: Vec<DocProfile>,
    pub postings: ClusterPostings,
    pub graph: GemGraph,
    pub ctop_r: usize,
}

/// Ensemble segment wrapping multiple per-modality segments.
pub struct EnsembleSegment {
    pub segments: Vec<ModalitySegment>,
    pub doc_ids: Vec<u64>,
    pub n_modalities: u8,
    /// Per-doc, per-modality: (token_start, token_end) into the modality-local space
    pub modality_slices: Vec<Vec<(usize, usize)>>,
    pub dim: usize,
}

impl EnsembleSegment {
    /// Build ensemble from multi-modal token embeddings.
    ///
    /// - `all_vectors`: (N, dim) flat row-major vectors
    /// - `doc_ids`: external document IDs
    /// - `doc_offsets`: (start, end) token ranges into `all_vectors` per doc
    /// - `modality_tags`: per-token modality ID (0..n_modalities-1)
    /// - `n_modalities`: number of modalities
    pub fn build(
        all_vectors: &[f32],
        dim: usize,
        doc_ids: &[u64],
        doc_offsets: &[(usize, usize)],
        modality_tags: &[u8],
        n_modalities: u8,
        n_fine: usize,
        n_coarse: usize,
        max_degree: usize,
        ef_construction: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
    ) -> Self {
        let n_docs = doc_ids.len();
        let nm = n_modalities as usize;

        // Partition tokens by modality per doc
        let mut per_modality_vecs: Vec<Vec<f32>> = vec![Vec::new(); nm];
        let mut per_modality_doc_offsets: Vec<Vec<(usize, usize)>> = vec![vec![]; nm];
        let mut modality_slices: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n_docs);

        for &(start, end) in doc_offsets.iter() {
            let mut doc_slices = vec![(0usize, 0usize); nm];
            for m in 0..nm {
                let cur_start = per_modality_vecs[m].len() / dim;
                let mut count = 0usize;
                for (t, &tag) in modality_tags.iter().enumerate().take(end).skip(start) {
                    if tag as usize == m {
                        let vstart = t * dim;
                        let vend = (t + 1) * dim;
                        per_modality_vecs[m].extend_from_slice(&all_vectors[vstart..vend]);
                        count += 1;
                    }
                }
                doc_slices[m] = (cur_start, cur_start + count);
            }
            modality_slices.push(doc_slices.clone());

            // Track offsets per modality
            for m in 0..nm {
                per_modality_doc_offsets[m].push(doc_slices[m]);
            }
        }

        // Build per-modality segments
        let mut segments = Vec::with_capacity(nm);
        for m in 0..nm {
            let vecs = &per_modality_vecs[m];
            let offsets = &per_modality_doc_offsets[m];
            let n_vectors = vecs.len() / dim;

            // Skip empty modalities
            if n_vectors == 0 {
                segments.push(ModalitySegment {
                    codebook: TwoStageCodebook::build(&[], 0, dim, 1, 1, 1, 42),
                    flat_codes: FlatDocCodes::new(),
                    doc_profiles: vec![],
                    postings: ClusterPostings::new(0),
                    graph: GemGraph {
                        levels: vec![],
                        shortcuts: vec![],
                        shortcut_generations: vec![],
                        node_levels: vec![],
                        entry_point: 0,
                        max_degree,
                    },
                    ctop_r,
                });
                continue;
            }

            let actual_n_fine = n_fine.min(n_vectors);
            let actual_n_coarse = n_coarse.min(actual_n_fine / 2).max(1);

            let mut codebook = TwoStageCodebook::build(
                vecs, n_vectors, dim, actual_n_fine, actual_n_coarse, max_kmeans_iter, 42,
            );
            let all_assignments = codebook.assign_vectors(vecs, n_vectors);

            let mut doc_centroid_sets = Vec::with_capacity(n_docs);
            for &(start, end) in offsets {
                if end > start && end <= all_assignments.len() {
                    doc_centroid_sets.push(all_assignments[start..end].to_vec());
                } else {
                    doc_centroid_sets.push(Vec::new());
                }
            }
            codebook.update_idf(&doc_centroid_sets);

            codebook.refine_centroids_idf(vecs, n_vectors, 3);
            let all_assignments = codebook.assign_vectors(vecs, n_vectors);
            let mut doc_centroid_sets = Vec::with_capacity(n_docs);
            for &(start, end) in offsets {
                if end > start && end <= all_assignments.len() {
                    doc_centroid_sets.push(all_assignments[start..end].to_vec());
                } else {
                    doc_centroid_sets.push(Vec::new());
                }
            }
            codebook.update_idf(&doc_centroid_sets);

            let mut doc_profiles = Vec::with_capacity(n_docs);
            let mut flat_codes = FlatDocCodes::new();
            let mut postings = ClusterPostings::new(actual_n_coarse);
            for (doc_idx, cids) in doc_centroid_sets.into_iter().enumerate() {
                let ctop = if cids.is_empty() {
                    Vec::new()
                } else {
                    compute_ctop(&codebook, &cids, ctop_r)
                };
                postings.add_doc(doc_idx as u32, &ctop);
                flat_codes.add_doc(&cids);
                doc_profiles.push(DocProfile {
                    centroid_ids: cids,
                    ctop,
                });
            }
            postings.compute_medoids(&codebook, &flat_codes);

            let graph = build_graph(
                vecs, dim, offsets, &codebook, &flat_codes,
                &doc_profiles, &postings, max_degree, ef_construction,
            );

            segments.push(ModalitySegment {
                codebook,
                flat_codes,
                doc_profiles,
                postings,
                graph,
                ctop_r,
            });
        }

        EnsembleSegment {
            segments,
            doc_ids: doc_ids.to_vec(),
            n_modalities,
            modality_slices,
            dim,
        }
    }

    /// Search each modality independently, fuse via RRF.
    pub fn search(
        &self,
        query_vectors: &[f32],
        n_query: usize,
        query_modality_tags: &[u8],
        k: usize,
        ef: usize,
        n_probes: usize,
    ) -> Vec<(u64, f32)> {
        if k == 0 || self.doc_ids.is_empty() {
            return Vec::new();
        }

        let nm = self.n_modalities as usize;
        let dim = self.dim;

        // Partition query tokens by modality
        let mut modality_query_vecs: Vec<Vec<f32>> = vec![Vec::new(); nm];
        let mut modality_query_counts: Vec<usize> = vec![0; nm];
        for qi in 0..n_query {
            let raw_tag = query_modality_tags.get(qi).copied().unwrap_or(0) as usize;
            let m = if raw_tag < nm { raw_tag } else { 0 };
            let vstart = qi * dim;
            let vend = (qi + 1) * dim;
            modality_query_vecs[m].extend_from_slice(&query_vectors[vstart..vend]);
            modality_query_counts[m] += 1;
        }

        // Search each modality and collect per-modality ranked results
        let k_rrf = 60.0f32;
        let mut rrf_scores: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();

        for m in 0..nm {
            let n_q = modality_query_counts[m];
            if n_q == 0 {
                continue;
            }
            let seg = &self.segments[m];
            if seg.graph.levels.is_empty() {
                continue;
            }

            let mut query_scores = seg.codebook.compute_query_centroid_scores(
                &modality_query_vecs[m], n_q,
            );
            seg.codebook.apply_idf_weights(&mut query_scores, n_q);
            let n_fine = seg.codebook.n_fine;

            // Get entry points from cluster reps
            let query_cids = seg.codebook.assign_vectors(&modality_query_vecs[m], n_q);
            let query_ctop = compute_ctop(&seg.codebook, &query_cids, n_probes.max(seg.ctop_r));
            let mut entries: Vec<u32> = seg.postings.representatives_for_clusters(&query_ctop);
            entries.push(seg.graph.entry_point);
            entries.sort_unstable();
            entries.dedup();

            let results = beam_search(
                &seg.graph.levels[0],
                Some(&seg.graph.shortcuts),
                &entries,
                &query_scores,
                n_q,
                &seg.flat_codes,
                n_fine,
                ef,
                None,
                false,
            );

            // RRF fusion: 1-based rank to match hybrid_manager convention
            for (rank_0, &(int_idx, _score)) in results.iter().enumerate() {
                let rank = rank_0 + 1;
                let doc_idx = int_idx as usize;
                let rrf_contribution = 1.0 / (k_rrf + rank as f32);
                *rrf_scores.entry(doc_idx).or_insert(0.0) += rrf_contribution;
            }
        }

        // Sort by fused RRF score (higher = better), convert to external IDs
        let mut sorted: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
        sorted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        sorted.truncate(k);

        sorted
            .into_iter()
            .filter_map(|(idx, score)| {
                self.doc_ids.get(idx).map(|&did| (did, score))
            })
            .collect()
    }
}
