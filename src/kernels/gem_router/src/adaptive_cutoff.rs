//! Adaptive per-document cluster cutoff via a lightweight decision tree.
//!
//! GEM Paper Section 4.4.2: Instead of a fixed ctop_r for all documents,
//! train a small decision tree that predicts the optimal number of top
//! clusters r based on per-document features (TF-IDF scores, document length).
//!
//! This reduces index redundancy for short/focused documents and prevents
//! recall loss for long/diverse documents.

use serde::{Deserialize, Serialize};

/// A single node in the binary decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Internal (split) node: split on feature[feature_idx] <= threshold
    Split {
        feature_idx: usize,
        threshold: f32,
        left: usize,  // index into CutoffTree::nodes
        right: usize, // index into CutoffTree::nodes
    },
    /// Leaf node: predict this value for r
    Leaf {
        value: usize,
    },
}

/// Axis-aligned binary decision tree for predicting per-document cluster cutoff r.
///
/// Features per document (vector of r_max + 1 values):
/// - features[0..r_max] = top-r_max TF-IDF cluster scores (padded with 0)
/// - features[r_max] = number of vectors in the document (normalized)
///
/// Trained using CART (information gain / variance reduction splitting).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutoffTree {
    pub nodes: Vec<TreeNode>,
    pub r_max: usize,
}

impl CutoffTree {
    /// Train a decision tree from feature vectors and integer labels.
    ///
    /// - `features`: &[Vec<f32>] — one feature vector per training sample
    /// - `labels`: &[usize] — target r value per sample
    /// - `max_depth`: tree depth limit (recommended: 6-8)
    /// - `r_max`: maximum cutoff value
    pub fn train(features: &[Vec<f32>], labels: &[usize], max_depth: usize, r_max: usize) -> Self {
        assert_eq!(features.len(), labels.len());
        if features.is_empty() {
            return Self {
                nodes: vec![TreeNode::Leaf { value: r_max.max(1) }],
                r_max,
            };
        }

        let n_features = features[0].len();
        let indices: Vec<usize> = (0..features.len()).collect();
        let mut nodes: Vec<TreeNode> = Vec::new();

        build_tree_recursive(
            &mut nodes,
            features,
            labels,
            &indices,
            0,
            max_depth,
            n_features,
            2, // min_samples_leaf
        );

        if nodes.is_empty() {
            nodes.push(TreeNode::Leaf { value: majority_label(labels, &indices) });
        }

        Self { nodes, r_max }
    }

    /// Predict the cluster cutoff r for a document given its feature vector.
    #[inline]
    pub fn predict(&self, features: &[f32]) -> usize {
        if self.nodes.is_empty() {
            return self.r_max.max(1);
        }
        let mut idx = 0;
        loop {
            match &self.nodes[idx] {
                TreeNode::Leaf { value } => return *value,
                TreeNode::Split { feature_idx, threshold, left, right } => {
                    let feat_val = if *feature_idx < features.len() {
                        features[*feature_idx]
                    } else {
                        0.0
                    };
                    idx = if feat_val <= *threshold { *left } else { *right };
                }
            }
        }
    }

    /// Serialize to bytes for persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Recursively build the tree using CART (Classification And Regression Trees).
fn build_tree_recursive(
    nodes: &mut Vec<TreeNode>,
    features: &[Vec<f32>],
    labels: &[usize],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    n_features: usize,
    min_samples_leaf: usize,
) -> usize {
    let node_idx = nodes.len();

    // Check stopping conditions
    if depth >= max_depth || indices.len() < 2 * min_samples_leaf || all_same(labels, indices) {
        nodes.push(TreeNode::Leaf {
            value: majority_label(labels, indices),
        });
        return node_idx;
    }

    // Find best split
    if let Some((best_feat, best_thresh, left_indices, right_indices)) =
        find_best_split(features, labels, indices, n_features, min_samples_leaf)
    {
        // Reserve this node's slot
        nodes.push(TreeNode::Leaf { value: 0 }); // placeholder

        let left_idx = build_tree_recursive(
            nodes, features, labels, &left_indices, depth + 1,
            max_depth, n_features, min_samples_leaf,
        );
        let right_idx = build_tree_recursive(
            nodes, features, labels, &right_indices, depth + 1,
            max_depth, n_features, min_samples_leaf,
        );

        nodes[node_idx] = TreeNode::Split {
            feature_idx: best_feat,
            threshold: best_thresh,
            left: left_idx,
            right: right_idx,
        };
        node_idx
    } else {
        nodes.push(TreeNode::Leaf {
            value: majority_label(labels, indices),
        });
        node_idx
    }
}

/// Find the best axis-aligned split using Gini impurity reduction.
#[allow(clippy::needless_range_loop)]
fn find_best_split(
    features: &[Vec<f32>],
    labels: &[usize],
    indices: &[usize],
    n_features: usize,
    min_samples_leaf: usize,
) -> Option<(usize, f32, Vec<usize>, Vec<usize>)> {
    let parent_impurity = gini_impurity(labels, indices);
    let n = indices.len() as f64;

    let mut best_gain = 0.0f64;
    let mut best_feat = 0;
    let mut best_thresh = 0.0f32;
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    for feat in 0..n_features {
        // Collect and sort feature values
        let mut feat_vals: Vec<(f32, usize)> = indices
            .iter()
            .map(|&i| (features[i][feat], i))
            .collect();
        feat_vals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        // Try splits between consecutive distinct values
        for split_point in min_samples_leaf..=(feat_vals.len().saturating_sub(min_samples_leaf)) {
            if split_point == 0 || split_point >= feat_vals.len() {
                continue;
            }
            let left_val = feat_vals[split_point - 1].0;
            let right_val = feat_vals[split_point].0;
            if (left_val - right_val).abs() < 1e-10 {
                continue;
            }
            let threshold = (left_val + right_val) * 0.5;

            let left_idx: Vec<usize> = feat_vals[..split_point].iter().map(|&(_, i)| i).collect();
            let right_idx: Vec<usize> = feat_vals[split_point..].iter().map(|&(_, i)| i).collect();

            let left_impurity = gini_impurity(labels, &left_idx);
            let right_impurity = gini_impurity(labels, &right_idx);

            let left_n = left_idx.len() as f64;
            let right_n = right_idx.len() as f64;
            let weighted = left_n / n * left_impurity + right_n / n * right_impurity;
            let gain = parent_impurity - weighted;

            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                best_thresh = threshold;
                best_left = left_idx;
                best_right = right_idx;
            }
        }
    }

    if best_gain > 1e-10 && !best_left.is_empty() && !best_right.is_empty() {
        Some((best_feat, best_thresh, best_left, best_right))
    } else {
        None
    }
}

fn gini_impurity(labels: &[usize], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let n = indices.len() as f64;
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &i in indices {
        *counts.entry(labels[i]).or_insert(0) += 1;
    }
    let mut impurity = 1.0f64;
    for &count in counts.values() {
        let p = count as f64 / n;
        impurity -= p * p;
    }
    impurity
}

fn majority_label(labels: &[usize], indices: &[usize]) -> usize {
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &i in indices {
        *counts.entry(labels[i]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
        .unwrap_or(1)
}

fn all_same(labels: &[usize], indices: &[usize]) -> bool {
    if indices.is_empty() {
        return true;
    }
    let first = labels[indices[0]];
    indices.iter().all(|&i| labels[i] == first)
}

/// Build feature vectors for training the adaptive cutoff tree.
///
/// For each document:
/// - features[0..r_max] = top-r_max TF-IDF cluster scores
/// - features[r_max] = normalized document length
pub fn build_doc_features(
    cluster_scores: &[Vec<f32>],
    doc_lengths: &[usize],
    r_max: usize,
) -> Vec<Vec<f32>> {
    let max_len = doc_lengths.iter().copied().max().unwrap_or(1).max(1) as f32;

    cluster_scores
        .iter()
        .zip(doc_lengths.iter())
        .map(|(scores, &len)| {
            let mut feat = vec![0.0f32; r_max + 1];
            let mut sorted_scores: Vec<f32> = scores.clone();
            sorted_scores.sort_unstable_by(|a, b| b.total_cmp(a));
            for (i, &s) in sorted_scores.iter().take(r_max).enumerate() {
                feat[i] = s;
            }
            feat[r_max] = len as f32 / max_len;
            feat
        })
        .collect()
}

/// Compute training labels for the adaptive cutoff tree.
///
/// For each (query, positive_doc) pair:
/// - C_query = union of top-t centroids for each query vector
/// - r_label = rank of first cluster in doc's sorted TF-IDF profile that intersects C_query
pub fn compute_training_labels(
    query_top_clusters: &[Vec<u32>],
    doc_sorted_clusters: &[Vec<u32>],
    r_max: usize,
) -> Vec<usize> {
    query_top_clusters
        .iter()
        .zip(doc_sorted_clusters.iter())
        .map(|(q_clusters, d_clusters)| {
            let q_set: std::collections::HashSet<u32> = q_clusters.iter().copied().collect();
            for (rank, &c) in d_clusters.iter().enumerate() {
                if q_set.contains(&c) {
                    return (rank + 1).min(r_max);
                }
            }
            r_max
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tree() {
        // Binary classification: if feature[0] <= 5.0, label = 1; else label = 2
        let features: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32, 0.5])
            .collect();
        let labels: Vec<usize> = (0..20)
            .map(|i| if i <= 10 { 1 } else { 2 })
            .collect();

        let tree = CutoffTree::train(&features, &labels, 4, 5);
        assert_eq!(tree.predict(&[3.0, 0.5]), 1);
        assert_eq!(tree.predict(&[15.0, 0.5]), 2);
    }

    #[test]
    fn test_single_class() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![3, 3];
        let tree = CutoffTree::train(&features, &labels, 4, 5);
        assert_eq!(tree.predict(&[0.0, 0.0]), 3);
    }

    #[test]
    fn test_empty_training() {
        let tree = CutoffTree::train(&[], &[], 4, 5);
        assert_eq!(tree.predict(&[0.0]), 5);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let features: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32])
            .collect();
        let labels: Vec<usize> = (0..10)
            .map(|i| if i < 5 { 1 } else { 2 })
            .collect();

        let tree = CutoffTree::train(&features, &labels, 3, 4);
        let bytes = tree.to_bytes();
        let restored = CutoffTree::from_bytes(&bytes).unwrap();
        assert_eq!(tree.predict(&[2.0]), restored.predict(&[2.0]));
        assert_eq!(tree.predict(&[8.0]), restored.predict(&[8.0]));
    }

    #[test]
    fn test_build_doc_features() {
        let scores = vec![
            vec![0.5, 0.3, 0.1, 0.05],
            vec![0.8, 0.1, 0.05, 0.02],
        ];
        let lengths = vec![10, 50];
        let feats = build_doc_features(&scores, &lengths, 3);
        assert_eq!(feats.len(), 2);
        assert_eq!(feats[0].len(), 4); // r_max=3, plus 1 for length
        assert!((feats[0][3] - 0.2).abs() < 1e-5); // 10/50
        assert!((feats[1][3] - 1.0).abs() < 1e-5); // 50/50
    }

    #[test]
    fn test_compute_training_labels() {
        let q_clusters = vec![vec![0, 1, 2], vec![5, 6, 7]];
        let d_clusters = vec![vec![3, 1, 0], vec![8, 9, 5]];
        let labels = compute_training_labels(&q_clusters, &d_clusters, 5);
        assert_eq!(labels[0], 2); // cluster 1 at rank 1 (0-indexed)
        assert_eq!(labels[1], 3); // cluster 5 at rank 2
    }
}
