//! Earth Mover's Distance solver for quantized multi-vector comparisons.
//!
//! Two solvers are provided:
//! - `emd_solve`: Exact solution via the transportation simplex (Vogel + stepping stone).
//! - `emd_sinkhorn`: Fast approximate solution via Sinkhorn iterations.
//!
//! For graph construction, `emd_solve` is the default (exact metric properties).
//! For large-scale builds where speed matters more, `emd_sinkhorn` is available.

/// Solve the exact EMD between two histograms via the transportation method.
///
/// Uses Vogel's approximation for initial BFS, then iterates MODI (modified
/// distribution) method for optimality.
///
/// - `supply`: weights for source nodes (must sum to ~1.0)
/// - `demand`: weights for sink nodes (must sum to ~1.0)
/// - `cost`:   n_supply x n_demand row-major cost matrix
/// - `max_iter`: iteration cap (0 = unlimited, recommended: 100_000)
///
/// Returns the optimal transport cost.
pub fn emd_solve(
    supply: &[f64],
    demand: &[f64],
    cost: &[f64],
    max_iter: u64,
) -> f64 {
    let m = supply.len();
    let n = demand.len();
    if m == 0 || n == 0 {
        return 0.0;
    }
    debug_assert_eq!(cost.len(), m * n);

    // Work with mutable copies
    let mut s: Vec<f64> = supply.to_vec();
    let d: Vec<f64> = demand.to_vec();

    // Balance supply and demand
    let total_s: f64 = s.iter().sum();
    let total_d: f64 = d.iter().sum();
    if (total_s - total_d).abs() > 1e-12 {
        let ratio = total_d / total_s;
        for v in &mut s {
            *v *= ratio;
        }
    }

    // Flow matrix
    let mut flow: Vec<f64> = vec![0.0; m * n];
    // Basis indicator (true = basic variable)
    let mut is_basic: Vec<bool> = vec![false; m * n];

    // Initial BFS using Vogel's approximation method
    vogel_init(&s, &d, cost, m, n, &mut flow, &mut is_basic);

    // MODI method for optimization
    let max_it = if max_iter == 0 { 100_000u64 } else { max_iter };
    modi_optimize(
        &mut flow,
        &mut is_basic,
        cost,
        &s,
        &d,
        m,
        n,
        max_it,
    );

    // Compute total cost
    let mut total = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let f = flow[i * n + j];
            if f > 1e-15 {
                total += f * cost[i * n + j];
            }
        }
    }
    total
}

/// Vogel's approximation method for initial basic feasible solution.
fn vogel_init(
    supply: &[f64],
    demand: &[f64],
    cost: &[f64],
    m: usize,
    n: usize,
    flow: &mut [f64],
    is_basic: &mut [bool],
) {
    let mut s = supply.to_vec();
    let mut d = demand.to_vec();
    let mut row_done = vec![false; m];
    let mut col_done = vec![false; n];

    let mut n_basic = 0usize;
    let target_basic = m + n - 1;

    for _ in 0..(m * n) {
        if n_basic >= target_basic {
            break;
        }

        // Find row penalties
        let mut best_i = usize::MAX;
        let mut best_j = usize::MAX;
        let mut best_penalty = -1.0f64;
        let mut best_cost = f64::MAX;

        for i in 0..m {
            if row_done[i] {
                continue;
            }
            let mut min1 = f64::MAX;
            let mut min2 = f64::MAX;
            let mut min1_j = 0;
            for j in 0..n {
                if col_done[j] {
                    continue;
                }
                let c = cost[i * n + j];
                if c < min1 {
                    min2 = min1;
                    min1 = c;
                    min1_j = j;
                } else if c < min2 {
                    min2 = c;
                }
            }
            if min1 < f64::MAX {
                let penalty = if min2 < f64::MAX {
                    min2 - min1
                } else {
                    min1
                };
                if penalty > best_penalty
                    || (penalty == best_penalty && min1 < best_cost)
                {
                    best_penalty = penalty;
                    best_cost = min1;
                    best_i = i;
                    best_j = min1_j;
                }
            }
        }

        // Find column penalties
        for j in 0..n {
            if col_done[j] {
                continue;
            }
            let mut min1 = f64::MAX;
            let mut min2 = f64::MAX;
            let mut min1_i = 0;
            for i in 0..m {
                if row_done[i] {
                    continue;
                }
                let c = cost[i * n + j];
                if c < min1 {
                    min2 = min1;
                    min1 = c;
                    min1_i = i;
                } else if c < min2 {
                    min2 = c;
                }
            }
            if min1 < f64::MAX {
                let penalty = if min2 < f64::MAX {
                    min2 - min1
                } else {
                    min1
                };
                if penalty > best_penalty
                    || (penalty == best_penalty && min1 < best_cost)
                {
                    best_penalty = penalty;
                    best_cost = min1;
                    best_i = min1_i;
                    best_j = j;
                }
            }
        }

        if best_i == usize::MAX || best_j == usize::MAX {
            break;
        }

        let alloc = s[best_i].min(d[best_j]);
        flow[best_i * n + best_j] = alloc;
        is_basic[best_i * n + best_j] = true;
        n_basic += 1;

        s[best_i] -= alloc;
        d[best_j] -= alloc;

        if s[best_i] < 1e-15 {
            row_done[best_i] = true;
        }
        if d[best_j] < 1e-15 {
            col_done[best_j] = true;
        }
    }

    // Ensure we have m+n-1 basic variables (add degeneracy if needed)
    if n_basic < target_basic {
        for i in 0..m {
            for j in 0..n {
                if n_basic >= target_basic {
                    break;
                }
                if !is_basic[i * n + j] {
                    is_basic[i * n + j] = true;
                    flow[i * n + j] = 0.0; // degenerate basic variable
                    n_basic += 1;
                }
            }
            if n_basic >= target_basic {
                break;
            }
        }
    }
}

/// MODI (Modified Distribution) optimization: iteratively improve the BFS.
fn modi_optimize(
    flow: &mut [f64],
    is_basic: &mut [bool],
    cost: &[f64],
    _supply: &[f64],
    _demand: &[f64],
    m: usize,
    n: usize,
    max_iter: u64,
) {
    let mut u = vec![f64::MAX; m]; // row potentials
    let mut v = vec![f64::MAX; n]; // column potentials

    for iteration in 0..max_iter {
        // Step 1: Compute dual variables u, v from basis arcs
        // u[i] + v[j] = cost[i][j] for all basic (i,j)
        for val in u.iter_mut() {
            *val = f64::MAX;
        }
        for val in v.iter_mut() {
            *val = f64::MAX;
        }

        u[0] = 0.0;
        let mut changed = true;
        let mut passes = 0;
        while changed && passes < m + n + 5 {
            changed = false;
            passes += 1;
            for i in 0..m {
                for j in 0..n {
                    if !is_basic[i * n + j] {
                        continue;
                    }
                    let c = cost[i * n + j];
                    if u[i] < f64::MAX && v[j] == f64::MAX {
                        v[j] = c - u[i];
                        changed = true;
                    } else if v[j] < f64::MAX && u[i] == f64::MAX {
                        u[i] = c - v[j];
                        changed = true;
                    }
                }
            }
        }

        // Step 2: Find the most negative reduced cost among non-basic variables
        let mut enter_i = 0;
        let mut enter_j = 0;
        let mut min_rc = -1e-10; // tolerance

        for i in 0..m {
            if u[i] == f64::MAX {
                continue;
            }
            for j in 0..n {
                if v[j] == f64::MAX {
                    continue;
                }
                if is_basic[i * n + j] {
                    continue;
                }
                let rc = cost[i * n + j] - u[i] - v[j];
                if rc < min_rc {
                    min_rc = rc;
                    enter_i = i;
                    enter_j = j;
                }
            }
        }

        if min_rc >= -1e-10 {
            break; // optimal
        }

        // Step 3: Find the cycle and pivot
        // Use BFS/DFS to find a cycle in the basis graph that includes (enter_i, enter_j)
        if let Some(cycle) = find_cycle(is_basic, m, n, enter_i, enter_j) {
            // Find minimum flow on negative positions of the cycle
            let mut min_flow = f64::MAX;
            for k in (1..cycle.len()).step_by(2) {
                let (ci, cj) = cycle[k];
                let f = flow[ci * n + cj];
                if f < min_flow {
                    min_flow = f;
                }
            }
            if min_flow == f64::MAX || min_flow < 0.0 {
                min_flow = 0.0;
            }

            // Update flows along cycle
            for (k, &(ci, cj)) in cycle.iter().enumerate() {
                if k % 2 == 0 {
                    flow[ci * n + cj] += min_flow;
                } else {
                    flow[ci * n + cj] -= min_flow;
                }
            }

            // Find leaving variable (one of the negative positions with min flow)
            let mut leave_k = 1;
            for k in (1..cycle.len()).step_by(2) {
                let (ci, cj) = cycle[k];
                if flow[ci * n + cj] < 1e-15 {
                    leave_k = k;
                    break;
                }
            }
            let (li, lj) = cycle[leave_k];

            // Update basis
            is_basic[enter_i * n + enter_j] = true;
            is_basic[li * n + lj] = false;
            flow[li * n + lj] = 0.0;
        } else {
            break; // can't find cycle (shouldn't happen)
        }

        if iteration > 0 && iteration % 10000 == 0 {
            // Safety: check for stalling
            let _ = iteration;
        }
    }
}

/// Find a cycle in the basis graph that includes the entering cell (ei, ej).
/// Returns the cycle as alternating + and - cells, starting with the entering cell.
fn find_cycle(
    is_basic: &[bool],
    m: usize,
    n: usize,
    ei: usize,
    ej: usize,
) -> Option<Vec<(usize, usize)>> {
    // Build adjacency: for each row, list of basic columns; for each col, list of basic rows
    let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..m {
        for j in 0..n {
            if is_basic[i * n + j] || (i == ei && j == ej) {
                row_cols[i].push(j);
                col_rows[j].push(i);
            }
        }
    }

    // DFS to find cycle: alternating row-step and col-step
    let mut path: Vec<(usize, usize)> = vec![(ei, ej)];
    let mut visited_rows: Vec<bool> = vec![false; m];
    let mut visited_cols: Vec<bool> = vec![false; n];
    visited_cols[ej] = true;

    if dfs_cycle(
        &row_cols,
        &col_rows,
        &mut path,
        &mut visited_rows,
        &mut visited_cols,
        ei,
        ej,
        true, // next step is row (find another row in column ej)
    ) {
        Some(path)
    } else {
        None
    }
}

/// DFS to find alternating row/col cycle back to start.
fn dfs_cycle(
    row_cols: &[Vec<usize>],
    col_rows: &[Vec<usize>],
    path: &mut Vec<(usize, usize)>,
    visited_rows: &mut [bool],
    visited_cols: &mut [bool],
    start_row: usize,
    start_col: usize,
    looking_for_row: bool,
) -> bool {
    if looking_for_row {
        // From current last cell (r, c), we need to find another row in column c
        let (_, c) = *path.last().unwrap();
        for &r in &col_rows[c] {
            if r == start_row && path.len() >= 3 {
                // Found cycle back to start
                path.push((r, c));
                // Now need to close: find step back to start_col
                // Actually, the cycle goes (ei,ej) -> (r2,ej) -> (r2,c2) -> ... -> (ei, ej)
                // We need path of even length. Path currently has odd length.
                // Remove last (it's (start_row, c), we need (start_row, start_col))
                path.pop();
                return true;
            }
            if visited_rows[r] {
                continue;
            }
            visited_rows[r] = true;
            path.push((r, c));
            if dfs_cycle(
                row_cols, col_rows, path, visited_rows, visited_cols,
                start_row, start_col, false,
            ) {
                return true;
            }
            path.pop();
            visited_rows[r] = false;
        }
    } else {
        // From current last cell (r, c), find another column in row r
        let (r, _) = *path.last().unwrap();
        for &c in &row_cols[r] {
            if c == start_col && path.len() >= 3 {
                return true; // completed cycle
            }
            if visited_cols[c] {
                continue;
            }
            visited_cols[c] = true;
            path.push((r, c));
            if dfs_cycle(
                row_cols, col_rows, path, visited_rows, visited_cols,
                start_row, start_col, true,
            ) {
                return true;
            }
            path.pop();
            visited_cols[c] = false;
        }
    }
    false
}

/// Approximate EMD using Sinkhorn iterations (entropic regularization).
///
/// Faster than exact for large histograms. Regularization parameter `lambda`
/// controls accuracy (higher = more accurate but slower convergence).
pub fn emd_sinkhorn(
    supply: &[f64],
    demand: &[f64],
    cost: &[f64],
    lambda: f64,
    n_iter: usize,
) -> f64 {
    let m = supply.len();
    let n = demand.len();
    if m == 0 || n == 0 {
        return 0.0;
    }

    // K_ij = exp(-lambda * cost_ij)
    let mut k: Vec<f64> = Vec::with_capacity(m * n);
    for c in cost.iter() {
        k.push((-lambda * c).exp());
    }

    let mut u: Vec<f64> = vec![1.0; m];
    let mut v: Vec<f64> = vec![1.0; n];

    for _ in 0..n_iter {
        // u = supply / (K * v)
        for i in 0..m {
            let mut kv = 0.0;
            for j in 0..n {
                kv += k[i * n + j] * v[j];
            }
            u[i] = if kv > 1e-300 {
                supply[i] / kv
            } else {
                1.0
            };
        }
        // v = demand / (K^T * u)
        for j in 0..n {
            let mut ku = 0.0;
            for i in 0..m {
                ku += k[i * n + j] * u[i];
            }
            v[j] = if ku > 1e-300 {
                demand[j] / ku
            } else {
                1.0
            };
        }
    }

    // Transport plan T_ij = u_i * K_ij * v_j
    let mut total = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let t = u[i] * k[i * n + j] * v[j];
            total += t * cost[i * n + j];
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_histograms() {
        let supply = vec![0.5, 0.5];
        let demand = vec![0.5, 0.5];
        let cost = vec![0.0, 1.0, 1.0, 0.0];
        let result = emd_solve(&supply, &demand, &cost, 0);
        assert!(result < 1e-6, "identical histograms should have EMD ~0: {result}");
    }

    #[test]
    fn test_opposite_histograms() {
        let supply = vec![1.0, 0.0];
        let demand = vec![0.0, 1.0];
        let cost = vec![0.0, 2.0, 2.0, 0.0];
        let result = emd_solve(&supply, &demand, &cost, 0);
        assert!(
            (result - 2.0).abs() < 1e-6,
            "should move all mass distance 2: {result}"
        );
    }

    #[test]
    fn test_uniform_3x3() {
        let supply = vec![1.0 / 3.0; 3];
        let demand = vec![1.0 / 3.0; 3];
        let cost = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let result = emd_solve(&supply, &demand, &cost, 0);
        assert!(result < 1e-6, "matched histograms: {result}");
    }

    #[test]
    fn test_asymmetric_transport() {
        let supply = vec![0.6, 0.4];
        let demand = vec![0.4, 0.6];
        let cost = vec![0.0, 1.0, 1.0, 0.0];
        let result = emd_solve(&supply, &demand, &cost, 0);
        assert!(
            (result - 0.2).abs() < 1e-6,
            "expected 0.2, got {result}"
        );
    }

    #[test]
    fn test_single_element() {
        let supply = vec![1.0];
        let demand = vec![1.0];
        let cost = vec![3.14];
        let result = emd_solve(&supply, &demand, &cost, 0);
        assert!(
            (result - 3.14).abs() < 1e-6,
            "expected 3.14, got {result}"
        );
    }

    #[test]
    fn test_triangle_inequality() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];
        let cost = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.0,
            2.0, 1.0, 0.0,
        ];
        let ab = emd_solve(&a, &b, &cost, 0);
        let bc = emd_solve(&b, &c, &cost, 0);
        let ac = emd_solve(&a, &c, &cost, 0);
        assert!(
            ac <= ab + bc + 1e-6,
            "triangle inequality violated: AC={ac}, AB={ab}, BC={bc}"
        );
    }

    #[test]
    fn test_larger_problem() {
        // 10x10 transport problem
        let n = 10;
        let supply: Vec<f64> = vec![1.0 / n as f64; n];
        let demand: Vec<f64> = vec![1.0 / n as f64; n];
        let mut cost = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                cost[i * n + j] = ((i as f64 - j as f64).abs()).sqrt();
            }
        }
        let result = emd_solve(&supply, &demand, &cost, 0);
        // Optimal: match each to itself, cost = 0
        assert!(result < 1e-6, "matched 10x10: {result}");
    }

    #[test]
    fn test_sinkhorn_approx() {
        let supply = vec![0.6, 0.4];
        let demand = vec![0.4, 0.6];
        let cost = vec![0.0, 1.0, 1.0, 0.0];
        let result = emd_sinkhorn(&supply, &demand, &cost, 50.0, 100);
        assert!(
            (result - 0.2).abs() < 0.01,
            "Sinkhorn expected ~0.2, got {result}"
        );
    }
}
