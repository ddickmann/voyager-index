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
/// Production code uses `emd_sinkhorn` (faster, log-domain stable). This exact
/// solver is retained for testing and situations where true metric properties
/// are required.
///
/// - `supply`: weights for source nodes (must sum to ~1.0)
/// - `demand`: weights for sink nodes (must sum to ~1.0)
/// - `cost`:   n_supply x n_demand row-major cost matrix
/// - `max_iter`: iteration cap (0 = unlimited, recommended: 100_000)
///
/// Returns the optimal transport cost.
#[allow(dead_code)]
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
    if total_s < 1e-15 || total_d < 1e-15 {
        return 0.0;
    }
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

        let _ = iteration;
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

/// Approximate EMD using log-domain Sinkhorn iterations (entropic regularization).
///
/// Numerically stable: works in log-space to avoid exp overflow/underflow that
/// plagues the standard Sinkhorn for large `lambda * cost`.
///
/// Early stopping: checks marginal residual every 10 iterations and breaks
/// when converged within 1e-6 tolerance, typically saving 50-70% of iterations.
#[allow(clippy::needless_range_loop)]
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
    debug_assert_eq!(cost.len(), m * n, "cost matrix size must be m * n");

    let total_s: f64 = supply.iter().sum();
    let total_d: f64 = demand.iter().sum();
    if total_s < 1e-15 || total_d < 1e-15 {
        return 0.0;
    }

    let log_k: Vec<f64> = cost.iter().map(|&c| -lambda * c).collect();

    let mut log_u = vec![0.0f64; m];
    let mut log_v = vec![0.0f64; n];

    let log_supply: Vec<f64> = supply.iter().map(|&s| if s > 1e-300 { s.ln() } else { -700.0 }).collect();
    let log_demand: Vec<f64> = demand.iter().map(|&d| if d > 1e-300 { d.ln() } else { -700.0 }).collect();

    const CONVERGENCE_TOL: f64 = 1e-6;

    for iter in 0..n_iter {
        for i in 0..m {
            let row_base = i * n;
            let max_val = (0..n).map(|j| log_k[row_base + j] + log_v[j])
                .fold(f64::NEG_INFINITY, f64::max);
            if max_val == f64::NEG_INFINITY {
                log_u[i] = 0.0;
                continue;
            }
            let lse: f64 = (0..n)
                .map(|j| (log_k[row_base + j] + log_v[j] - max_val).exp())
                .sum::<f64>()
                .ln()
                + max_val;
            log_u[i] = log_supply[i] - lse;
        }

        for j in 0..n {
            let max_val = (0..m).map(|i| log_k[i * n + j] + log_u[i])
                .fold(f64::NEG_INFINITY, f64::max);
            if max_val == f64::NEG_INFINITY {
                log_v[j] = 0.0;
                continue;
            }
            let lse: f64 = (0..m)
                .map(|i| (log_k[i * n + j] + log_u[i] - max_val).exp())
                .sum::<f64>()
                .ln()
                + max_val;
            log_v[j] = log_demand[j] - lse;
        }

        if (iter + 1) % 10 == 0 {
            let mut max_residual = 0.0f64;
            for i in 0..m {
                let row_base = i * n;
                let max_val = (0..n).map(|j| log_u[i] + log_k[row_base + j] + log_v[j])
                    .fold(f64::NEG_INFINITY, f64::max);
                let row_sum = if max_val == f64::NEG_INFINITY {
                    0.0
                } else {
                    (0..n)
                        .map(|j| (log_u[i] + log_k[row_base + j] + log_v[j] - max_val).exp())
                        .sum::<f64>()
                        * max_val.exp()
                };
                let residual = (row_sum - supply[i]).abs();
                if residual > max_residual {
                    max_residual = residual;
                }
            }
            if max_residual < CONVERGENCE_TOL {
                break;
            }
        }
    }

    let mut total = 0.0f64;
    for i in 0..m {
        let row_base = i * n;
        for j in 0..n {
            let log_t = log_u[i] + log_k[row_base + j] + log_v[j];
            if log_t > -500.0 {
                total += log_t.exp() * cost[row_base + j];
            }
        }
    }

    if total.is_nan() || total.is_infinite() {
        log::warn!("emd_sinkhorn produced non-finite result ({total}), returning 0.0");
        return 0.0;
    }
    total
}

/// f32-only Sinkhorn for graph construction where f64 precision is unnecessary.
///
/// Halves memory bandwidth and enables wider SIMD vectorization (8 f32 vs 4 f64
/// per 256-bit register). The result is used for ranking only (cast to f32 anyway),
/// so the reduced precision is acceptable.
///
/// `check_every`: how often to check convergence (5 for fast mode, 10 for precise)
/// `tol`: convergence tolerance (1e-4 for fast, 1e-6 for precise)
#[allow(clippy::needless_range_loop)]
pub fn emd_sinkhorn_f32(
    supply: &[f32],
    demand: &[f32],
    cost: &[f32],
    lambda: f32,
    n_iter: usize,
    check_every: usize,
    tol: f32,
) -> f32 {
    let m = supply.len();
    let n = demand.len();
    if m == 0 || n == 0 {
        return 0.0;
    }
    debug_assert_eq!(cost.len(), m * n, "cost matrix size must be m * n");

    let total_s: f32 = supply.iter().sum();
    let total_d: f32 = demand.iter().sum();
    if total_s < 1e-10 || total_d < 1e-10 {
        return 0.0;
    }

    let log_k: Vec<f32> = cost.iter().map(|&c| (-lambda * c).max(-80.0)).collect();

    let mut log_u = vec![0.0f32; m];
    let mut log_v = vec![0.0f32; n];

    let log_supply: Vec<f32> = supply.iter().map(|&s| if s > 1e-30 { s.ln() } else { -70.0 }).collect();
    let log_demand: Vec<f32> = demand.iter().map(|&d| if d > 1e-30 { d.ln() } else { -70.0 }).collect();

    let check_interval = check_every.max(1);

    for iter in 0..n_iter {
        for i in 0..m {
            let row_base = i * n;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..n {
                let v = log_k[row_base + j] + log_v[j];
                if v > max_val { max_val = v; }
            }
            if max_val == f32::NEG_INFINITY {
                log_u[i] = 0.0;
                continue;
            }
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += (log_k[row_base + j] + log_v[j] - max_val).exp();
            }
            log_u[i] = log_supply[i] - acc.ln() - max_val;
        }

        for j in 0..n {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..m {
                let v = log_k[i * n + j] + log_u[i];
                if v > max_val { max_val = v; }
            }
            if max_val == f32::NEG_INFINITY {
                log_v[j] = 0.0;
                continue;
            }
            let mut acc = 0.0f32;
            for i in 0..m {
                acc += (log_k[i * n + j] + log_u[i] - max_val).exp();
            }
            log_v[j] = log_demand[j] - acc.ln() - max_val;
        }

        if (iter + 1) % check_interval == 0 {
            let mut max_residual = 0.0f32;
            for i in 0..m {
                let row_base = i * n;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..n {
                    let v = log_u[i] + log_k[row_base + j] + log_v[j];
                    if v > max_val { max_val = v; }
                }
                let row_sum = if max_val == f32::NEG_INFINITY {
                    0.0
                } else {
                    let mut s = 0.0f32;
                    for j in 0..n {
                        s += (log_u[i] + log_k[row_base + j] + log_v[j] - max_val).exp();
                    }
                    s * max_val.exp()
                };
                let residual = (row_sum - supply[i]).abs();
                if residual > max_residual {
                    max_residual = residual;
                }
            }
            if max_residual < tol {
                break;
            }
        }
    }

    let mut total = 0.0f32;
    for i in 0..m {
        let row_base = i * n;
        for j in 0..n {
            let log_t = log_u[i] + log_k[row_base + j] + log_v[j];
            if log_t > -50.0 {
                total += log_t.exp() * cost[row_base + j];
            }
        }
    }

    if total.is_nan() || total.is_infinite() {
        log::warn!("emd_sinkhorn_f32 produced non-finite result ({total}), returning 0.0");
        return 0.0;
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
        let cost = vec![std::f64::consts::PI];
        let result = emd_solve(&supply, &demand, &cost, 0);
        assert!(
            (result - std::f64::consts::PI).abs() < 1e-6,
            "expected PI, got {result}"
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

    /// Same as `emd_sinkhorn`, but returns the number of iterations completed (early exit or cap).
    fn emd_sinkhorn_last_iter(
        supply: &[f64],
        demand: &[f64],
        cost: &[f64],
        lambda: f64,
        n_iter: usize,
    ) -> (f64, usize) {
        let m = supply.len();
        let n = demand.len();
        if m == 0 || n == 0 {
            return (0.0, 0);
        }
        let total_s: f64 = supply.iter().sum();
        let total_d: f64 = demand.iter().sum();
        if total_s < 1e-15 || total_d < 1e-15 {
            return (0.0, 0);
        }
        let log_k: Vec<f64> = cost.iter().map(|&c| -lambda * c).collect();
        let mut log_u = vec![0.0f64; m];
        let mut log_v = vec![0.0f64; n];
        let log_supply: Vec<f64> = supply
            .iter()
            .map(|&s| if s > 1e-300 { s.ln() } else { -700.0 })
            .collect();
        let log_demand: Vec<f64> = demand
            .iter()
            .map(|&d| if d > 1e-300 { d.ln() } else { -700.0 })
            .collect();
        const CONVERGENCE_TOL: f64 = 1e-6;
        let mut completed = n_iter;
        for iter in 0..n_iter {
            for i in 0..m {
                let row_base = i * n;
                let max_val = (0..n)
                    .map(|j| log_k[row_base + j] + log_v[j])
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_val == f64::NEG_INFINITY {
                    log_u[i] = 0.0;
                    continue;
                }
                let lse: f64 = (0..n)
                    .map(|j| (log_k[row_base + j] + log_v[j] - max_val).exp())
                    .sum::<f64>()
                    .ln()
                    + max_val;
                log_u[i] = log_supply[i] - lse;
            }
            for j in 0..n {
                let max_val = (0..m)
                    .map(|i| log_k[i * n + j] + log_u[i])
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_val == f64::NEG_INFINITY {
                    log_v[j] = 0.0;
                    continue;
                }
                let lse: f64 = (0..m)
                    .map(|i| (log_k[i * n + j] + log_u[i] - max_val).exp())
                    .sum::<f64>()
                    .ln()
                    + max_val;
                log_v[j] = log_demand[j] - lse;
            }
            if (iter + 1) % 10 == 0 {
                let mut max_residual = 0.0f64;
                for i in 0..m {
                    let row_base = i * n;
                    let max_val = (0..n)
                        .map(|j| log_u[i] + log_k[row_base + j] + log_v[j])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let row_sum = if max_val == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (0..n)
                            .map(|j| {
                                (log_u[i] + log_k[row_base + j] + log_v[j] - max_val).exp()
                            })
                            .sum::<f64>()
                            * max_val.exp()
                    };
                    let residual = (row_sum - supply[i]).abs();
                    if residual > max_residual {
                        max_residual = residual;
                    }
                }
                if max_residual < CONVERGENCE_TOL {
                    completed = iter + 1;
                    break;
                }
            }
        }
        let mut total = 0.0f64;
        for (i, &lu) in log_u.iter().enumerate().take(m) {
            let row_base = i * n;
            for j in 0..n {
                let log_t = lu + log_k[row_base + j] + log_v[j];
                if log_t > -500.0 {
                    total += log_t.exp() * cost[row_base + j];
                }
            }
        }
        if total.is_nan() || total.is_infinite() {
            return (0.0, completed);
        }
        (total, completed)
    }

    #[test]
    fn test_sinkhorn_3x3_known() {
        let supply = vec![0.5, 0.3, 0.2];
        let demand = vec![0.2, 0.5, 0.3];
        let cost = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let result = emd_sinkhorn(&supply, &demand, &cost, 50.0, 500);
        assert!(result > 0.0, "expected positive cost, got {result}");
        assert!(result.is_finite(), "expected finite cost, got {result}");
    }

    #[test]
    fn test_sinkhorn_early_stop() {
        let supply = vec![0.6, 0.4];
        let demand = vec![0.4, 0.6];
        let cost = vec![0.0, 1.0, 1.0, 0.0];
        let exact = emd_solve(&supply, &demand, &cost, 0);
        let (approx, iters) = emd_sinkhorn_last_iter(&supply, &demand, &cost, 50.0, 1000);
        assert!(
            iters < 1000,
            "early stop should finish before max_iter, got {iters}"
        );
        assert!(
            (approx - exact).abs() < 0.05,
            "Sinkhorn vs exact: exact={exact}, approx={approx}"
        );
    }

    #[test]
    fn test_sinkhorn_numerical_stability() {
        let supply = vec![0.25, 0.25, 0.25, 0.25];
        let demand = vec![0.25, 0.25, 0.25, 0.25];
        let cost = vec![
            0.0, 2.5, 5.0, 10.0,
            3.0, 0.0, 7.5, 4.0,
            10.0, 1.0, 0.0, 6.0,
            8.0, 9.0, 2.0, 0.0,
        ];
        let result = emd_sinkhorn(&supply, &demand, &cost, 200.0, 500);
        assert!(!result.is_nan(), "unexpected NaN");
        assert!(!result.is_infinite(), "unexpected Inf");
    }

    #[test]
    fn test_sinkhorn_zero_supply() {
        let cost_2x2 = vec![0.0, 1.0, 1.0, 0.0];
        assert_eq!(emd_sinkhorn(&[], &[], &[], 20.0, 100), 0.0);
        assert_eq!(emd_solve(&[], &[], &[], 0), 0.0);
        assert_eq!(
            emd_sinkhorn(&[0.0, 0.0], &[0.5, 0.5], &cost_2x2, 20.0, 100),
            0.0
        );
    }

    #[test]
    fn test_sinkhorn_large_64x64() {
        let n = 64;
        let w = 1.0 / n as f64;
        let supply: Vec<f64> = vec![w; n];
        let demand: Vec<f64> = vec![w; n];
        let mut cost = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                cost[i * n + j] = ((i * 7 + j * 13) % 100) as f64 / 100.0;
            }
        }
        let result = emd_sinkhorn(&supply, &demand, &cost, 30.0, 200);
        assert!(result >= 0.0, "expected non-negative, got {result}");
        assert!(result.is_finite(), "expected finite, got {result}");
    }

    #[test]
    fn test_sinkhorn_f32_approx() {
        let supply = vec![0.6f32, 0.4];
        let demand = vec![0.4f32, 0.6];
        let cost = vec![0.0f32, 1.0, 1.0, 0.0];
        let result = emd_sinkhorn_f32(&supply, &demand, &cost, 50.0, 100, 10, 1e-6);
        assert!(
            (result - 0.2).abs() < 0.02,
            "f32 Sinkhorn expected ~0.2, got {result}"
        );
    }

    #[test]
    fn test_sinkhorn_f32_fast_mode() {
        let supply = vec![0.6f32, 0.4];
        let demand = vec![0.4f32, 0.6];
        let cost = vec![0.0f32, 1.0, 1.0, 0.0];
        let result = emd_sinkhorn_f32(&supply, &demand, &cost, 20.0, 30, 5, 1e-4);
        assert!(
            (result - 0.2).abs() < 0.05,
            "f32 fast Sinkhorn expected ~0.2, got {result}"
        );
    }

    #[test]
    fn test_sinkhorn_f32_agrees_with_f64() {
        let supply_f64 = vec![0.5, 0.3, 0.2];
        let demand_f64 = vec![0.2, 0.5, 0.3];
        let cost_f64 = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let supply_f32: Vec<f32> = supply_f64.iter().map(|&v| v as f32).collect();
        let demand_f32: Vec<f32> = demand_f64.iter().map(|&v| v as f32).collect();
        let cost_f32: Vec<f32> = cost_f64.iter().map(|&v| v as f32).collect();

        let f64_result = emd_sinkhorn(&supply_f64, &demand_f64, &cost_f64, 50.0, 500);
        let f32_result = emd_sinkhorn_f32(&supply_f32, &demand_f32, &cost_f32, 50.0, 500, 10, 1e-6);
        let diff = (f64_result as f32 - f32_result).abs();
        assert!(
            diff < 0.05,
            "f32 and f64 Sinkhorn should agree within 0.05: f64={f64_result}, f32={f32_result}, diff={diff}"
        );
    }

    #[test]
    fn test_sinkhorn_f32_numerical_stability() {
        let supply = vec![0.25f32; 4];
        let demand = vec![0.25f32; 4];
        let cost: Vec<f32> = vec![
            0.0, 2.5, 5.0, 10.0,
            3.0, 0.0, 7.5, 4.0,
            10.0, 1.0, 0.0, 6.0,
            8.0, 9.0, 2.0, 0.0,
        ];
        let result = emd_sinkhorn_f32(&supply, &demand, &cost, 200.0, 500, 10, 1e-6);
        assert!(!result.is_nan(), "unexpected NaN for f32");
        assert!(!result.is_infinite(), "unexpected Inf for f32");
    }
}
