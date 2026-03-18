# QZR-guided Sensitivity-Aware Mixed-Precision IO Resolution Allocation

## Normalized Vector Replay

For each backward vector $dy_v$ in module $m = (\text{layer}, \text{sublayer})$:

$$z_v = dy_v / \alpha_v$$

where $\alpha_v = \max_i |dy_{v,i}|$ (ABS_MAX scaling).

## Quantization Replay

For target bit $b$:

- $\tau_b = 1 / (2^b - 2)$ (zero threshold)

- $\Delta_b = 2 / (2^b - 2)$ (step size)

- $z_q^{(b)} = \text{clip}(\text{round}(z / \Delta_b) \cdot \Delta_b, -1, 1)$

## Primary Metrics

1. **QZR_nonzero(b)**: fraction of originally nonzero $z_i$ with $|z_i| < \tau_b$

2. **WQZR_body(b)**: body-weighted QZR with $w_i = \min(|z_i|/(\rho + \epsilon), 1)$, $\rho = P_{99}(|z|)$

3. **sign_agree_nonzero(b)**: fraction of nonzero elements preserving sign

4. **trimmed_rel_l2_{p90,p99,p999}(b)**: rel-L2 after removing top-{10%,1%,0.1%} outliers

   - p90: aggressive body (768-dim → ~691 elements kept)

   - p99: moderate body (768-dim → ~760 elements kept)

   - p999: conservative body (768-dim → ~767 elements kept, best for high-ODR like FFN1)

## Surrogate Risk

$$R_m(b) = q_m(b) + \epsilon_w \cdot w_m(b) + \epsilon_s \cdot (1 - s_m(b)) + \epsilon_t \cdot t_m(b)$$

Default: $\epsilon_w = 0.05$, $\epsilon_s = 0.02$, $\epsilon_t = 0.01$

## Marginal Gain

$$g_m(k) = R_m(b_{\min} + k - 1) - R_m(b_{\min} + k)$$

## Budget-Conditioned Optimization

$$\max \sum_{m,k} g_m(k) \cdot x_{m,k}$$

$$\text{s.t. } \sum_{m,k} x_{m,k} = K = |M| \cdot (B_{\text{avg}} - b_{\min})$$

$$x_{m,k} \in \{0,1\}, \quad x_{m,k} \leq x_{m,k-1}$$

Solved via greedy marginal gain. Optimal when gains are non-increasing

(concave risk curves), which is empirically verified. Otherwise near-optimal.

## Recovery

$$b_m = b_{\min} + \sum_k x_{m,k}$$
