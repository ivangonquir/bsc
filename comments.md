# Conclusions
## Attention features alone can beat embedding-based methods
- `head_entropy` in LUNAR beats the paper's baseline. 
- For SO-GAAL, attention-only combos reach 0.85, beating every method of the paper. **This is a non-trivial finding**: 12-48 dim feature vectors derived from attention beat 768-3072 dim embeddings.

## The right featurization depends strongly on the detector
- Density-based (LOF, LUNAR): `head_entropy`
- Tree-based (IForest): `cls_token_pca`
- Reconstruction-based (AE, VAE)_ `cls_token_` (`raw` or `pca`) is the foundation; attention adds modestly.
- Adversarial (SO-GAAL): `cls_mean` + dispersion stats; `cls_token` actively hurts.
- CDF-based (ECOD): `cls_token_pca + cls_std`

This reflects what each detector's algorithm can extract from feature geometry.


Now to see if each feature is significant, we need to set up a **factorial design**:
- **Factors**: 5 binary attention features $\times$ 1 cls_token $\times$ 8 detectors = 40 factors
- **Response variable**: AUROC (and AUPRC)
- **Design**: Full factorial over the 5 attention features (all $2^5 = 32$ combinations), partial over the cls_token factor (only PCA paired, not raw)
- **Replications**: 3 seeds for stochastic detectors, 1 for deterministic.
- **Blocking variable**: detector (analyze within each detector separately)


## Factorial study
- Feature usefulness is statistically confirmed to be dataset-dependent.
- *A factorial analysis across 63 feature combinations × 8 detectors × 3 datasets shows that feature contributions are highly conditional. Only the BERT embedding (cls_token) has a consistently positive main effect (significant in 22/24 detector–dataset cells, mean +0.13 AUROC). Among attention-derived features, head_entropy is the strongest but context-dependent (significant positive in 13 cells, negative in 5). The remaining attention statistics have main effects indistinguishable from zero. A feature × dataset interaction test rejects effect homogeneity for all 8 detectors (p < 0.001), confirming that attention-feature usefulness depends on dataset type. Notably, cls_token and head_entropy interact sub-additively (significant interference in 12/24 cells): head_entropy carries anomaly signal that is largely redundant with the embedding, making it most useful as a low-dimensional replacement (12 dims vs. 768) rather than an addition.*