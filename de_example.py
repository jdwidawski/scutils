import scanpy as sc
import pandas as pd
import numpy as np
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

def make_pseudobulk_data(adata, groupby="sampleID", log=True, min_cells_in_group=50):
    sample_matrices = []
    out_groups = []
    for group in adata.obs[groupby].cat.categories:
        adata_sample = adata[adata.obs[groupby] == group]
        #print(group, adata_sample.n_obs)
        if min_cells_in_group is not None:
            if adata_sample.n_obs < min_cells_in_group:
                out_groups.append(group)
                continue
            
        sample_matrices.append(
            adata_sample.X.sum(axis=0)
        )


    pseudobulk_adata = sc.AnnData(
        X = np.vstack(sample_matrices),
        obs = pd.DataFrame(index=[x for x in adata.obs[groupby].cat.categories if x not in out_groups]),
        var = adata.var
    )
    
    if log==True:
        sc.pp.log1p(pseudobulk_adata)
        
    return pseudobulk_adata

def get_column_dict(adata, sampleid_col="sampleID", column=None):
    column_dict = dict(adata.obs[[sampleid_col, column]].values)
    return column_dict

def run_de(adata, ctype):
    # Create pseudobulk data object
    adata_ctype = adata[adata.obs["cell_type_04"] == ctype].copy()
    adata_pseudobulk = make_pseudobulk_data(adata_ctype,
                                            groupby="sampleID",
                                            log=False)

    adata_ctype.obs['group'] = adata_ctype.obs["Group"].copy()

    # Transfer relevant adata annotation
    adata_pseudobulk.obs["group"] = adata_pseudobulk.obs_names.map(
        get_column_dict(adata_ctype, column="group")).astype("category")
    adata_pseudobulk.obs["sampleID"] = adata_pseudobulk.obs_names.tolist()
    adata_pseudobulk.var["gene_name"] = adata_pseudobulk.var_names.tolist()

    # Generate variables to create DESeq object
    counts_df = pd.DataFrame(
        adata_pseudobulk.X.T,
        columns = adata_pseudobulk.obs["sampleID"].tolist(), index = adata_pseudobulk.var_names.tolist()).T
    metadata = adata_pseudobulk.obs
    vardata = adata_pseudobulk.var


    # Run DE
    dds = DeseqDataSet(
        counts=counts_df,
        clinical=metadata,
        design_factors="group",
        refit_cooks=True,
        n_cpus=8)

    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.refit()

    stat_res = DeseqStats(dds, alpha=0.05, cooks_filter=True, independent_filter=True,
                          contrast=("group", "positive_control ", "negative_control"))

    stat_res.run_wald_test()
    if stat_res.independent_filter:
        stat_res._independent_filtering()
    else:
        stat_res._p_value_adjustment()

    stat_res.summary()
    stat_res.lfc_shrink(coeff="group_positive_control _vs_negative_control")
    res = stat_res.results_df
    
    
    return res