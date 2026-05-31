"""
Multivariate Statistical Analysis Module for AMFOrA Ceramic Analysis.

This module provides comprehensive statistical analysis capabilities for ceramic
fabric analysis, including Principal Component Analysis (PCA), clustering,
correlation analysis, and specialized archaeological interpretations.

Features:
- Data preprocessing and normalization for ceramic measurements
- Principal Component Analysis with archaeological interpretation
- Hierarchical and k-means clustering
- Correlation analysis and feature selection
- Statistical visualization functions
- Manufacturing technique identification
- Assemblage comparison tools

Author: Enhanced AMFOrA Statistical Suite
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import warnings
from typing import Dict, List, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    'CeramicStatisticalAnalyzer', 'CeramicVisualization',
    'quick_ceramic_analysis', 'compare_assemblages',
]


class CeramicStatisticalAnalyzer:
    """
    Comprehensive statistical analysis class for ceramic fabric data.

    This class provides all the statistical methods used in the streamlit application
    in a standalone, reusable format suitable for batch analysis and research.

    Typical workflow::

        analyzer = CeramicStatisticalAnalyzer()
        analyzer.prepare_data(df, scaling_method='standard')

        pca_results = analyzer.perform_pca(variance_threshold=0.90)
        cluster_results = analyzer.perform_clustering(method='hierarchical')
        corr_results = analyzer.correlation_analysis(method='pearson')

        # Or run everything at once:
        report = analyzer.generate_report()

    Use `CeramicVisualization` to plot results (biplots, dendrograms, etc.).
    """
    
    def __init__(self):
        """Initialize the statistical analyzer with default parameters."""
        self.scaler = None
        self.pca_model = None
        self.scaled_data = None
        self.feature_names = None
        self.sample_names = None
        
    def prepare_data(self, data: Union[pd.DataFrame, Dict],
                    exclude_columns: List[str] = None,
                    include_only: List[str] = None,
                    scaling_method: str = 'standard') -> pd.DataFrame:
        """
        Prepare and preprocess ceramic analysis data for statistical analysis.

        Must be called before any analysis method (``perform_pca``,
        ``perform_clustering``, ``correlation_analysis``, ``generate_report``).

        Non-numeric columns and constant columns (zero variance) are
        automatically removed. If the DataFrame contains a ``filename``
        column, those values are stored as sample labels and used in PCA
        score tables and dendrogram leaf labels.

        Parameters
        ----------
        data : pd.DataFrame or dict
            Raw ceramic analysis data. If a dict, it is converted to a
            DataFrame first. Each row is one sherd/sample; columns are
            numeric measurement features (e.g. from ``full_analysis()``).
        exclude_columns : list of str, optional
            Column names to drop before analysis. Useful for removing
            metadata columns that survived automatic filtering, e.g.
            ``['notes', 'context']``.
        include_only : list of str, optional
            If provided, **only** these columns are kept (after numeric
            filtering). Takes precedence — ``exclude_columns`` is applied
            first, then ``include_only`` filters the remainder.
        scaling_method : str, default 'standard'
            How to scale features before multivariate analysis:

            - ``'standard'`` — zero-mean, unit-variance (scikit-learn
              ``StandardScaler``). Best general-purpose choice.
            - ``'robust'`` — median-centered, IQR-scaled (``RobustScaler``).
              Use when data contains outliers.
            - ``'none'`` — no scaling. Only appropriate when all features
              share the same units and comparable ranges.

        Returns
        -------
        pd.DataFrame
            Preprocessed, scaled data stored internally (also accessible
            as ``self.scaled_data``).

        Notes
        -----
        **Feature selection tips** — AMFOrA ``full_analysis()`` can
        produce ~80+ columns per sherd, many of which are redundant
        (e.g. ``blob_mean_diameter_mm`` and ``blob_mean_area_mm2``
        measure essentially the same thing). Trimming redundant features
        improves clustering stability and is **required** for distance
        metrics like Mahalanobis (which need more samples than features
        for a well-conditioned covariance matrix).

        Strategies for reducing feature count:

        1. **Drop one from each highly-correlated pair.** Run
           ``correlation_analysis()`` first and inspect
           ``significant_correlations`` — pairs with |r| > 0.9 are
           near-duplicates. Keep whichever is more interpretable.
        2. **Pick one detection method.** Blob and contour columns
           (``blob_*`` vs ``contour_*``) measure the same properties
           with different algorithms. Use ``include_only`` with one
           prefix, or ``exclude_columns`` to drop the other.
        3. **Separate concerns.** Color features (``*_color_l/a/b``),
           size features (``*_diameter_*``, ``*_area_*``), and
           orientation features (``*_orientation_*``) capture different
           aspects of the fabric. For a focused analysis, include only
           the relevant group.
        4. **Exclude summary statistics that overlap.** Mean and median
           size, or count and density, often carry the same information.
           Keep one representative per concept.
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Handle missing values
        df = df.dropna()
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply column selection filters
        if include_only:
            numeric_cols = [col for col in numeric_cols if col in include_only]
        
        if exclude_columns:
            numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
            
        # Store sample information
        if 'filename' in df.columns:
            self.sample_names = df['filename'].tolist()
        elif 'Name' in df.columns:
            self.sample_names = df['Name'].tolist()
        else:
            self.sample_names = [f"Sample_{i+1}" for i in range(len(df))]
            
        # Extract numeric data
        analysis_df = df[numeric_cols].copy()
        
        # Remove constant columns (no variation)
        analysis_df = analysis_df.loc[:, analysis_df.std() > 1e-10]
        
        # Store feature names
        self.feature_names = analysis_df.columns.tolist()
        
        # Apply scaling
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError("scaling_method must be 'standard', 'robust', or 'none'")
            
        if self.scaler:
            self.scaled_data = pd.DataFrame(
                self.scaler.fit_transform(analysis_df),
                columns=analysis_df.columns,
                index=analysis_df.index
            )
        else:
            self.scaled_data = analysis_df
            
        return self.scaled_data
    
    def perform_pca(self, n_components: Optional[int] = None,
                   variance_threshold: float = 0.95) -> Dict:
        """
        Perform Principal Component Analysis on ceramic data.

        Requires ``prepare_data()`` to have been called first. When
        ``n_components`` is not set, the number of components is chosen
        automatically as the fewest that cumulatively explain at least
        ``variance_threshold`` of the total variance (minimum 2).

        Parameters
        ----------
        n_components : int, optional
            Exact number of principal components to retain. If ``None``
            (default), determined automatically from *variance_threshold*.
        variance_threshold : float, default 0.95
            Cumulative explained-variance ratio at which to stop adding
            components. Only used when *n_components* is ``None``.
            E.g. ``0.90`` keeps enough components to explain 90 % of
            variance — useful for reducing dimensionality while retaining
            most information.

        Returns
        -------
        dict
            PCA results with keys:

            - ``'scores'`` — DataFrame of sample scores (PC1, PC2, …)
              with a ``Sample`` column (from filenames).
            - ``'loadings'`` — DataFrame of feature loadings per component.
            - ``'explained_variance'`` — array of per-component variance
              ratios.
            - ``'cumulative_variance'`` — cumulative sum of the above.
            - ``'interpretations'`` — auto-generated archaeological
              interpretation of each component based on top loadings.
            - ``'model'`` — fitted ``sklearn.decomposition.PCA`` object.
            - ``'n_components'`` — number of components retained.
        """
        if self.scaled_data is None:
            raise ValueError("Data must be prepared first using prepare_data()")
            
        # Determine number of components
        if n_components is None:
            # Fit PCA with all components to determine optimal number
            pca_full = PCA()
            pca_full.fit(self.scaled_data)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumvar >= variance_threshold) + 1
            n_components = max(2, min(n_components, len(self.feature_names)))
            
        # Fit final PCA model
        self.pca_model = PCA(n_components=n_components)
        pca_scores = self.pca_model.fit_transform(self.scaled_data)
        
        # Create scores DataFrame
        pc_names = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_scores, columns=pc_names)
        pca_df['Sample'] = self.sample_names
        
        # Calculate loadings
        loadings = self.pca_model.components_.T * np.sqrt(self.pca_model.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings,
            columns=pc_names,
            index=self.feature_names
        )
        
        # Interpret components
        interpretations = self._interpret_pca_components(loadings_df)
        
        return {
            'scores': pca_df,
            'loadings': loadings_df,
            'explained_variance': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca_model.explained_variance_ratio_),
            'interpretations': interpretations,
            'model': self.pca_model,
            'n_components': n_components
        }
    
    def _interpret_pca_components(self, loadings_df: pd.DataFrame) -> Dict:
        """
        Provide archaeological interpretation of PCA components based on loadings.
        
        Parameters
        ----------
        loadings_df : pd.DataFrame
            PCA loadings matrix
            
        Returns
        -------
        dict
            Interpretations for each component
        """
        interpretations = {}
        
        for pc in loadings_df.columns:
            loadings = loadings_df[pc].abs().sort_values(ascending=False)
            top_features = loadings.head(5)
            
            # Analyze feature patterns for archaeological interpretation
            size_features = [f for f in top_features.index if any(x in f.lower() for x in ['size', 'area', 'diameter', 'length'])]
            color_features = [f for f in top_features.index if any(x in f.lower() for x in ['color', 'rgb', 'lab', 'hue'])]
            orientation_features = [f for f in top_features.index if any(x in f.lower() for x in ['orientation', 'angle', 'alignment'])]
            geometric_features = [f for f in top_features.index if any(x in f.lower() for x in ['geometric', 'roundness', 'angularity'])]
            
            # Generate interpretation
            interpretation = []
            if size_features:
                interpretation.append(f"Size characteristics ({', '.join(size_features[:2])})")
            if color_features:
                interpretation.append(f"Color properties ({', '.join(color_features[:2])})")
            if orientation_features:
                interpretation.append(f"Fabric alignment ({', '.join(orientation_features[:2])})")
            if geometric_features:
                interpretation.append(f"Inclusion geometry ({', '.join(geometric_features[:2])})")
                
            if not interpretation:
                interpretation = [f"Mixed characteristics (top: {top_features.index[0]})"]
                
            interpretations[pc] = {
                'description': ' + '.join(interpretation),
                'top_features': top_features.to_dict(),
                'archaeological_meaning': self._get_archaeological_meaning(pc, top_features)
            }
            
        return interpretations
    
    def _get_archaeological_meaning(self, component: str, top_features: pd.Series) -> str:
        """Generate archaeological interpretation for a PCA component."""
        feature_names = [f.lower() for f in top_features.index[:3]]
        
        if any('size' in f for f in feature_names):
            if any('orientation' in f for f in feature_names):
                return "Manufacturing technique axis: relates to how inclusions were aligned during forming"
            elif any('color' in f for f in feature_names):
                return "Temper composition axis: relates to the size and type of added materials"
            else:
                return "Inclusion scale axis: relates to the overall size distribution of temper"
                
        elif any('color' in f for f in feature_names):
            if any('geometric' in f for f in feature_names):
                return "Clay preparation axis: relates to the composition and processing of raw materials"
            else:
                return "Raw material axis: relates to the source and type of clay and temper"
                
        elif any('orientation' in f for f in feature_names):
            return "Fabric structure axis: relates to manufacturing technique and clay working methods"
            
        else:
            return "Mixed technological axis: combines multiple aspects of ceramic technology"
    
    def perform_clustering(self, method: str = 'hierarchical',
                          n_clusters: Optional[int] = None,
                          linkage_method: str = 'ward',
                          distance_metric: str = 'euclidean') -> Dict:
        """
        Perform clustering analysis on ceramic data.

        Requires ``prepare_data()`` to have been called first. When
        *n_clusters* is ``None``, the optimal count is estimated
        automatically (elbow method for hierarchical, silhouette scan
        for k-means).

        Parameters
        ----------
        method : str, default 'hierarchical'
            Clustering algorithm to use:

            - ``'hierarchical'`` — agglomerative hierarchical clustering.
              Produces a linkage matrix suitable for dendrogram plotting
              via ``CeramicVisualization.plot_dendrogram()``.
            - ``'kmeans'`` — K-means partitioning. Good when the number
              of groups is known or roughly estimated. Does not produce
              a dendrogram.
            - ``'dbscan'`` — density-based clustering. Does not require
              *n_clusters*; discovers clusters of arbitrary shape and
              labels outliers as noise (cluster label ``-1``). Parameters
              *eps* and *min_samples* are chosen automatically.

        n_clusters : int, optional
            Number of clusters for hierarchical and k-means. Ignored by
            DBSCAN. If ``None`` (default), determined automatically.
        linkage_method : str, default 'ward'
            Linkage criterion for hierarchical clustering (ignored by
            k-means/DBSCAN). Passed to ``scipy.cluster.hierarchy.linkage()``:

            - ``'ward'`` — minimizes within-cluster variance (requires
              Euclidean distance). Generally best for balanced clusters.
            - ``'complete'`` — maximum inter-cluster distance. Tends to
              produce compact, equally-sized clusters.
            - ``'average'`` — mean inter-cluster distance (UPGMA).
            - ``'single'`` — minimum inter-cluster distance. Susceptible
              to chaining; useful for detecting elongated clusters.
            - ``'centroid'``, ``'median'``, ``'weighted'`` — less common
              alternatives (see scipy docs).

        distance_metric : str, default 'euclidean'
            Distance metric for hierarchical clustering (ignored by
            k-means/DBSCAN). Any metric accepted by
            ``scipy.spatial.distance.pdist()`` is valid, including:

            - ``'euclidean'`` — standard L2 distance. Required when
              *linkage_method* is ``'ward'``.
            - ``'mahalanobis'`` — accounts for feature correlations and
              unequal variances. The inverse covariance matrix is
              computed automatically from the data. Well-suited for
              compositional / provenance studies. Falls back to
              pseudo-inverse with a warning if the covariance matrix
              is singular (e.g. when n < number of features).
              Incompatible with ``'ward'`` linkage.
            - ``'correlation'`` — ``1 - Pearson r``. Useful when the
              *shape* of the feature profile matters more than magnitude.
            - ``'cosine'``, ``'cityblock'`` (Manhattan),
              ``'chebyshev'``, etc.

        Returns
        -------
        dict
            Clustering results. Keys common to all methods:

            - ``'method'`` — the method string used.
            - ``'cluster_labels'`` — integer array of cluster assignments.
            - ``'n_clusters'`` — number of clusters found.
            - ``'silhouette_score'`` — mean silhouette coefficient (−1 to 1;
              higher is better). ``-1`` if only one cluster.
            - ``'calinski_harabasz_score'`` — Calinski-Harabasz index.
            - ``'cophenetic_correlation'`` — cophenetic correlation coefficient
              (0 to 1; higher means the dendrogram faithfully represents
              pairwise distances). Only set for hierarchical clustering;
              ``None`` for k-means and DBSCAN.
            - ``'cluster_summary'`` — per-cluster size and feature means.
            - ``'sample_names'`` — list of sample labels (from filenames).

            Additional keys by method:

            - *hierarchical*: ``'linkage_matrix'``, ``'linkage_method'``,
              ``'distance_metric'``.
            - *kmeans*: ``'cluster_centers'``, ``'inertia'``.
            - *dbscan*: ``'eps'``, ``'min_samples'``, ``'n_noise'``.
        """
        if self.scaled_data is None:
            raise ValueError("Data must be prepared first using prepare_data()")
            
        results = {}
        
        if method == 'hierarchical':
            # Perform hierarchical clustering
            if distance_metric == 'mahalanobis':
                cov = np.cov(self.scaled_data.values, rowvar=False)
                try:
                    VI = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    # Singular covariance (n < features or collinear columns)
                    # — fall back to pseudo-inverse
                    VI = np.linalg.pinv(cov)
                    warnings.warn(
                        "Covariance matrix is singular; using pseudo-inverse "
                        "for Mahalanobis distance. Consider reducing "
                        "dimensionality with PCA first."
                    )
                distances = pdist(self.scaled_data, metric='mahalanobis', VI=VI)
            elif distance_metric == 'correlation':
                distances = pdist(self.scaled_data, metric='correlation')
            else:
                distances = pdist(self.scaled_data, metric=distance_metric)
                
            linkage_matrix = linkage(distances, method=linkage_method)
            
            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                n_clusters = self._optimal_clusters_hierarchical(linkage_matrix)
                
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            results.update({
                'method': 'hierarchical',
                'linkage_matrix': linkage_matrix,
                'cluster_labels': cluster_labels,
                'n_clusters': n_clusters,
                'linkage_method': linkage_method,
                'distance_metric': distance_metric
            })
            
        elif method == 'kmeans':
            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                n_clusters = self._optimal_clusters_kmeans()
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            
            results.update({
                'method': 'kmeans',
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'n_clusters': n_clusters,
                'inertia': kmeans.inertia_
            })
            
        elif method == 'dbscan':
            # DBSCAN with automatic parameter selection
            eps, min_samples = self._optimal_dbscan_params()
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(self.scaled_data)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            results.update({
                'method': 'dbscan',
                'cluster_labels': cluster_labels,
                'n_clusters': n_clusters,
                'eps': eps,
                'min_samples': min_samples,
                'n_noise': list(cluster_labels).count(-1)
            })
            
        # Calculate clustering metrics
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(self.scaled_data, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_data, cluster_labels)
        else:
            silhouette = -1
            calinski_harabasz = 0
            
        # Cophenetic correlation (hierarchical only)
        if method == 'hierarchical' and len(set(cluster_labels)) > 1:
            coph_corr, _ = cophenet(linkage_matrix, distances)
        else:
            coph_corr = None

        results.update({
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'cophenetic_correlation': coph_corr,
            'cluster_summary': self._summarize_clusters(cluster_labels),
            'sample_names': self.sample_names
        })
        
        return results
    
    def _optimal_clusters_hierarchical(self, linkage_matrix: np.ndarray) -> int:
        """Determine optimal number of clusters for hierarchical clustering."""
        # Use elbow method on linkage distances
        distances = linkage_matrix[:, 2]
        if len(distances) < 3:
            return 2
            
        # Find largest jump in distances (elbow)
        diffs = np.diff(distances)
        if len(diffs) > 0:
            optimal_idx = np.argmax(diffs)
            optimal_clusters = len(distances) - optimal_idx
            return min(max(optimal_clusters, 2), 8)  # Reasonable range
        else:
            return 2
    
    def _optimal_clusters_kmeans(self) -> int:
        """Determine optimal number of clusters for k-means using elbow method."""
        max_k = min(10, len(self.scaled_data) - 1)
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
            
        # Find elbow using rate of change
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because we started from k=2
            return min(elbow_idx, max_k)
        else:
            return 3
    
    def _optimal_dbscan_params(self) -> Tuple[float, int]:
        """Determine optimal parameters for DBSCAN clustering."""
        from sklearn.neighbors import NearestNeighbors
        
        # Use k-distance method for eps
        k = 4  # Rule of thumb: 2 * dimensions
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.scaled_data)
        distances, indices = nbrs.kneighbors(self.scaled_data)
        
        # Sort distances and find knee point
        k_distances = np.sort(distances[:, k-1])
        
        # Simple knee detection
        if len(k_distances) > 10:
            diffs = np.diff(k_distances)
            knee_idx = np.argmax(diffs)
            eps = k_distances[knee_idx]
        else:
            eps = np.mean(k_distances)
            
        min_samples = max(3, int(len(self.scaled_data) * 0.05))  # 5% of data points
        
        return eps, min_samples
    
    def _summarize_clusters(self, cluster_labels: np.ndarray) -> Dict:
        """Create summary statistics for each cluster."""
        summary = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise cluster in DBSCAN
                cluster_name = 'Noise'
            else:
                cluster_name = f'Cluster_{label + 1}'
                
            mask = cluster_labels == label
            cluster_data = self.scaled_data.iloc[mask]
            
            summary[cluster_name] = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(cluster_labels) * 100),
                'samples': [self.sample_names[i] for i in np.where(mask)[0]],
                'mean_values': cluster_data.mean().to_dict(),
                'std_values': cluster_data.std().to_dict()
            }
            
        return summary
    
    def correlation_analysis(self, method: str = 'pearson',
                           significance_level: float = 0.05) -> Dict:
        """
        Perform pairwise correlation analysis on ceramic features.

        Computes the full correlation matrix, tests each pair for
        statistical significance, and returns significant pairs sorted
        by absolute strength.

        Parameters
        ----------
        method : str, default 'pearson'
            Correlation coefficient to compute:

            - ``'pearson'`` — linear correlation (parametric). Assumes
              approximately normal distributions.
            - ``'spearman'`` — rank correlation (non-parametric). Robust
              to non-linearity and outliers; recommended when data are
              ordinal or heavily skewed.
            - ``'kendall'`` — Kendall's tau (non-parametric). More robust
              than Spearman for small sample sizes but slower to compute.

        significance_level : float, default 0.05
            Alpha threshold for identifying significant correlations.
            Pairs with *p* < *significance_level* are included in the
            ``significant_correlations`` list.

        Returns
        -------
        dict
            - ``'correlation_matrix'`` — DataFrame (features x features).
            - ``'p_values'`` — DataFrame of p-values for each pair.
            - ``'significant_correlations'`` — list of dicts sorted by
              ``|correlation|``, each with ``feature1``, ``feature2``,
              ``correlation``, ``p_value``, and ``strength``
              (Strong/Moderate/Weak/Very weak).
            - ``'method'``, ``'significance_level'`` — echo of inputs.
        """
        if self.scaled_data is None:
            raise ValueError("Data must be prepared first using prepare_data()")
            
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = self.scaled_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = self.scaled_data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = self.scaled_data.corr(method='kendall')
        else:
            raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
            
        # Calculate p-values
        n = len(self.scaled_data)
        p_values = np.zeros((len(self.feature_names), len(self.feature_names)))
        
        for i, feature1 in enumerate(self.feature_names):
            for j, feature2 in enumerate(self.feature_names):
                if i != j:
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(self.scaled_data[feature1], 
                                                self.scaled_data[feature2])
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(self.scaled_data[feature1], 
                                                 self.scaled_data[feature2])
                    elif method == 'kendall':
                        _, p_val = stats.kendalltau(self.scaled_data[feature1], 
                                                  self.scaled_data[feature2])
                    p_values[i, j] = p_val
                    
        p_values_df = pd.DataFrame(p_values, 
                                 index=self.feature_names, 
                                 columns=self.feature_names)
        
        # Find significant correlations
        significant_corr = []
        for i, feature1 in enumerate(self.feature_names):
            for j, feature2 in enumerate(self.feature_names):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = p_values_df.iloc[i, j]
                    
                    if p_val < significance_level:
                        significant_corr.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation': corr_val,
                            'p_value': p_val,
                            'strength': self._interpret_correlation_strength(abs(corr_val))
                        })
                        
        # Sort by absolute correlation strength
        significant_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix,
            'p_values': p_values_df,
            'significant_correlations': significant_corr,
            'method': method,
            'significance_level': significance_level
        }
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength using archaeological standards."""
        if abs_corr >= 0.7:
            return 'Strong'
        elif abs_corr >= 0.5:
            return 'Moderate'
        elif abs_corr >= 0.3:
            return 'Weak'
        else:
            return 'Very weak'
    
    def assemblage_comparison(self, group_column: str,
                            comparison_type: str = 'all') -> Dict:
        """
        Compare ceramic assemblages between different groups.

        .. note:: Not yet implemented — raises ``NotImplementedError``.

        Parameters
        ----------
        group_column : str
            Column name containing group identifiers
        comparison_type : str, default 'all'
            Type of comparison: ``'all'``, ``'pairwise'``, or
            ``'one_vs_rest'``

        Returns
        -------
        dict
            Comprehensive assemblage comparison results
        """
        # This would require the original unscaled data with group information
        # Implementation would include:
        # - ANOVA/Kruskal-Wallis tests for group differences
        # - Post-hoc tests for pairwise comparisons
        # - Effect size calculations
        # - Discriminant analysis
        # - Visualization of group differences
        
        raise NotImplementedError("Assemblage comparison requires group data - implement based on specific needs")
    
    def generate_report(self, include_plots: bool = True) -> Dict:
        """
        Run PCA, hierarchical clustering, and correlation analysis in
        one call and bundle results into a single report dict.

        Each analysis is run with default parameters (see
        ``perform_pca``, ``perform_clustering``, ``correlation_analysis``
        for details). If any individual analysis fails, its entry will
        contain ``{'error': '<message>'}`` rather than raising.

        Parameters
        ----------
        include_plots : bool, default True
            Whether to include visualization plots in the report
            (currently unused — reserved for future HTML export).

        Returns
        -------
        dict
            - ``'data_summary'`` — sample/feature counts and names.
            - ``'pca_analysis'`` — output of ``perform_pca()``.
            - ``'cluster_analysis'`` — output of ``perform_clustering()``.
            - ``'correlation_analysis'`` — output of
              ``correlation_analysis()``.
            - ``'archaeological_interpretation'`` — auto-generated
              summary of main findings and technological insights.
        """
        if self.scaled_data is None:
            raise ValueError("Data must be prepared first using prepare_data()")
            
        report = {
            'data_summary': {
                'n_samples': len(self.scaled_data),
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'sample_names': self.sample_names
            }
        }
        
        # Perform all analyses
        try:
            pca_results = self.perform_pca()
            report['pca_analysis'] = pca_results
        except Exception as e:
            report['pca_analysis'] = {'error': str(e)}
            
        try:
            cluster_results = self.perform_clustering()
            report['cluster_analysis'] = cluster_results
        except Exception as e:
            report['cluster_analysis'] = {'error': str(e)}
            
        try:
            corr_results = self.correlation_analysis()
            report['correlation_analysis'] = corr_results
        except Exception as e:
            report['correlation_analysis'] = {'error': str(e)}
            
        # Add archaeological interpretations
        report['archaeological_interpretation'] = self._generate_archaeological_summary(report)
        
        return report
    
    def _generate_archaeological_summary(self, report: Dict) -> Dict:
        """Generate archaeological interpretation summary."""
        summary = {
            'main_findings': [],
            'technological_insights': [],
            'recommendations': []
        }
        
        # Analyze PCA results
        if 'pca_analysis' in report and 'error' not in report['pca_analysis']:
            pca = report['pca_analysis']
            total_variance = sum(pca['explained_variance'][:3])
            
            if total_variance > 0.7:
                summary['main_findings'].append(
                    f"First 3 components explain {total_variance:.1%} of variation - strong dimensional structure"
                )
            
            # Interpret dominant components
            if 'interpretations' in pca:
                for pc, interp in list(pca['interpretations'].items())[:2]:
                    summary['technological_insights'].append(
                        f"{pc}: {interp['archaeological_meaning']}"
                    )
        
        # Analyze clustering results
        if 'cluster_analysis' in report and 'error' not in report['cluster_analysis']:
            cluster = report['cluster_analysis']
            n_clusters = cluster.get('n_clusters', 0)
            
            if n_clusters > 1:
                silhouette = cluster.get('silhouette_score', 0)
                if silhouette > 0.5:
                    summary['main_findings'].append(
                        f"Clear grouping structure identified ({n_clusters} clusters, silhouette={silhouette:.2f})"
                    )
                elif silhouette > 0.25:
                    summary['main_findings'].append(
                        f"Moderate grouping structure ({n_clusters} clusters, silhouette={silhouette:.2f})"
                    )
        
        # Analyze correlations
        if 'correlation_analysis' in report and 'error' not in report['correlation_analysis']:
            corr = report['correlation_analysis']
            strong_corr = [c for c in corr['significant_correlations'] if c['strength'] == 'Strong']
            
            if strong_corr:
                summary['technological_insights'].append(
                    f"Strong correlations found between key variables (n={len(strong_corr)})"
                )
        
        # Add recommendations
        summary['recommendations'] = [
            "Consider archaeological context when interpreting statistical patterns",
            "Validate clustering results with petrographic analysis if available",
            "Compare results with known technological traditions from the region",
            "Consider taphonomic factors that might affect measurements"
        ]
        
        return summary


def quick_ceramic_analysis(data: Union[pd.DataFrame, Dict],
                          scaling_method: str = 'standard') -> Dict:
    """
    One-liner convenience wrapper: prepare data and run ``generate_report()``.

    Parameters
    ----------
    data : pd.DataFrame or dict
        Ceramic analysis data (e.g. output of ``full_analysis()``).
    scaling_method : str, default 'standard'
        Scaling method passed to ``prepare_data()`` — see that method
        for options (``'standard'``, ``'robust'``, ``'none'``).

    Returns
    -------
    dict
        Complete analysis report (same as ``generate_report()``).
    """
    analyzer = CeramicStatisticalAnalyzer()
    analyzer.prepare_data(data, scaling_method=scaling_method)
    return analyzer.generate_report(include_plots=False)


class CeramicVisualization:
    """
    Static visualization methods for ceramic statistical analysis results.

    All methods are ``@staticmethod`` — no instantiation required::

        from core.statistics import CeramicVisualization as viz

        fig = viz.plot_pca_biplot(pca_results)
        fig = viz.plot_dendrogram(cluster_results)
        fig = viz.plot_correlation_heatmap(corr_results)

    Every method returns a ``plotly.graph_objects.Figure`` that can be
    displayed with ``fig.show()`` or saved with ``fig.write_image()``.
    """
    
    @staticmethod
    def plot_pca_biplot(pca_results: Dict, pc_x: str = 'PC1', pc_y: str = 'PC2',
                       show_loadings: bool = True, max_arrows: int = 10) -> go.Figure:
        """
        Create PCA biplot with sample scores and feature-loading arrows.

        Parameters
        ----------
        pca_results : dict
            Output of ``CeramicStatisticalAnalyzer.perform_pca()``.
        pc_x, pc_y : str, default 'PC1' / 'PC2'
            Which principal components to plot on the x/y axes. Use
            ``'PC3'``, ``'PC4'``, etc. to explore higher components.
        show_loadings : bool, default True
            Overlay feature-loading vectors as arrows. Arrows point in
            the direction each feature contributes to the two plotted
            components; longer arrows indicate stronger influence.
        max_arrows : int, default 10
            Cap on the number of loading arrows shown (top features by
            combined loading magnitude). Reduces clutter in high-
            dimensional datasets.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive PCA biplot.
        """
        scores_df = pca_results['scores']
        loadings_df = pca_results['loadings']
        
        # Create scatter plot of scores
        fig = px.scatter(
            scores_df, x=pc_x, y=pc_y,
            hover_data=['Sample'],
            title=f'PCA Biplot: {pc_x} vs {pc_y}',
            labels={
                pc_x: f'{pc_x} ({pca_results["explained_variance"][int(pc_x[2:])-1]:.1%} variance)',
                pc_y: f'{pc_y} ({pca_results["explained_variance"][int(pc_y[2:])-1]:.1%} variance)'
            }
        )
        
        if show_loadings and pc_x in loadings_df.columns and pc_y in loadings_df.columns:
            # Select top loading vectors by magnitude
            loadings_subset = loadings_df[[pc_x, pc_y]]
            loadings_magnitude = np.sqrt(loadings_subset[pc_x]**2 + loadings_subset[pc_y]**2)
            top_loadings = loadings_magnitude.nlargest(max_arrows).index
            
            # Add loading vectors
            for feature in top_loadings:
                loading_x = loadings_df.loc[feature, pc_x]
                loading_y = loadings_df.loc[feature, pc_y]
                
                # Scale loadings for visualization
                scale_factor = 3.0
                loading_x *= scale_factor
                loading_y *= scale_factor
                
                # Add arrow
                fig.add_annotation(
                    x=loading_x, y=loading_y,
                    ax=0, ay=0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor='red', opacity=0.7
                )
                
                # Add label
                fig.add_annotation(
                    x=loading_x * 1.1, y=loading_y * 1.1,
                    text=feature, showarrow=False,
                    font=dict(size=10), opacity=0.8
                )
        
        fig.update_layout(
            showlegend=False,
            width=800, height=600,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_pca_3d(pca_results: Dict, pc_x: str = 'PC1', pc_y: str = 'PC2',
                   pc_z: str = 'PC3', cluster_labels: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create interactive 3D PCA scatter plot.

        Parameters
        ----------
        pca_results : dict
            Output of ``CeramicStatisticalAnalyzer.perform_pca()``.
        pc_x, pc_y, pc_z : str, default 'PC1' / 'PC2' / 'PC3'
            Principal components mapped to the x / y / z axes.
        cluster_labels : array-like of int, optional
            Cluster assignment per sample (e.g. from
            ``perform_clustering()['cluster_labels']``). When provided,
            points are color-coded by cluster; noise samples (label
            ``-1`` from DBSCAN) appear as "Noise".

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive 3D scatter (rotate/zoom with mouse).
        """
        scores_df = pca_results['scores']
        
        # Add cluster colors if provided
        if cluster_labels is not None:
            scores_df = scores_df.copy()
            scores_df['Cluster'] = [f'Cluster {label+1}' if label >= 0 else 'Noise' 
                                  for label in cluster_labels]
            color_column = 'Cluster'
        else:
            color_column = None
        
        fig = px.scatter_3d(
            scores_df, x=pc_x, y=pc_y, z=pc_z,
            color=color_column,
            hover_data=['Sample'],
            title='3D PCA Plot',
            labels={
                pc_x: f'{pc_x} ({pca_results["explained_variance"][int(pc_x[2:])-1]:.1%})',
                pc_y: f'{pc_y} ({pca_results["explained_variance"][int(pc_y[2:])-1]:.1%})',
                pc_z: f'{pc_z} ({pca_results["explained_variance"][int(pc_z[2:])-1]:.1%})'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f'{pc_x} ({pca_results["explained_variance"][int(pc_x[2:])-1]:.1%})',
                yaxis_title=f'{pc_y} ({pca_results["explained_variance"][int(pc_y[2:])-1]:.1%})',
                zaxis_title=f'{pc_z} ({pca_results["explained_variance"][int(pc_z[2:])-1]:.1%})'
            ),
            width=800, height=600
        )
        
        return fig
    
    @staticmethod
    def plot_scree_plot(pca_results: Dict) -> go.Figure:
        """
        Create scree plot showing per-component explained variance.

        Displays both individual (bars) and cumulative (line) variance
        ratios, helping determine how many components to retain.

        Parameters
        ----------
        pca_results : dict
            Output of ``CeramicStatisticalAnalyzer.perform_pca()``.

        Returns
        -------
        plotly.graph_objects.Figure
            Scree / elbow plot.
        """
        n_components = len(pca_results['explained_variance'])
        components = [f'PC{i+1}' for i in range(n_components)]
        
        fig = go.Figure()
        
        # Individual variance
        fig.add_trace(go.Scatter(
            x=components,
            y=pca_results['explained_variance'] * 100,
            mode='lines+markers',
            name='Individual',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=components,
            y=pca_results['cumulative_variance'] * 100,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Add 95% variance line
        fig.add_hline(y=95, line_dash="dash", line_color="gray",
                     annotation_text="95% Variance")
        
        fig.update_layout(
            title='PCA Scree Plot - Explained Variance',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance (%)',
            template='plotly_white',
            width=800, height=500,
            legend=dict(x=0.7, y=0.95)
        )
        
        return fig
    
    @staticmethod
    def plot_dendrogram(cluster_results: Dict, orientation: str = 'bottom',
                       max_labels: int = 20, sample_names: list = None) -> go.Figure:
        """
        Create hierarchical clustering dendrogram.

        Parameters
        ----------
        cluster_results : dict
            Results from hierarchical clustering
        orientation : str, default 'bottom'
            Dendrogram orientation
        max_labels : int, default 20
            Maximum number of labels to show
        sample_names : list, optional
            Labels for leaf nodes. If None, uses cluster_results['sample_names']
            if available, otherwise falls back to numeric indices.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive dendrogram
        """
        if cluster_results['method'] != 'hierarchical':
            raise ValueError("Dendrogram requires hierarchical clustering results")

        linkage_matrix = cluster_results['linkage_matrix']
        n_clusters = cluster_results.get('n_clusters', 2)

        if sample_names is None:
            sample_names = cluster_results.get('sample_names')

        # Create dendrogram using scipy
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, fcluster

        # Determine if truncation is needed
        truncating = len(linkage_matrix) > max_labels

        # Find the distance threshold that produces the desired number of clusters
        # so scipy colors branches by cluster membership
        if n_clusters is not None and n_clusters >= 2 and len(linkage_matrix) >= 1:
            # Distances at which merges happen are in column 2 of the linkage matrix
            merge_distances = linkage_matrix[:, 2]
            # The threshold sits between the (n_clusters)th-to-last and
            # (n_clusters-1)th-to-last merge distances
            if n_clusters <= len(merge_distances):
                idx = len(merge_distances) - n_clusters
                color_threshold = (merge_distances[idx] + merge_distances[idx + 1]) / 2
            else:
                color_threshold = None
        else:
            color_threshold = None

        # Calculate dendrogram
        dend_kwargs = dict(
            no_plot=True,
            truncate_mode='lastp' if truncating else None,
            p=max_labels,
            color_threshold=color_threshold,
        )
        if sample_names is not None and not truncating:
            dend_kwargs['labels'] = np.array(sample_names)

        dend = scipy_dendrogram(linkage_matrix, **dend_kwargs)

        # Build a mapping from scipy's default color keys to the Set1 palette
        # so that each cluster branch gets a distinct, recognisable color
        palette = px.colors.qualitative.Set1
        scipy_color_keys = sorted(
            {c for c in dend['color_list'] if c != 'C0'},
            key=lambda c: dend['color_list'].index(c)
            if c in dend['color_list'] else 999
        )
        color_map = {key: palette[i % len(palette)]
                     for i, key in enumerate(scipy_color_keys)}
        # Links above the threshold get a neutral grey
        color_map['C0'] = '#999999'

        # Create plotly figure
        fig = go.Figure()

        # Add dendrogram lines colored by cluster
        for i, d, sc in zip(dend['icoord'], dend['dcoord'],
                            dend['color_list']):
            fig.add_trace(go.Scatter(
                x=i, y=d,
                mode='lines',
                line=dict(color=color_map.get(sc, '#999999'), width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add legend entries for each cluster color
        for sc_key, plotly_color in color_map.items():
            if sc_key == 'C0':
                label = 'Above threshold'
            else:
                idx = scipy_color_keys.index(sc_key) + 1
                label = f'Cluster {idx}'
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=plotly_color, width=2),
                name=label,
                showlegend=True,
            ))

        # Use filename labels on x-axis when available and not truncating
        has_labels = sample_names is not None and not truncating
        xaxis_title = 'Sample' if has_labels else 'Sample Index'

        layout_kwargs = dict(
            title='Hierarchical Clustering Dendrogram',
            xaxis_title=xaxis_title,
            yaxis_title='Distance',
            template='plotly_white',
            width=1000, height=600,
            showlegend=True
        )

        if has_labels:
            # Map scipy's x-tick positions to the leaf labels
            leaf_labels = [dend['ivl'][idx] for idx in range(len(dend['ivl']))]
            tick_positions = list(range(5, 10 * len(leaf_labels), 10))
            layout_kwargs['xaxis'] = dict(
                tickmode='array',
                tickvals=tick_positions,
                ticktext=leaf_labels,
                tickangle=-45,
                title=xaxis_title
            )

        fig.update_layout(**layout_kwargs)

        return fig
    
    @staticmethod
    def plot_correlation_heatmap(correlation_results: Dict) -> go.Figure:
        """
        Create interactive correlation-matrix heatmap.

        Parameters
        ----------
        correlation_results : dict
            Output of ``CeramicStatisticalAnalyzer.correlation_analysis()``.

        Returns
        -------
        plotly.graph_objects.Figure
            Heatmap colored by correlation coefficient (−1 to +1).
        """
        corr_matrix = correlation_results['correlation_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=800, height=800,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_cluster_comparison(cluster_results: Dict, feature_data: pd.DataFrame,
                              features_to_plot: List[str] = None) -> go.Figure:
        """
        Create box-plot grid comparing feature distributions across clusters.

        Parameters
        ----------
        cluster_results : dict
            Output of ``CeramicStatisticalAnalyzer.perform_clustering()``
            (any method).
        feature_data : pd.DataFrame
            The (unscaled) feature data — typically
            ``analyzer.scaled_data`` or the original DataFrame subset.
        features_to_plot : list of str, optional
            Column names to include. If ``None``, the first 4 columns of
            *feature_data* are used.

        Returns
        -------
        plotly.graph_objects.Figure
            Grid of box plots, one per feature, colored by cluster.
        """
        cluster_labels = cluster_results['cluster_labels']
        
        # Select features to plot
        if features_to_plot is None:
            # Select first 4 features or all if fewer
            features_to_plot = feature_data.columns[:min(4, len(feature_data.columns))]
        
        # Create subplots
        n_features = len(features_to_plot)
        cols = min(2, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=features_to_plot,
            vertical_spacing=0.1
        )
        
        # Add data for each cluster
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set1[:len(unique_clusters)]
        
        for idx, feature in enumerate(features_to_plot):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            for cluster_id, color in zip(unique_clusters, colors):
                mask = cluster_labels == cluster_id
                cluster_name = f'Cluster {cluster_id+1}' if cluster_id >= 0 else 'Noise'
                
                fig.add_trace(
                    go.Box(
                        y=feature_data.loc[mask, feature],
                        name=cluster_name,
                        marker_color=color,
                        showlegend=(idx == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Cluster Feature Comparison',
            height=300 * rows,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_archaeological_summary(report: Dict) -> go.Figure:
        """
        Create a 2x2 dashboard summarizing PCA variance, cluster quality,
        top correlations, and sample distribution.

        Parameters
        ----------
        report : dict
            Output of ``CeramicStatisticalAnalyzer.generate_report()``.

        Returns
        -------
        plotly.graph_objects.Figure
            Four-panel summary figure.
        """
        # Create a summary dashboard-style plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'PCA Variance Explained',
                'Cluster Quality Metrics',
                'Top Correlations',
                'Sample Distribution'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # PCA variance (if available)
        if 'pca_analysis' in report and 'explained_variance' in report['pca_analysis']:
            pca = report['pca_analysis']
            n_components = min(5, len(pca['explained_variance']))
            components = [f'PC{i+1}' for i in range(n_components)]
            
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=pca['explained_variance'][:n_components] * 100,
                    name='Variance %',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
        
        # Cluster metrics (if available)
        if 'cluster_analysis' in report and 'silhouette_score' in report['cluster_analysis']:
            cluster = report['cluster_analysis']
            metrics = ['Silhouette Score', 'Calinski-Harabasz']
            values = [
                cluster.get('silhouette_score', 0) * 100,  # Scale to 0-100
                min(cluster.get('calinski_harabasz_score', 0) / 100, 100)  # Scale and cap
            ]
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values,
                    name='Quality',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
        
        # Top correlations (if available)
        if 'correlation_analysis' in report and 'significant_correlations' in report['correlation_analysis']:
            corr = report['correlation_analysis']['significant_correlations'][:5]  # Top 5
            if corr:
                feature_pairs = [f"{c['feature1'][:8]}...{c['feature2'][:8]}" for c in corr]
                correlations = [abs(c['correlation']) for c in corr]
                
                fig.add_trace(
                    go.Bar(
                        x=feature_pairs,
                        y=correlations,
                        name='|Correlation|',
                        marker_color='lightcoral'
                    ),
                    row=2, col=1
                )
        
        # Sample distribution (placeholder - would show spatial or temporal distribution)
        if 'data_summary' in report:
            n_samples = report['data_summary']['n_samples']
            n_features = report['data_summary']['n_features']
            
            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3],
                    y=[n_samples, n_features, n_samples/n_features],
                    mode='markers+text',
                    text=['Samples', 'Features', 'Ratio'],
                    textposition='top center',
                    marker=dict(size=[n_samples/5, n_features/2, 20], 
                               color=['blue', 'red', 'green']),
                    name='Data Overview'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Archaeological Analysis Summary Dashboard',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig


def compare_assemblages(data: pd.DataFrame, group_column: str,
                       analysis_type: str = 'comprehensive') -> Dict:
    """
    Compare ceramic assemblages between different archaeological contexts.

    .. note:: Not yet implemented — raises ``NotImplementedError``.

    Parameters
    ----------
    data : pd.DataFrame
        Ceramic data with group identifiers
    group_column : str
        Column containing group/context identifiers
    analysis_type : str, default 'comprehensive'
        Type of analysis: ``'basic'``, ``'comprehensive'``, or
        ``'advanced'``

    Returns
    -------
    dict
        Assemblage comparison results
    """
    # Implementation for assemblage comparison
    # Would include statistical tests, effect sizes, discriminant analysis
    raise NotImplementedError("Assemblage comparison functionality - implement based on specific research needs")