# HeliXplore: A Python package for analyzing multi-strand helix deformations

**Date:** 2025-10-24

**Authors:**
- Ronnie Mondal<sup>1,2</sup> ([ORCID: 0000-0003-4393-018X](https://orcid.org/0000-0003-4393-018X))
- Valerie Vaissier Welborn<sup>1,2,*</sup> ([ORCID: 0000-0003-0834-4441](https://orcid.org/0000-0003-0834-4441))

**Affiliations:**
1. Department of Chemistry, Virginia Tech, Blacksburg, VA 24061
2. Macromolecules Innovation Institute, Virginia Tech, Blacksburg, VA 24061

**Email:** <vwelborn@vt.edu>

**Tags:** molecular modeling, scientific computing, helix deformation, Python

---

## Summary

Helices are a recurring structural motif in biomolecular systems, most prominently found in DNA and collagen [@doi:10.1038/171737a0; @doi:10.1038/174269c0]. Multi-stranded helices can also appear in other biological and synthetic polymers and macromolecules, where mechanical and functional roles are often tied to deformations. These structures undergo distortions at both local and global scales, sensitive to sequence variation, mutations, and environmental perturbations. Current Molecular Dynamics (MD) studies lack a consistent way to quantify helix deformation. Existing metrics like polysaccharide atom distances [@doi:10.1039/D1RA00071C], collagen cross-sectional triangles [@doi:10.1002/prot.22026], pitch, or principal axis measures [@doi:10.1021/ja057693+; @doi:10.1021/acs.jctc.9b00630] are system dependent and miss key local or collective distortions, so their interpretation changes with context. Therefore, without unit-level or inter-strand descriptors, comparisons across sequences, mutations, or conditions remain largely qualitative. 

We present `HeliXplore`, an open-source Python package for systematic and quantitative analysis of multi-strand helix deformation. Inspired by collagen but generalizable to any helical bundle, `HeliXplore` characterizes how helices deviate from ideal geometry using strand backbone markers. `HeliXplore` runs calculations in three sections: Section 1 for intra-strand deformations in rise, radius and twist at the unit-, strand- or time-level, for any number of strands; Section 2 for local deformations, helical regularity, axial shifts and inter-strand deformations; and Section 3 for inter-helix deformations in the special case of the triple helix, using the size and shape of the cross-sectional triangle. 

In practice, one only needs `HeliXplore.py`, the MD trajectory file (in TINKER `.arc` format or the standard RCSB `.pdb` format) and the number of strands to be able to run the code. The number of units in each strand is an optional input. Users can also input the atom names or atom types (for TINKER `.arc` format) to select for helix backbone markers. The `read_tinker_arc()` or `read_traj_pdb()` functions can also be replaced to cater to other trajectory file formats. It checks for the four required Python dependencies (`numpy`, `scipy`, `pandas` and `matplotlib`) before running the main code, and no other installations are required. For a detailed description of the inputs, commands to run example MD trajectories, and other notes, see the `README` file on GitHub. A shorter description is provided in the menu in `python HeliXplore.py --help`.

## Statement of Need

`HeliXplore` provides the first open-source, Python-based implementation of a quantitative framework for analyzing poly-helix deformations. By resolving intra- and inter-helix deformations, `HeliXplore` enables unit-level resolution of structural distortions that have previously eluded MD studies. It provides a platform for systematic comparisons across sequences, mutations, and force fields. More broadly, `HeliXplore` establishes a transferable methodology for analyzing helices, positioning it as a foundational tool for future studies of collagen (or other biomolecules') biophysics, synthetic mimetics, and disease-related structural perturbations. Users of `HeliXplore` are free to modify the code and underlying mathematical formulations to adapt to their specific research needs.

## Mathematics

### Section I:

For a multi-strand helix of length $N$, with coordinates of unit $i$ in strand $m$ over the trajectory being $\mathbf{p}^{i,m}(t)$, the __helical axis vector__ $\mathbf{v}^m_1(t)$ is determined via PCA, satisfying

$$\left[\frac{1}{N}\sum_{i=1}^{N}(\mathbf{p}^{i,m}(t) - \mathbf{c}^m(t))(\mathbf{p}^{i,m}(t) - \mathbf{c}^m(t))^T\right] \mathbf{v}^m_1(t) = \lambda^m_1(t) \mathbf{v}^m_1(t)$$

where $\mathbf{c}^m(t)$ are the average coordinates of strand $m$ at time $t$ and $\lambda^m_1$ indicates that it is the eigenvalue of the principal eigenvector. The unit vector of $\mathbf{v}^m_1(t)$ is hereafter referred to as $\hat{\mathbf{v}}^m_1(t)$.

For intra-helix metrics, three consecutive units of a strand are chosen to form a trimer. __Deviations in rise__ ($\text{Rise}^{i,m}(t)$) for unit $i$ at strand $m$ are calculated as 

$$\text{Rise}^{i,m}(t) = ((\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i,m}(t)) \cdot \mathbf{v}^m_1(t) - \text{Rise}^{i,m}(0)) \hspace{2pt} / \hspace{2pt} \text{Rise}^{i,m}(0)$$

__Deviations in radius__ ($\text{Radius}^{i,m}(t)$) are calculated using the circumradius $R^{i,m}$ as

$$\text{Radius}^{i,m}(t) = \left(\text{R}^{i,m}(t) - \text{R}^{i,m}(0) \right) / \text{R}^{i,m}(0)$$

where

$$\text{R}^{i,m}(t) = \dfrac{\|\mathbf{p}^{i,m}(t) - \mathbf{p}^{i-1,m}(t)\|  \|\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i-1,m}(t)\| \|\mathbf{p}^{i,m}(t) - \mathbf{p}^{i+1,m}(t)\|}{4 \times \frac{1}{2}\|(\mathbf{p}^{i,m}(t) - \mathbf{p}^{i-1,m}(t)) \times (\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i-1,m}(t))\|}$$

__Deviations in twist__ ($\text{Twist}^{i,m}(t)$) for consecutive trimers on strand $m$ are calculated as the dihedral angle between the normals of the planes defined by units $(i-1, i, i+1)$ ($\mathbf{n}^{i,m}_a(t)$) and $(i, i+1, i+2)$ ($\mathbf{n}^{i,m}_b(t)$) as

$$\text{Twist}^{i,m}(t) = \left(\arccos\left(\frac{\mathbf{n}^{i,m}_a(t) \cdot \mathbf{n}^{i,m}_b(t)}{\|\mathbf{n}^{i,m}_a(t)\|\|\mathbf{n}^{i,m}_b(t)\|}\right) - \theta^{i,m}(0) \right) \hspace{2pt} / \hspace{2pt} \theta^{i,m}(0)$$

__Intra-strand deviations__ ($\kappa^{i,m}(t)$) can then be calculated from the output files as

$$\langle \kappa^{i,m}(t) \rangle = w_{\text{Rise}} * \langle \text{Rise}^{i,m}(t) \rangle + w_{\text{Radius}} * \langle \text{Radius}^{i,m}(t) \rangle + w_{\text{Twist}} * \langle \text{Twist}^{i,m}(t) \rangle$$

where $w_{\text{Rise}}$, $w_{\text{Radius}}$ and $w_{\text{Twist}}$ are user-defined weights during post-processing and the average can be taken over units, strands, time or a combination of the three.

### Section II:

__Local deformations__ ($d^{i,m}_{\text{local}}(t)$) are calculated using a sliding window approach quantifying local deviations after optimal superposition. A window around unit $i$ is defined and then aligned to the corresponding segment in the reference structure using the Kabsch algorithm [@doi:10.1107/S0567739476001873].

$$\mathbf{q}^{i,m}(t) = \mathbf{K} \left(\mathbf{p}^{i,m}(t),\mathbf{p}^{i,m}(0) \right)$$

where $\mathbf{q}^{i,m}(t)$ is the representative position of the aligned unit at time $t$, and $\mathbf{K}$ is Kabsch alignment operator, which returns the optimally rotated and translated version of $\mathbf{p}^{i,m}(t)$ aligned to $\mathbf{p}^{i,m}(0)$. The scalar positional deviations are then calculated as

$$d^{i,m}_{\text{local}}(t) = \|\mathbf{q}^{i,m}(t) - \mathbf{p}^{i,m}(0)\|$$

__Helical regularities__ ($\mathcal{R}^m(t)$) for a strand $m$ are calculated as 

$$\mathcal{R}^{m}(t) = \frac{\sigma^i_{\text{Rise}^{i,m}(t)}}{\langle |\text{Rise}^{i,m}(t)| \rangle_i} + \frac{\sigma^i_{\text{Radius}^{i,m}(t)}}{\langle |\text{Radius}^{i,m}(t)| \rangle_i} + \frac{\sigma^i_{\text{Twist}^{i,m}(t)}}{\langle |\text{Twist}^{i,m}(t)| \rangle_i}$$

where $\sigma^i$ means the standard deviation is done over $i$.

For inter-helix metrics, angles and distances between each strand pair are calculated. 

__Axial shifts__ of one strand with respect to another along the central axis are calculated by first reorienting consistently to prevent sign ambiguity of direction. $\tilde{\mathbf{v}}_1^m(t)$ is defined by enforcing $\hat{\mathbf{v}}_1^m\cdot\hat{\mathbf{v}}_1^n>0 \hspace{2pt} \forall m,n \text{ and } m \neq n$. If found negative, $\tilde{\mathbf{v}}_1^n(t) = -\hat{\mathbf{v}}_1^n$, otherwise $\tilde{\mathbf{v}}_1^n(t) = \hat{\mathbf{v}}_1^n$. The axial shifts ($s^{mn}(t)$) are then calculated as

$$s^{mn}(t) = \bigl| \langle \tilde{\mathbf{v}}_1^{m}(t) \rangle_m \cdot (\mathbf{c}^m(t) - \mathbf{c}^n(t))\bigr|$$

__Deviations in axis angles__ ($\delta_{\theta}^{mn}(t)$) between the unit vectors of the principal axes of the two strands $m$ and $n$ are calculated as

$$\delta_{\theta}^{mn}(t) = \frac{|\arccos(\hat{\mathbf{v}}_1^m(t) \cdot \hat{\mathbf{v}}_1^n(t)) - \theta^{mn}(0)|}{\theta^{mn}(0)}$$

__Deviations in axis distances__ ($\delta_d^{mn}(t)$) are calculated as

$$\delta_d^{mn}(t) = \left( |(\mathbf{c}^m(t) - \mathbf{c}^n(t)) \cdot (\hat{\mathbf{v}}_1^m(t) \times \hat{\mathbf{v}}_1^n(t))| - \delta_d^{mn}(0) \right) / \delta_d^{mn}(0)$$

__Deviations in centroid distances__ ($d_c^{mn}(t)$) are calculated as the distance between the average coordinates for each strand

$$d_c^{mn}(t) = \left(\| \mathbf{c}^m(t) - \mathbf{c}^n(t)\| - d_c^{mn}(0) \right) / d_c^{mn}(0)$$

### Section III:

For the special case of a triple helix, one unit on each strand is taken to form a triangular cross-section at that unit. 

__Deviations in size__ ($\text{Size}^i(t)$) are calculated from the area of the triangle ($\text{Area}^i(t)$) as 

$$\text{Size}^i(t) = \left(\frac{1}{2} \|(\mathbf{p}^{i,2}(t) - \mathbf{p}^{i,1}(t)) \times (\mathbf{p}^{i,3}(t) - \mathbf{p}^{i,1}(t))\| - \text{Area}^i(0)\right) / \text{Area}^i (0)$$ 

__Deviations in shape__ ($\text{Shape}^i(t)$) are calculated from the normalized isoperimetric ratio (IP) as 

$$\text{Shape}^i(t) = \left( \frac{4\pi \text{Area}^i(t)}{P^i(t)^2} - \text{IP}^i(0) \right) / \text{IP}^i(0)$$ 

where $P^i(t)$ is the corresponding perimeter.

__Inter-strand deviations__ ($\zeta^{i}(t)$) can then be calculated for a triple helix from the output files as

$$\zeta^{i}(t) = w_{\text{Size}} * \langle \text{Size}^{i}(t) \rangle + w_{\text{Shape}} * \langle \text{Shape}^{i}(t) \rangle$$

where $w_{\text{Size}}$ and $w_{\text{Shape}}$ are user-defined weights during post-processing and the average can be done over units or time or both.

The first frame is taken as the reference. The reference can be changed by appending a new reference frame to the beginning of the input MD trajectory. All mathematical details are also outlined as comments in the code.

## Acknowledgements

The authors thank the National Institute of Health, National Institute of General Medical Sciences, under award number R35-GM150409 and Advanced Research Computing at Virginia Tech for providing computational resources and technical support that have contributed to the results reported within this paper.

## References
