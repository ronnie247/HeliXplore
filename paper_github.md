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

# Summary

Multi-stranded helices are a recurring structural motif in biomolecular systems, most prominently found in DNA and collagen [@doi:10.1038/171737a0; @doi:10.1038/174269c0]. They also appear in synthetic polymers and macromolecules [@doi:10.1021/cr900162q]. Multi-stranded helices can undergo local and global deformations that directly impact their function. Helix properties depend on strand length, strand composition, mutations and change in the environment, which can be modeled using classical Molecular Dynamics (MD). However, MD lacks a tool to systematically quantify multi-strand helix deformation across systems and conditions. In the past, people have used polysaccharide atom distances [@doi:10.1039/D1RA00071C], collagen cross-sectional triangles [@doi:10.1002/prot.22026], pitch, or principal axis measures [@doi:10.1021/ja057693+; @doi:10.1021/acs.jctc.9b00630]. Although adequate for the system under investigation, these metrics are not generally useful and miss key local or collective distortions. Without local or inter-strand descriptors, comparisons across systems remain largely qualitative. 

We present `HeliXplore`, an open-source Python package for the systematic and quantitative analysis of multi-strand helix deformation. Inspired by collagen but generalizable to any helical bundle, `HeliXplore` measures how helices deviate from their ideal geometry using backbone markers. `HeliXplore` runs calculations in three sections: Section 1 for intra-strand deformations in rise, radius and twist per atom or group of atoms and per time frame; Section 2 for axial shifts, inter-strand deformations, local deformations and helical regularity; and Section 3 for triple-helix deformations, using the area and shape of the cross-sectional triangle. 

In practice, one only needs `HeliXplore.py`, the MD trajectory file (in TINKER `.arc` format or the standard RCSB `.pdb` format) and the number of strands to be able to run the code. The number of units in each strand is an optional input. Users can also input the atom names or atom types (for TINKER `.arc` format) to select backbone markers. `read_tinker_arc()` and `read_traj_pdb()` functions can be replaced to cater to other trajectory file formats. `HeliXplore` checks for the four required Python dependencies (`numpy`, `scipy`, `pandas` and `matplotlib`) before running the main code, and no other installations are required. For a detailed description of the inputs and examples, see the `README` file on GitHub. A shorter description is provided with `python HeliXplore.py --help`.

## Statement of Need

`HeliXplore` provides the first open-source, Python-based implementation of a quantitative framework for analyzing poly-helix deformations. By resolving intra- and inter-helix deformations, `HeliXplore` enables atomic resolution of structural distortions that have previously eluded MD studies. More broadly, `HeliXplore` establishes a transferable methodology for analyzing helices, positioning it as a foundational tool for systematic comparisons across systems, conditions and force fields. Users of `HeliXplore` are free to modify the code and underlying mathematical formulations to adapt to their specific research needs.

# Mathematics

### Section I:

For a helix of length $N$, with $\mathbf{p}^{i,m}(t)$ the coordinates of atom $i$ in strand $m$ over time $t$, the __helical axis vector__ $\mathbf{v}^m_1(t)$ is determined via principal component analysis:

$$ 
\left[\frac{1}{N}\sum_{i=1}^{N}(\mathbf{p}^{i,m}(t) - \mathbf{c}^m(t))(\mathbf{p}^{i,m}(t) - \mathbf{c}^m(t))^T\right] \mathbf{v}^m_1(t) = \lambda^m_1(t) \mathbf{v}^m_1(t)
$$

where $\mathbf{c}^m(t)$ is the average coordinates of strand $m$ at time $t$ defined as:

$$
\mathbf{c}^m(t) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{p}^{i,m}(t)
$$

and $\lambda^m_1$ indicates the principal eigenvalue. The unit vector of $\mathbf{v}^m_1(t)$ is hereafter referred to as $\hat{\mathbf{v}}^m_1(t)$.

For intra-helix metrics, two consecutive atoms of a strand are chosen for rise and three atoms for the radius and twist. __Deviations in rise__ for unit $i$ at strand $m$ are calculated as: 

$$
\delta^{i,m}_{\text{Rise}}(t) = ((\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i,m}(t)) \cdot \mathbf{v}^m_1(t) - \text{Rise}^{i,m}(0)) \hspace{2pt} / \hspace{2pt} \text{Rise}^{i,m}(0)
$$

__Deviations in radius__ are calculated using the circumradius $R^{i,m}$ as:

$$
\delta^{i,m}_{\text{Radius}}(t) = \left(\text{R}^{i,m}(t) - \text{R}^{i,m}(0) \right) / \text{R}^{i,m}(0)
$$

where:

$$
\text{R}^{i,m}(t) =
\dfrac{\|\mathbf{p}^{i,m}(t) - \mathbf{p}^{i-1,m}(t)\|  \|\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i-1,m}(t)\| \|\mathbf{p}^{i,m}(t) - \mathbf{p}^{i+1,m}(t)\|}{4 \times \frac{1}{2}\|(\mathbf{p}^{i,m}(t) - \mathbf{p}^{i-1,m}(t)) \times (\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i-1,m}(t))\|}
$$

__Deviations in twist__ for consecutive set of three atoms on strand $m$ are calculated as the angle between the normals of the planes defined by atoms $(i-1, i, i+1)$ ($\mathbf{n}^{i,m}_a(t)$) and $(i, i+1, i+2)$ ($\mathbf{n}^{i,m}_b(t)$) as:

$$
\delta^{i,m}_{\text{Twist}}(t) = \left(\arccos\left(\frac{\mathbf{n}^{i,m}_a(t) \cdot \mathbf{n}^{i,m}_b(t)}{\|\mathbf{n}^{i,m}_a(t)\|\|\mathbf{n}^{i,m}_b(t)\|}\right) - \text{Twist}^{i,m}(0) \right) \hspace{2pt} / \hspace{2pt} \text{Twist}^{i,m}(0)
$$

__Intra-strand deviations__ ($\kappa^{i,m}(t)$) can then be calculated from the output files as:

$$
\langle \kappa^{i,m}(t) \rangle = w_{\text{Rise}} * \langle \delta^{i,m}_{\text{Rise}}(t) \rangle + w_{\text{Radius}} * \langle \delta^{i,m}_{\text{Radius}}(t) \rangle + w_{\text{Twist}} * \langle \delta^{i,m}_{\text{Twist}}(t) \rangle
$$

where $w_{\text{Rise}}$, $w_{\text{Radius}}$ and $w_{\text{Twist}}$ are user-defined weights during post-processing and the average can be taken over atoms, strands, time or a combination of the three.

### Section II:

__Local deformations__ are calculated using a sliding window approach which quantifies local deviations after optimal superposition. A window around unit $i$ is defined and then aligned to the reference structure using the Kabsch algorithm [@doi:10.1107/S0567739476001873] as:

$$
\mathbf{q}^{i,m}(t) = \mathbf{K} \left(\mathbf{p}^{i,m}(t),\mathbf{p}^{i,m}(0) \right)
$$

where $\mathbf{q}^{i,m}(t)$ is the position of atom $i$ on strand $m$ at time $t$ after alignment, and $\mathbf{K}$ is Kabsch alignment operator, which returns the optimally rotated and translated version of $\mathbf{p}^{i,m}(t)$ aligned to $\mathbf{p}^{i,m}(0)$. The local deformation is then the scalar positional deviation, calculated as

$$
d^{i,m}_{\text{local}}(t) = \|\mathbf{q}^{i,m}(t) - \mathbf{p}^{i,m}(0)\|
$$

__Helical regularity__ for a strand $m$ is calculated as: 

$$
\mathcal{R}^{m}(t) = \dfrac{\sigma^i_{\text{Rise}}(t)}{\langle |\text{Rise}^{i,m}(t)| \rangle_i} + \dfrac{\sigma^i_{\text{Radius}}(t)}{\langle |\text{Radius}^{i,m}(t)| \rangle_i} + \dfrac{\sigma^i_{\text{Twist}}(t)}{\langle |\text{Twist}^{i,m}(t)| \rangle_i}
$$

where $\sigma^i$ is the standard deviation of the metric over $i$. 

__Axial shift__ of one strand $m$ with respect to another strand $n$ along the central axis is calculated by first reorienting consistently to prevent sign ambiguity of direction. $\tilde{\mathbf{v}}_1^m(t)$ is defined by enforcing $\hat{\mathbf{v}}_1^m\cdot\hat{\mathbf{v}}_1^n>0 \hspace{2pt} \forall m,n \text{ and } m \neq n$. If found negative, $\tilde{\mathbf{v}}_1^n(t) = -\hat{\mathbf{v}}_1^n$, otherwise $\tilde{\mathbf{v}}_1^n(t) = \hat{\mathbf{v}}_1^n$. The axial shift is calculated as:

$$
s^{mn}(t) = \bigl| \langle \tilde{\mathbf{v}}_1^{m}(t) \rangle_m \cdot (\mathbf{c}^m(t) - \mathbf{c}^n(t))\bigr|
$$

__Deviations in axis angles__ are calculated between the unit vectors of the principal axes of the two strands $m$ and $n$ as:

$$
\delta_{\theta}^{mn}(t) = \frac{|\arccos(\hat{\mathbf{v}}_1^m(t) \cdot \hat{\mathbf{v}}_1^n(t)) - \theta^{mn}(0)|}{\theta^{mn}(0)}
$$

__Deviations in axis distances__ are calculated using the perpendicular distance between the axes of strands $m$ and $n$ as:

$$
\delta_d^{mn}(t) = \left( |(\mathbf{c}^m(t) - \mathbf{c}^n(t)) \cdot (\hat{\mathbf{v}}_1^m(t) \times \hat{\mathbf{v}}_1^n(t))| - d^{mn}(0) \right) / d^{mn}(0)
$$

__Deviations in centroid distances__ are calculated using the distance between the average coordinates of strands $m$ and $n$ as:

$$
d_c^{mn}(t) = \left(\| \mathbf{c}^m(t) - \mathbf{c}^n(t)\| - c^{mn}(0) \right) / c^{mn}(0)
$$

### Section III:

For triple helices, one atom on each strand is taken to form a triangular cross-section at that unit. 

__Deviations in area__ are calculated from the area of the triangle as 
$$
\delta^{i}_{\text{Area}}(t) = \left(\frac{1}{2} \|(\mathbf{p}^{i,2}(t) - \mathbf{p}^{i,1}(t)) \times (\mathbf{p}^{i,3}(t) - \mathbf{p}^{i,1}(t))\| - \text{Area}^i(0)\right) / \text{Area}^i (0) 
$$ 

__Deviations in shape__ are calculated from the normalized isoperimetric ratio (IP) as 
$$
\delta^{i}_{\text{Shape}}(t) = \left( \frac{4\pi \text{Area}^i(t)}{P^i(t)^2} - \text{IP}^i(0) \right) / \text{IP}^i(0)
$$ 
where $P^i(t)$ is the corresponding perimeter.

__Inter-strand deviations__ ($\zeta^{i}(t)$) can then be calculated for a triple helix from the output files as

$$
\zeta^{i}(t) = w_{\text{Area}} * \langle \delta^{i}_{\text{Area}}(t) \rangle + w_{\text{Shape}} * \langle \delta^{i}_{\text{Shape}}(t) \rangle
$$

where $w_{\text{Area}}$ and $w_{\text{Shape}}$ are user-defined weights during post-processing and the average can be done over atoms or time or both.

The first frame is taken as the reference. The reference can be changed by appending a new reference frame to the beginning of the input MD trajectory. All mathematical details are also outlined as comments in the code.

# Acknowledgements
The authors thank the National Institute of Health, National Institute of General Medical Sciences, under award number R35-GM150409 and Advanced Research Computing at Virginia Tech for providing computational resources and technical support that have contributed to the results reported within this paper.

# References