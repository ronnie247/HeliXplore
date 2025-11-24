---
title: "HeliXplore: A Python package for analyzing multi-strand helix deformations"
tags:
  - molecular modeling
  - scientific computing
  - helix deformation
  - Python
authors:
  - name: Ronnie Mondal
    orcid: 0000-0003-4393-018X
    affiliation: [1, 2]
  - name: Valerie Vaissier Welborn^[Corresponding author]
    orcid: 0000-0003-0834-4441
    affiliation: [1, 2]
    email: vwelborn@vt.edu
    corresponding: true 
affiliations:
  - index: 1
    name: Department of Chemistry, Virginia Tech, Blacksburg, VA 24061
  - index: 2
    name: Macromolecules Innovation Institute, Virginia Tech, Blacksburg, VA 24061
format:
  html:
    toc: true
    number-sections: true
date: 2025-11-24
bibliography: references.bib
---
Corresponding author email: [vwelborn@vt.edu](mailto:vwelborn@vt.edu)


# Summary

Multi-stranded helices are a recurring structural motif in biomolecular systems, most prominently found in DNA and collagen [@watson1953molecular; @ramachandran1954structure]. They also appear in synthetic polymers and macromolecules [@yashima2009helical]. Helix properties depend on strand length, strand composition and the environment, which can be modeled using classical Molecular Dynamics (MD). However, multi-stranded helices can undergo local and global deformations that directly impact their function. MD lacks a tool to systematically quantify deformation across systems and conditions. In the past, people have used polysaccharide atom distances [@khatami2021using], collagen cross-sectional triangles [@ravikumar2008region], pitch, or principal axis measures [@zhang2006direct; @koneru2019quantitative]. Although adequate for the system under investigation, these metrics are not generalizable and miss key local or collective distortions. Without local or inter-strand descriptors, comparisons across systems remain largely qualitative. 

We present `HeliXplore`, an open-source Python package for the systematic and quantitative analysis of multi-strand helix deformation. Originally inspired by collagen, `HeliXplore` is generalizable to any helical bundle, including single-stranded helices. `HeliXplore` measures how helices deviate from their ideal geometry using user-defined backbone atoms. `HeliXplore` runs calculations in three sections: Section 1 for intra-strand deformations (rise, radius, twist and windowed deviations) per atom, or group of atoms and per time frame and helical regularity per strand; Section 2 for inter-strand deformations (axial shifts, axis angle deviations, axis distance deviations and averaged distance deviations); and Section 3 for triple-helix deformations, using the area and shape of the cross-sectional triangle. 

In practice, one only needs `HeliXplore.py`, the MD trajectory file (in [tinker]{.smallcaps} `.arc` format or the standard [rcsb]{.smallcaps} `.pdb` format) and the number of strands to be able to run the code. Users can also input the atom names or atom types (for [tinker]{.smallcaps} `.arc` format) to mark the backbone. `read_tinker_arc()` and `read_traj_pdb()` functions can be replaced to cater to other trajectory file formats. `HeliXplore` checks for the four required Python dependencies (`numpy`, `scipy`, `pandas` and `matplotlib`) before running the main code. No other installations are required. For a detailed description of the inputs and examples, see the `README` file on GitHub. A shorter description is provided with `python HeliXplore.py --help`.

## Statement of Need

`HeliXplore` provides the first open-source, Python-based implementation of a quantitative framework for analyzing multi-helix deformations. By resolving intra- and inter-strand deformations, `HeliXplore` enables atomic resolution of structural distortions that have previously eluded MD studies. More broadly, `HeliXplore` establishes a transferable methodology for analyzing helices, positioning it as a foundational tool for systematic comparisons across systems, conditions and force fields. Users of `HeliXplore` are free to modify the code and underlying mathematical formulations to adapt to their specific research needs.

# Mathematics 

### Section I:

For a helix of length $N$, with $\mathbf{p}^{i,m}(t)$ the coordinates of atom $i$ in strand $m$ at time $t$, the __helical axis vector__ $\mathbf{v}^m_1(t)$ is determined via principal component analysis:

$$ 
\left[\dfrac{1}{N}\sum_{i=1}^{N}(\mathbf{p}^{i,m}(t) - \mathbf{c}^m(t))(\mathbf{p}^{i,m}(t) - \mathbf{c}^m(t))^T\right] \mathbf{v}^m_1(t) = \lambda^m_1(t) \mathbf{v}^m_1(t),
$$

where $\mathbf{c}^m(t)$ is the average coordinates of strand $m$ at time $t$:

$$
\mathbf{c}^m(t) = \dfrac{1}{N} \sum_{i=1}^{N} \mathbf{p}^{i,m}(t),
$$

and $\lambda^m_1$ indicates the principal eigenvalue. The unit vector of $\mathbf{v}^m_1(t)$ is hereafter referred to as $\hat{\mathbf{v}}^m_1(t)$.

__Deviations in rise__ for atom $i$ at strand $m$ are calculated as: 

$$
\delta^{i,m}_{\text{Rise}}(t) = \frac{\overbrace{\big( \mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i,m}(t) \big) \cdot \mathbf{v}^m_1(t)}^{\displaystyle \text{Rise}^{i,m}(t)} - \text{Rise}^{i,m}(0)}{\text{Rise}^{i,m}(0)}.
$$

__Deviations in radius__ are calculated using the circumradius $R^{i,m}$ as:

$$
\delta^{i,m}_{\text{Radius}}(t) = \dfrac{\text{R}^{i,m}(t) - \text{R}^{i,m}(0)}{\text{R}^{i,m}(0)},
$$

where:

$$
\text{R}^{i,m}(t) =
\dfrac{\|\mathbf{p}^{i,m}(t) - \mathbf{p}^{i-1,m}(t)\|  \|\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i-1,m}(t)\| \|\mathbf{p}^{i,m}(t) - \mathbf{p}^{i+1,m}(t)\|}{4 \times \dfrac{1}{2}\|(\mathbf{p}^{i,m}(t) - \mathbf{p}^{i-1,m}(t)) \times (\mathbf{p}^{i+1,m}(t) - \mathbf{p}^{i-1,m}(t))\|}.
$$

__Deviations in twist__ for atom $i$ on strand $m$ at time $t$ are calculated as the angle between the normals of the planes defined by consecutive atoms $(i-1, i, i+1)$ ($\mathbf{n}^{i,m}_a(t)$) and $(i, i+1, i+2)$ ($\mathbf{n}^{i,m}_b(t)$) as:

$$
\delta^{i,m}_{\text{Twist}}(t) = \dfrac{\overbrace{\arccos\left(\dfrac{\mathbf{n}^{i,m}_a(t) \cdot \mathbf{n}^{i,m}_b(t)}{\|\mathbf{n}^{i,m}_a(t)\|\|\mathbf{n}^{i,m}_b(t)\|}\right)}^{\displaystyle \text{Twist}^{i,m}(t)} - \text{Twist}^{i,m}(0)}{\text{Twist}^{i,m}(0)}.
$$

__Windowed deviations__ are calculated after superposition using a window of 5 atoms centered around atom $i$ with the Kabsch algorithm [@kabsch1976solution] as:

$$
d^{i,m}_{\text{local}}(t) = \|\mathbf{q}^{i,m}(t) - \mathbf{p}^{i,m}(0)\|,
$$

where $\mathbf{q}^{i,m}(t)$ are the aligned coordinates of atom $i$ obtained from the window $j$ with coordinates $\mathbf{p}^{j,m}(t)$ as:

$$
\mathbf{q}^{j,m}(t) = \mathbf{R}^{j,m}(t) \left(\mathbf{p}^{j,m}(t) - \frac{1}{5}\sum_{j} \mathbf{p}^{j,m}(t)\right) + \frac{1}{5}\sum_{j} \mathbf{p}^{j,m}(0),
$$

where the optimal rotation matrix $\mathbf{R}^{j,m}(t)$ is obtained by singular value decomposition of the cross-covariance matrix $\mathbf{H}^{j,m}(t)$:

$$
\mathbf{H}^{j,m}(t) = \sum_{j} \left(\mathbf{p}^{j,m}(t) - \frac{1}{5}\sum_{j} \mathbf{p}^{j,m}(t)\right) \left(\mathbf{p}^{j,m}(0) - \frac{1}{5}\sum_{j} \mathbf{p}^{j,m}(0)\right)^T,
$$

such that $\mathbf{H}^{j,m}(t) = \mathbf{U}^{j,m}(t) \mathbf{S}^{j,m}(t) (\mathbf{V}^{j,m}(t))^T$ and $\mathbf{R}^{j,m}(t) = \mathbf{U}^{j,m}(t) (\mathbf{V}^{j,m}(t))^T$ with the constraint that $\det(\mathbf{R}^{j,m}(t)) = +1$ to ensure a proper rotation. If $\det(\mathbf{U}^{j,m}(t)(\mathbf{V}^{j,m}(t))^T) = -1$, the sign of the last column of $\mathbf{V}^{j,m}(t)$ is flipped to fulfil the constraint.

__Helical regularity__ for a strand $m$ is calculated as: 

$$
\mathcal{R}^{m}(t) = \dfrac{\sigma_{\text{Rise}}(t)}{\langle |\text{Rise}^{i,m}(t)| \rangle_i} + \dfrac{\sigma_{\text{Radius}}(t)}{\langle |\text{Radius}^{i,m}(t)| \rangle_i} + \dfrac{\sigma_{\text{Twist}}(t)}{\langle |\text{Twist}^{i,m}(t)| \rangle_i},
$$

where $\sigma$ is the standard deviation of the metric over $i$. 

### Section II:

__Axial shift__ of strand $m$ with respect to strand $n$ along the central axis is calculated as:

$$
s^{mn}(t) = \bigl| \langle \tilde{\mathbf{v}}_1^{m}(t) \rangle_m \cdot (\mathbf{c}^m(t) - \mathbf{c}^n(t))\bigr|, \quad \text{where} \quad \tilde{\mathbf{v}}_1^n(t) =
\begin{cases}
-\hat{\mathbf{v}}_1^n, & \text{when } \hat{\mathbf{v}}_1^m \cdot \hat{\mathbf{v}}_1^n < 0 \\
\hat{\mathbf{v}}_1^n,  & \text{otherwise.}
\end{cases}
$$

__Deviations in axis angles__ are calculated between the unit vectors of the principal axes of the two strands $m$ and $n$ as:

$$
\delta_{\theta}^{mn}(t) = \dfrac{\overbrace{|\arccos(\hat{\mathbf{v}}_1^m(t) \cdot \hat{\mathbf{v}}_1^n(t))}^{\displaystyle \theta^{mn}(t)} - \theta^{mn}(0)|}{\theta^{mn}(0)}.
$$

__Deviations in axis distances__ are calculated using the perpendicular distance between the axes of strands $m$ and $n$ as:

$$
\delta_d^{mn}(t) = \dfrac{ \overbrace{|(\mathbf{c}^m(t) - \mathbf{c}^n(t)) \cdot (\hat{\mathbf{v}}_1^m(t) \times \hat{\mathbf{v}}_1^n(t))|}^{\displaystyle d^{mn}(t)} - d^{mn}(0)}{d^{mn}(0)}.
$$

__Deviations in averaged distances__ are calculated using the distance between the average coordinates of strands $m$ and $n$ as:

$$
\delta_c^{mn}(t) = \dfrac{\overbrace{\| \mathbf{c}^m(t) - \mathbf{c}^n(t)\|}^{\displaystyle c^{mn}(t)} - c^{mn}(0)}{c^{mn}(0)}.
$$

### Section III:

For triple helices, one atom $i$ on each strand is taken to form a triangular cross-section. 

__Deviations in area__ are calculated from the area of the triangle as: 

$$
\delta^{i}_{\text{Area}}(t) = \dfrac{\overbrace{\dfrac{1}{2} \|(\mathbf{p}^{i,2}(t) - \mathbf{p}^{i,1}(t)) \times (\mathbf{p}^{i,3}(t) - \mathbf{p}^{i,1}(t))\|}^{\displaystyle \text{Area}^i(t)} - \text{Area}^i(0)}{\text{Area}^i (0)}.
$$ 

__Deviations in shape__ are calculated from the normalized isoperimetric ratio (IP) as:

$$
\delta^{i}_{\text{Shape}}(t) = \dfrac{\overbrace{\dfrac{4\pi \text{Area}^i(t)}{P^i(t)^2}}^{\displaystyle \text{IP}^i(t)} - \text{IP}^i(0)}{\text{IP}^i(0)},
$$ 
where $P^i(t)$ is the corresponding perimeter.

The first frame is taken as the reference. The reference can be changed by appending a new reference frame to the beginning of the input MD trajectory. All mathematical details are also outlined as comments in the code.

# Acknowledgements

The authors thank the National Institute of Health, National Institute of General Medical Sciences, under award number R35-GM150409 and Advanced Research Computing at Virginia Tech for providing computational resources and technical support.

# References
